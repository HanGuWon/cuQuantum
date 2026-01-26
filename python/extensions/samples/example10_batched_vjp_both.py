# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal example combining batched operator action with VJP using BOTH:
1. Batched tensor operator (elementary operator with tensor callback)
2. Batched scalar coefficient callback

This demonstrates the most general case where both operators and coefficients
are computed via callbacks and depend on batched parameters.
"""

import cupy as cp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import (
    ElementaryOperator,
    OperatorTerm,
    Operator,
    operator_action
)


# ============================================================================
# Batched tensor callbacks for elementary operator
# storage has shape (mode_extents..., batch_size), params has shape (num_params, batch_size)
# ============================================================================

def n_callback(t, params, storage):
    """
    Batched tensor callback for number-like operator.
    Uses params[0, :] for tensor scaling.
    """
    dim = storage.shape[0]
    batch_size = storage.shape[-1]
    storage[:] = 0.0
    for m in range(dim):
        for n in range(dim):
            # Operator elements depend on params[0, :batch]
            storage[m, n, :] = m * n * cp.tan(params[0, :batch_size] * t)
            if storage.dtype.kind == 'c':
                storage[m, n, :] += 1j * m * n / cp.tan(params[0, :batch_size] * t + 0.1)


def n_grad_callback(t, params, tensor_grad, params_grad):
    """
    Batched tensor gradient callback for number-like operator.
    Gradient w.r.t. params[0, :batch].
    """
    dim = tensor_grad.shape[0]
    batch_size = tensor_grad.shape[-1]
    for m in range(dim):
        for n in range(dim):
            params_grad[0, :batch_size] += 2 * (
                tensor_grad[m, n, :] * (m * n * t / cp.cos(params[0, :batch_size] * t) ** 2)
            ).real
            if tensor_grad.dtype.kind == 'c':
                params_grad[0, :batch_size] += 2 * (
                    tensor_grad[m, n, :] * (-1j * m * n * t / cp.sin(params[0, :batch_size] * t + 0.1) ** 2)
                ).real


def a_callback(t, params, storage):
    """Batched annihilation operator (static)."""
    storage[:] = 0.0
    dim = storage.shape[0]
    for i in range(1, dim):
        storage[i - 1, i, :] = cp.sqrt(i)


def ad_callback(t, params, storage):
    """Batched creation operator (static)."""
    storage[:] = 0.0
    dim = storage.shape[0]
    for i in range(1, dim):
        storage[i, i - 1, :] = cp.sqrt(i)


# ============================================================================
# Batched scalar coefficient callbacks
# scalar has shape (batch_size,), params has shape (num_params, batch_size)
# ============================================================================

def coeff_callback(t, params, scalar):
    """
    Batched scalar coefficient callback.
    Uses params[1, :] for coefficient scaling (separate from tensor params).
    """
    batch_size = len(scalar)
    # Use second param row for coefficient
    scalar[:] = cp.sin(params[1, :batch_size] * t)


def coeff_grad_callback(t, params, scalar_grad, params_grad):
    """
    Gradient callback for scalar coefficient w.r.t. params.
    Updates params_grad[1, :batch_size].
    """
    batch_size = scalar_grad.shape[-1]
    # Gradient of sin(params[1,:] * t) w.r.t. params[1,:] is t * cos(params[1,:] * t)
    params_grad[1, :batch_size] += 2 * (
        t * scalar_grad * cp.cos(params[1, :batch_size] * t)
    ).real


# ============================================================================
# Wrap callbacks outside jitted scope
# ============================================================================
n_wrapped_callback = cudm.WrappedTensorCallback(n_callback, cudm.CallbackDevice.GPU)
n_wrapped_grad_callback = cudm.WrappedTensorGradientCallback(n_grad_callback, cudm.CallbackDevice.GPU)
a_wrapped_callback = cudm.WrappedTensorCallback(a_callback, cudm.CallbackDevice.GPU)
ad_wrapped_callback = cudm.WrappedTensorCallback(ad_callback, cudm.CallbackDevice.GPU)

wrapped_coeff_callback = cudm.WrappedScalarCallback(coeff_callback, cudm.CallbackDevice.GPU)
wrapped_coeff_grad_callback = cudm.WrappedScalarGradientCallback(coeff_grad_callback, cudm.CallbackDevice.GPU)


@jax.jit
def main():
    """
    Demonstrates batched operator action with BOTH batched tensor operator
    AND batched scalar coefficient, with per-batch VJP gradients.
    """
    time = 1.1
    
    # Batched params: shape (batch_size, num_params)
    # params[:, 0] used by tensor callback, params[:, 1] used by coefficient callback
    # For 2 params and 3 batch elements: shape (3, 2) -> numParams = 2
    params = jnp.array([
        [1.0, 0.5],   # batch 0: tensor param, coeff param
        [2.0, 1.0],   # batch 1
        [3.0, 1.5],   # batch 2
    ])
    jax.debug.print("Params shape: {shape}", shape=params.shape)
    
    # Batched input state: (batch_size, dim, dim) density matrices
    state_in = jax.random.uniform(key, (batch_size, *space_mode_extents, *space_mode_extents), dtype=jnp.float64)
    state_in = state_in.astype(dtype)
    jax.debug.print("Input state shape: {shape}", shape=state_in.shape)

    # Elementary operators with batched tensor callbacks
    # ShapeDtypeStruct uses C-order (batch first), callback receives F-order (batch last)
    n_data_empty = jax.ShapeDtypeStruct((batch_size, space_mode_extents[0], space_mode_extents[0]), dtype)
    a_data_empty = jax.ShapeDtypeStruct((batch_size, space_mode_extents[1], space_mode_extents[1]), dtype)
    ad_data_empty = jax.ShapeDtypeStruct((batch_size, space_mode_extents[1], space_mode_extents[1]), dtype)

    n_elem_op = ElementaryOperator(n_data_empty, callback=n_wrapped_callback, grad_callback=n_wrapped_grad_callback)
    a_elem_op = ElementaryOperator(a_data_empty, callback=a_wrapped_callback)
    ad_elem_op = ElementaryOperator(ad_data_empty, callback=ad_wrapped_callback)
    jax.debug.print("Created elementary operators with batched tensor callbacks.")

    # Build Hamiltonian term: H = N (batched tensor operator)
    H = OperatorTerm(space_mode_extents)
    H.append([n_elem_op], modes=[0], duals=[False], coeff=1.0)

    # Build dissipator: L† a L - 1/2 {L† L, ρ} pattern
    Ls = OperatorTerm(space_mode_extents)
    Ls.append([ad_elem_op, a_elem_op], modes=[1, 1], duals=[False, True], coeff=1.0)
    Ls.append([a_elem_op, ad_elem_op], modes=[1, 1], duals=[False, False], coeff=-0.5)
    Ls.append([ad_elem_op, a_elem_op], modes=[1, 1], duals=[True, True], coeff=-0.5)
    jax.debug.print("Constructed operator terms.")

    # Batched coefficients for Liouvillian terms
    static_coeffs = jnp.array([-1.0j] * batch_size, dtype=dtype)
    total_coeffs = jax.ShapeDtypeStruct((batch_size,), dtype)
    static_coeffs_dual = jnp.array([1.0j] * batch_size, dtype=dtype)
    total_coeffs_dual = jax.ShapeDtypeStruct((batch_size,), dtype)
    static_coeffs_diss = jnp.array([1.0] * batch_size, dtype=dtype)
    total_coeffs_diss = jax.ShapeDtypeStruct((batch_size,), dtype)

    # Build Liouvillian with:
    # - Batched tensor operators (from H and Ls)
    # - Batched scalar coefficient callback (on first term)
    liouvillian = Operator(space_mode_extents)
    liouvillian.append(
        H, dual=False,
        static_coeffs=static_coeffs,
        total_coeffs=total_coeffs,
        coeff_callback=wrapped_coeff_callback,
        coeff_grad_callback=wrapped_coeff_grad_callback
    )
    liouvillian.append(
        H, dual=True,
        static_coeffs=static_coeffs_dual,
        total_coeffs=total_coeffs_dual
    )
    liouvillian.append(
        Ls, dual=False,
        static_coeffs=static_coeffs_diss,
        total_coeffs=total_coeffs_diss
    )
    jax.debug.print("Built Liouvillian with batched tensor operators AND batched coefficient callback.")

    # Forward pass + VJP function
    state_out, vjp_fn = jax.vjp(operator_action, liouvillian, time, state_in, params)
    jax.debug.print("Computed operator action and obtained VJP function.")

    # Compute gradients using conjugate of output as adjoint
    state_out_adj = jnp.conj(state_out)
    _, _, state_in_adj, params_grad = vjp_fn(state_out_adj)
    jax.debug.print("Computed per-batch gradients for both tensor and coefficient params.")

    return state_out, params_grad


if __name__ == "__main__":
    # Global parameters
    key = jax.random.key(42)
    batch_size = 3
    space_mode_extents = (3, 5)
    dtype = jnp.complex128

    state_out, params_grad = main()
    print(f"Output state shape: {state_out.shape}")
    print(f"Params gradient shape: {params_grad.shape}")
    print(f"Params gradient (per batch):")
    print(f"  - Tensor param grad (params[:, 0]): {params_grad[:, 0]}")
    print(f"  - Coeff param grad (params[:, 1]):  {params_grad[:, 1]}")
    print("Finished batched VJP with both batched operator AND coefficient successfully!")
