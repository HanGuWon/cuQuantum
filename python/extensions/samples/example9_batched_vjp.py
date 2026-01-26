# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal example combining batched operator action with VJP (gradient computation).

This demonstrates:
1. Batched states with batch_size density matrices
2. Batched scalar coefficient callbacks with gradient callbacks
3. Per-batch gradients via jax.vjp
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


# Batched scalar coefficient callbacks (handle batch dimension)
# NOTE: The bindings reconstruct params as shape (num_params, batch_size) in Fortran order.
# The C++ code reads dimensions()[1] as numParams, so pass params with shape (batch_size, num_params).
def coeff_callback(t, params, scalar):
    """
    Scalar coefficient callback with batch support.
    scalar has shape (batch_size,), params is reconstructed as (num_params, batch_size).
    Computes: coeff = tan(params[0, :] * t)
    """
    batch_size = len(scalar)
    scalar[:] = cp.tan(params[0, :batch_size] * t)


def coeff_grad_callback(t, params, scalar_grad, params_grad):
    """
    Gradient callback for scalar coefficient w.r.t. params.
    Updates params_grad[0, :batch_size] with gradients.
    """
    batch_size = scalar_grad.shape[-1]
    params_grad[0, :batch_size] += 2 * (
        t * scalar_grad / cp.cos(params[0, :batch_size] * t) ** 2
    ).real


# Wrap callbacks outside jitted scope
wrapped_coeff_callback = cudm.WrappedScalarCallback(coeff_callback, cudm.CallbackDevice.GPU)
wrapped_coeff_grad_callback = cudm.WrappedScalarGradientCallback(coeff_grad_callback, cudm.CallbackDevice.GPU)


@jax.jit
def main():
    """
    Demonstrates batched operator action with per-batch VJP gradients.
    """
    time = 1.1
    
    # Batched params: shape (batch_size, num_params) because C++ reads dimensions()[1] as numParams.
    # The bindings then reconstruct as (num_params, batch_size) in Fortran order.
    # For 1 param and 3 batch elements: shape (3, 1) -> numParams = 1, buffer size = 3
    params = jnp.array([[1.0], [2.0], [3.0]])  # 3 batch elements, 1 param each
    jax.debug.print("Params shape: {shape}", shape=params.shape)
    
    # Batched input state: (batch_size, dim, dim) density matrices
    state_in = jax.random.uniform(key, (batch_size, *space_mode_extents, *space_mode_extents), dtype=jnp.float64)
    state_in = state_in.astype(dtype)
    jax.debug.print("Input state shape: {shape}", shape=state_in.shape)

    # Elementary operators with static data (number operator)
    n_data = jnp.diag(jnp.arange(space_mode_extents[0], dtype=dtype))
    n_elem_op = ElementaryOperator(n_data)
    jax.debug.print("Created number operator.")

    # Build Hamiltonian term: H = N
    H = OperatorTerm(space_mode_extents)
    H.append([n_elem_op], modes=[0], duals=[False], coeff=1.0)

    # Batched coefficients with callbacks
    static_coeffs = jnp.array([-1.0j] * batch_size, dtype=dtype)
    total_coeffs = jax.ShapeDtypeStruct((batch_size,), dtype)
    static_coeffs_dual = jnp.array([1.0j] * batch_size, dtype=dtype)
    total_coeffs_dual = jax.ShapeDtypeStruct((batch_size,), dtype)

    # Build Liouvillian with batched coefficient callbacks
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
    jax.debug.print("Built Liouvillian with batched coefficient callbacks.")

    # Forward pass + VJP function
    state_out, vjp_fn = jax.vjp(operator_action, liouvillian, time, state_in, params)
    jax.debug.print("Computed operator action and obtained VJP function.")

    # Compute gradients using conjugate of output as adjoint
    state_out_adj = jnp.conj(state_out)
    _, _, state_in_adj, params_grad = vjp_fn(state_out_adj)
    jax.debug.print("Computed per-batch gradients.")

    return state_out, params_grad


if __name__ == "__main__":
    # Global parameters
    key = jax.random.key(42)
    batch_size = 3
    space_mode_extents = (4,)
    dtype = jnp.complex128

    state_out, params_grad = main()
    print(f"Output state shape: {state_out.shape}")
    print(f"Params gradient shape: {params_grad.shape}")
    print(f"Params gradient (per batch): {params_grad}")
    print("Finished batched VJP computation successfully!")
