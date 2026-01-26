# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

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


class ScalarCallbackGetter:
    """
    Utility class that provides a wrapped scalar callback getter.
    """

    @classmethod
    def get_wrapped_callback(cls, dtype, kind=""):
        assert kind in ["", "deriv", "grad", None]
        if jnp.dtype(dtype).kind == "c":
            if kind == "":
                return cls.real_callback
            elif kind == "deriv":
                return cls.real_deriv_callback
            elif kind == "grad":
                return cls.real_grad_callback
        else:
            if kind == "":
                return cls.real_callback
            elif kind == "deriv":
                return cls.real_deriv_callback
            elif kind == "grad":
                return cls.real_grad_callback
        # If callback kind is None, None is returned.


class TrigScalarCallback(ScalarCallbackGetter):
    """
    Trigonometric scalar callback.
    """

    # Scalar callback doesn't have an imaginary part since states might be real.
    def _real_callback(t, params, scalar):
        batch_size = len(scalar)
        scalar[:] = cp.tan(params[0, :batch_size] * t)

    def _real_deriv_callback(t, params, scalar):
        batch_size = len(scalar)
        scalar[:] = t / cp.cos(params[0, :batch_size] * t) ** 2

    def _real_grad_callback(t, params, scalar_grad, params_grad):
        batch_size = scalar_grad.shape[-1]
        params_grad[0, :batch_size] += 2 * (t * scalar_grad / cp.cos(params[0, :batch_size] * t) ** 2).real

    def _complex_callback(t, params, scalar):
        batch_size = len(scalar)
        scalar[:] = cp.tan(params[0, :batch_size] * t) + 1j / cp.tan(params[1, :batch_size] * t)

    def _complex_deriv_callback(t, params, scalar):
        batch_size = len(scalar)
        scalar[:] = t / cp.cos(params[0, :batch_size] * t) ** 2 - 1j * t / cp.sin(params[1, :batch_size] * t) ** 2

    def _complex_grad_callback(t, params, scalar_grad, params_grad):
        batch_size = scalar_grad.shape[-1]
        params_grad[0, :batch_size] += 2 * (t * scalar_grad / cp.cos(params[0, :batch_size] * t) ** 2).real
        # params_grad[1, :batch_size] -= 2j * (t * scalar_grad / cp.sin(params[1, :batch_size] * t) ** 2).real

    real_callback = cudm.WrappedScalarCallback(_real_callback, cudm.CallbackDevice.GPU)
    real_deriv_callback = cudm.WrappedScalarCallback(_real_deriv_callback, cudm.CallbackDevice.GPU)
    real_grad_callback = cudm.WrappedScalarGradientCallback(_real_grad_callback, cudm.CallbackDevice.GPU)
    complex_callback = cudm.WrappedScalarCallback(_complex_callback, cudm.CallbackDevice.GPU)
    complex_deriv_callback = cudm.WrappedScalarCallback(_complex_deriv_callback, cudm.CallbackDevice.GPU)
    complex_grad_callback = cudm.WrappedScalarGradientCallback(_complex_grad_callback, cudm.CallbackDevice.GPU)


@jax.jit
def main():
    time = 0.0
    batch_size = 2 
    state_in = jnp.array(
        jax.random.uniform(key, (batch_size, *space_mode_extents, *space_mode_extents)),
        dtype=dtype)
    jax.debug.print("Defined input state data buffer.")

    # Original data arrays.
    n_data = jnp.asarray(
        jnp.diag(jnp.arange(space_mode_extents[0])),
        dtype=dtype)
    a_data = jnp.asarray(
        jnp.diag(jnp.sqrt(jnp.arange(1, space_mode_extents[1])), k=1),
        dtype=dtype)
    ad_data = jnp.asarray(
        jnp.diag(jnp.sqrt(jnp.arange(1, space_mode_extents[1])), k=-1),
        dtype=dtype)
    jax.debug.print("Defined elementary operator data buffers.")

    n_elem_op = ElementaryOperator(n_data)
    ad_elem_op = ElementaryOperator(ad_data)
    a_elem_op = ElementaryOperator(a_data)
    jax.debug.print("Created elementary operator objects.")

    # Create the Hamiltonian and dissipators.
    H = OperatorTerm(space_mode_extents)
    Ls = OperatorTerm(space_mode_extents)
    jax.debug.print("Constructed operator terms from elementary operators.")

    H.append([n_elem_op], modes=[0], duals=[False], coeff=1.0)
    Ls.append([ad_elem_op, a_elem_op], modes=[1, 1], duals=[False, True], coeff=1.0)
    Ls.append([a_elem_op, ad_elem_op], modes=[1, 1], duals=[False, False], coeff=-0.5)
    Ls.append([ad_elem_op, a_elem_op], modes=[1, 1], duals=[True, True], coeff=-0.5)
    jax.debug.print("Constructed operator terms from elementary operators.")

    # Batched coefficients for the Hamiltonian and dissipators.
    # total_coeffs must be ShapeDtypeStruct (not jnp.array) because they describe output buffers
    # and become primitive parameters which must be hashable.
    static_coeffs = jnp.array([-1.0j, -1.0j], dtype=dtype)
    total_coeffs = jax.ShapeDtypeStruct((batch_size,), dtype)
    static_coeffs1 = jnp.array([1.0j, 1.0j], dtype=dtype)
    total_coeffs1 = jax.ShapeDtypeStruct((batch_size,), dtype)
    static_coeffs2 = jnp.array([1.0, 1.0], dtype=dtype)
    total_coeffs2 = jax.ShapeDtypeStruct((batch_size,), dtype)

    coeff_callback = TrigScalarCallback.get_wrapped_callback(dtype, kind="")

    liouvillian = Operator(space_mode_extents)
    liouvillian.append(H, dual=False, static_coeffs=static_coeffs, total_coeffs=total_coeffs, coeff_callback=coeff_callback)
    liouvillian.append(H, dual=True, static_coeffs=static_coeffs1, total_coeffs=total_coeffs1)
    liouvillian.append(Ls, dual=False, static_coeffs=static_coeffs2, total_coeffs=total_coeffs2)
    jax.debug.print("Constructed operator from operator terms.")

    state_out = operator_action(liouvillian, time, state_in)
    jax.debug.print("Performed operator action on the input state.")

    return state_out


if __name__ == "__main__":

    key = jax.random.key(42)
    space_mode_extents = (3, 5)
    dtype = jnp.complex128

    state_out = main()
    print(f"Finished computation and exit.")
