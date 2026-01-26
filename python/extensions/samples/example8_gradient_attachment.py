# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cuquantum.bindings import cudensitymat as cudm
from cuquantum.densitymat.jax import (
    ElementaryOperator,
    OperatorTerm,
    Operator,
    operator_action,
)

import logging

# Enable logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s [%(levelname)s] %(message)s'
)

def coherent_state(n_levels, alpha):
    """
    Create a coherent state |alpha⟩ using JAX numpy via displacement operator.
    
    |alpha⟩ = D(alpha)|0⟩ where D(alpha) = exp(alpha a† - alpha* a)
    
    Args:
        n_levels: Dimension of the Fock space (number of levels)
        alpha: Complex amplitude of the coherent state
    
    Returns:
        Complex array of shape (n_levels,) representing the coherent state
    """
    # Create annihilation operator a
    a = jnp.diag(jnp.sqrt(jnp.arange(1, n_levels, dtype=jnp.complex128)), k=1)
    
    # Create creation operator a† (a_dag)
    a_dag = a.conj().T
    
    # Compute displacement operator D(α) = exp(α a† - α* a)
    displacement_arg = alpha * a_dag - jnp.conj(alpha) * a
    D = jax.scipy.linalg.expm(displacement_arg)
    
    # Create ground state |0⟩
    ground_state = jnp.zeros(n_levels, dtype=jnp.complex128)
    ground_state = ground_state.at[0].set(1.0)
    
    # Apply displacement operator: |α⟩ = D(α)|0⟩
    coherent = D @ ground_state
    
    return coherent


def build_cuqnt_elementary_operator(arr: jax.Array) -> ElementaryOperator:
    """
    Construct the elementary operator for cuQuantum Python JAX.

    Args:
        arr: The array to construct the elementary operator from.

    Returns:
        The elementary operator.
    """
    def f_grad_callback(t, args, tensor_grad, params_grad):
        # xyz is the random name we assign to the gradient of the elementary operator.
        # NOTE: Latest JAX does not support from_dlpack(..., copy=True). Solution from https://github.com/jax-ml/jax/issues/33790.
        tensor_grad_copy = jax.dlpack.from_dlpack(tensor_grad[:, :, 0], copy=False)
        f_grad_callback.xyz += jnp.array(tensor_grad_copy, copy=True)
    f_grad_callback.xyz = jnp.expand_dims(jnp.zeros_like(arr), 0)
    grad_callback = cudm.WrappedTensorGradientCallback(f_grad_callback, cudm.CallbackDevice.GPU)

    elem_op = ElementaryOperator(arr, grad_callback=grad_callback)
    return elem_op


def main(omega, kappa, alpha0):
    """
    Compute oscillator population using cuQuantum Python JAX.
    """
    # initialize operators, initial state and saving times
    h_key = jax.random.key(41)
    h_data = jax.random.normal(h_key, (dims[0], dims[0]), dtype=jnp.complex128)
    h_data = omega * (h_data + h_data.conj().T) / 2

    h = build_cuqnt_elementary_operator(h_data)

    # Construct operator term for the Hamiltonian
    H1j = OperatorTerm(dims)
    Hm1j = OperatorTerm(dims)
    H1j.append([h], modes=modes)
    Hm1j.append([h], modes=modes)

    l_key = jax.random.key(42)
    l_data = jnp.sqrt(kappa) * jax.random.normal(l_key, (dims[0], dims[0]), dtype=jnp.complex128)

    # extract elementary operators
    l = build_cuqnt_elementary_operator(l_data)
    ld = build_cuqnt_elementary_operator(l_data.conj().T)

    Ls = OperatorTerm(dims)
    Ls.append([l, ld], modes=(0, 0), duals=(False, True), coeff=1.0)
    Ls.append([l, ld], modes=(0, 0), duals=(False, False), coeff=-0.5)
    Ls.append([ld, l], modes=(0, 0), duals=(True, True), coeff=-0.5)

    psi0 = coherent_state(dims[0], alpha0)
    rho0 = jnp.outer(psi0, psi0.conj())
    
    liouvillian = Operator(dims)
    liouvillian.append(Hm1j, dual=False, coeff=-1.0j)
    liouvillian.append(H1j, dual=True, coeff=1.0j)
    liouvillian.append(Ls, dual=False, coeff=1.0)

    # xyz is the random name we assign to the gradient of the elementary operator.
    options = {"base_op_grad_attr": "xyz"}
    rho1 = operator_action(liouvillian, 0.0, rho0, options=options)

    number_op = jnp.diag(jnp.arange(dims[0], dtype=jnp.complex128))
    return (number_op @ rho1).trace().real


if __name__ == "__main__":

    # parameters
    dims = (5,)     # Hilbert space dimension
    modes = (0,)
    omega = 1.0     # frequency
    kappa = 0.1     # decay rate
    alpha0 = 1.0    # initial coherent state amplitude

    # Compute gradient with respect to omega, kappa and alpha
    result = jax.grad(main, argnums=(0, 1, 2))(omega, kappa, alpha0)
    
    print(result[0])
    print(result[1])
    print(result[2])
    print("Finished computing gradients.")
