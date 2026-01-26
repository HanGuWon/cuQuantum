# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Sequence, List, Tuple

import jax
import jax.numpy as jnp

from .pysrc.context import CudensitymatContext
from .pysrc.operator import Operator
from .pysrc.operator_action_prim import (
    operator_action_prim,
    operator_action_backward_diff_prim,
    OperatorActionPrimitive,
    OperatorActionBackwardDiffPrimitive
)
from .utils import (
    maybe_expand_dim,
    maybe_squeeze_dim,
    get_state_batch_size_and_purity,
    check_and_return_final_batch_size,
)

logger = logging.getLogger("cudensitymat-jax.operator_action")


def operator_action(op: Operator,
                    t: float,
                    state_in_bufs: jax.Array | Sequence[jax.Array],
                    params: jax.Array | None = None,
                    device: jax.Device | None = None,
                    batch_size: int = 1,
                    options: dict = {}
                    ) -> jax.Array | List[jax.Array]:
    """
    Compute the action of an operator on a state.

    Args:
        op: Operator to compute the action of.
        t: The time to compute the operator action at.
        state_in_bufs: Buffers of the input state components.
        params: Callback parameters used to construct the operator.
        device: Device to use for the operator action.
        batch_size: Batch size of the operator action.
        options: Dictionary of options for the operator action. Currently, the available options are names of the gradients attached to the callbacks. See ``example8_gradient_attachment.py`` for an example.

            - ``"op_term_grad_attr"``: Name of the operator term coefficient gradient attached to the coefficient gradient callback. Default is ``"scalar_grad"``.
            - ``"op_prod_grad_attr"``: Name of the operator product coefficient gradient attached to the coefficient gradient callback. Default is ``"scalar_grad"``.
            - ``"base_op_grad_attr"``: Name of the tensor gradient attached to the tensor gradient callback. Default is ``"tensor_grad"``.

    Returns:
        Buffers of the output state components.
    """
    logger.info(f"Calling operator_action")

    # Check if GPU is available and set to GPU 0.
    if device is None:
        gpu_devices = jax.devices('gpu')
        if gpu_devices == []:
            raise RuntimeError("No GPU devices found.")
        device = gpu_devices[0]
        logger.info("Running a single-GPU version of the library and setting device to GPU 0.")

    # Process input arguments.
    if isinstance(state_in_bufs, jax.Array):
        state_in_bufs = (state_in_bufs,)
    else:
        state_in_bufs = tuple(state_in_bufs)

    # Check state shape and maybe expand to a leading batch dimension.
    state_batch_size, purity = get_state_batch_size_and_purity(state_in_bufs, len(op.dims), batch_size)
    batch_size = check_and_return_final_batch_size(state_in_bufs, state_batch_size, op.batch_size, batch_size)

    state_in_bufs = maybe_expand_dim(state_in_bufs, len(op.dims))

    # Prepare library context for forward operator action.
    CudensitymatContext.maybe_create_context(op, device, batch_size, purity)
    
    # Store only the pointer to avoid leaking JAX tracers during transformations
    OperatorActionPrimitive.operator_ptr = op._ptr

    # Obtain the operator action context and assign corresponding attributes.
    op_act_ctx = CudensitymatContext.get_context(op)
    op_act_ctx.num_state_components = len(state_in_bufs)

    # Create lists for other_out attributes.
    other_out_types = []
    other_out_ptrs = []
    other_out_shape_dtypes = []

    other_in_types = []
    other_in_ptrs = []
    
    op_term_coeffs_indices = []
    op_prod_coeffs_indices = []
    base_op_indices = []

    # Extract temporary batched coefficient buffers for operator terms.
    count = 0
    for ptr, buf in zip(op.static_coeffs_ptrs, op.static_coeffs):
        if buf is not None and ptr is not None and ptr not in other_in_ptrs:
            other_in_types.append(4)
            other_in_ptrs.append(ptr)
            op_term_coeffs_indices.append(count)
        count += 1
    for ptr, buf in zip(op.total_coeffs_ptrs, op.total_coeffs):
        if buf is not None and ptr is not None and ptr not in other_out_ptrs:
            other_out_types.append(4)
            other_out_ptrs.append(ptr)
            other_out_shape_dtypes.append(buf)

    # Extract temporary batched coefficient buffers for operator products.
    count = 0
    static_coeffs = [x for op_term in op.op_terms for x in op_term.static_coeffs]
    total_coeffs = [x for op_term in op.op_terms for x in op_term.total_coeffs]
    static_coeffs_ptrs = [x for op_term in op.op_terms for x in op_term.static_coeffs_ptrs]
    total_coeffs_ptrs = [x for op_term in op.op_terms for x in op_term.total_coeffs_ptrs]
    for ptr, buf in zip(static_coeffs_ptrs, static_coeffs):
        if buf is not None and ptr is not None and ptr not in other_in_ptrs:
            other_in_types.append(3)
            other_in_ptrs.append(ptr)
            op_prod_coeffs_indices.append(count)
        count += 1
    for ptr, buf in zip(total_coeffs_ptrs, total_coeffs):
        if buf is not None and ptr is not None and ptr not in other_out_ptrs:
            other_out_types.append(3)
            other_out_ptrs.append(ptr)
            other_out_shape_dtypes.append(buf)

    # Assign certain static attributes to the operator action context.
    base_ops = [base_op for op_term in op.op_terms for op_prod in op_term.op_prods for base_op in op_prod]
    count = 0
    for base_op in base_ops:
        # Use callback being None or not to detect whether data is user-provided or to be allocated.
        if not base_op._to_be_allocated and base_op._ptr is not None:
            other_in_types.append(2 - base_op._is_elementary)  # 1 -> 1, 0 -> 2
            other_in_ptrs.append(base_op._ptr)
            base_op_indices.append(count)
        elif base_op._to_be_allocated and base_op._ptr is not None:
            other_out_types.append(2 - base_op._is_elementary)  # 1 -> 1, 0 -> 2
            other_out_ptrs.append(base_op._ptr)
            shape_dtype = jax.ShapeDtypeStruct(base_op.data.shape, base_op.data.dtype)
            other_out_shape_dtypes.append(shape_dtype)
        count += 1

    # Assign other output attributes to the operator action context.
    op_act_ctx.other_out_shape_dtypes = other_out_shape_dtypes
    op_act_ctx.other_out_types = other_out_types
    op_act_ctx.other_out_ptrs = other_out_ptrs

    op_act_ctx.other_in_types = other_in_types
    op_act_ctx.other_in_ptrs = other_in_ptrs

    op_act_ctx.op_term_coeffs_indices = op_term_coeffs_indices
    op_act_ctx.op_prod_coeffs_indices = op_prod_coeffs_indices
    op_act_ctx.base_op_indices = base_op_indices

    if params is None or params.shape == (0,):
        # Empty params causes a problem when reconstructing params from pointer 
        # in the bindings. We're providing a value here to guard against that.
        params = jnp.array([[0.0]])
    elif params.ndim == 1:
        params = jnp.array([params])
    else:
        if params.dtype != jnp.float64:
            raise ValueError("params must be a float64 array")
        
    op_act_ctx.options = options

    # Invoke operator action.
    state_out_bufs = _operator_action(op, t, state_in_bufs, params)
    state_out_bufs = maybe_squeeze_dim(state_out_bufs, len(op.dims))

    # Process output argument.
    if len(state_out_bufs) == 1:
        state_out_bufs = state_out_bufs[0]

    return state_out_bufs


@jax.custom_vjp
def _operator_action(op: Operator,
                     t: float,
                     state_in_bufs: Tuple[jax.Array, ...],
                     params: jax.Array,
                     ) -> List[jax.Array]:
    """
    Custom VJP rule for operator_action.
    """
    logger.info(f"Calling _operator_action")
    state_out_bufs, _ = _operator_action_fwd(op, t, state_in_bufs, params)
    return state_out_bufs


def _operator_action_fwd(op: Operator,
                         t: float,
                         state_in_bufs: Tuple[jax.Array, ...],
                         params: jax.Array,
                         ) -> Tuple[List[jax.Array], tuple]:
    """
    Forward rule for operator_action.
    """
    logger.info(f"Calling _operator_action_fwd")
    state_out_bufs = operator_action_prim(op, t, state_in_bufs, params)
    return state_out_bufs, (op, t, state_in_bufs, params)


def _operator_action_bwd(res: tuple, state_out_adj_bufs: jax.Array | Sequence[jax.Array]) -> tuple:
    """
    Backward rule for operator_action.

    Args:
        state_out_adj_bufs: Data buffers of the output state adjoint.
    """
    logger.info(f"Calling _operator_action_bwd")

    op, t, state_in_bufs, params = res
    
    # Prepare library context for backward operator action
    # Store only the pointer to avoid leaking JAX tracers during transformations
    OperatorActionBackwardDiffPrimitive.operator_ptr = op._ptr
    op_act_ctx = CudensitymatContext.get_context(op)
    op_act_ctx.create_adjoint_buffers()

    # Process input argument.
    if isinstance(state_out_adj_bufs, jax.Array):
        state_out_adj_bufs = (state_out_adj_bufs,)
    else:
        state_out_adj_bufs = tuple(state_out_adj_bufs)

    if len(state_in_bufs) != len(state_out_adj_bufs):
        raise ValueError("state_in_bufs and state_out_adj_bufs must have the same number of components.")

    params_grad, *state_in_adj_bufs = operator_action_backward_diff_prim(
        op, t, state_in_bufs, state_out_adj_bufs, params)
    
    if len(state_in_adj_bufs) == 1:
        state_in_adj_bufs = state_in_adj_bufs[0]
    
    # Lists used to store pointers to the different quantities, so that we know which gradient
    # has been set already.
    op_term_ptrs = []
    op_prod_ptrs = []
    base_op_ptrs = []

    options = op_act_ctx.options
    op_term_grad_attr = options.get("op_term_grad_attr", "scalar_grad")
    op_prod_grad_attr = options.get("op_prod_grad_attr", "scalar_grad")
    base_op_grad_attr = options.get("base_op_grad_attr", "tensor_grad")

    op_grad = op.copy()
    for i, (op_term, coeff_grad_callback) in enumerate(zip(op_grad.op_terms, op_grad.coeff_grad_callbacks)):
        # We need to set gradient for operator term coefficients only when there is coefficient
        # gradient callback, the coefficient gradient callback has a scalar_grad field, and gradient
        # on this operator term has not been set.
        set_op_term_coeffs_grad = (
            coeff_grad_callback is not None and
            hasattr(coeff_grad_callback.callback, op_term_grad_attr) and
            op_term._ptr not in op_term_ptrs
        )

        if op.static_coeffs[i] is None:  # non-batched
            if set_op_term_coeffs_grad:
                op_grad.coeffs[i] = getattr(coeff_grad_callback.callback, op_term_grad_attr)
                op_term_ptrs.append(op_term._ptr)
            # else:
            #     op_grad.coeffs[i] = 0.0j
        else:  # batched
            if set_op_term_coeffs_grad:
                op_grad.static_coeffs[i] = getattr(coeff_grad_callback.callback, op_term_grad_attr)
                op_term_ptrs.append(op_term._ptr)
            else:
                op_grad.static_coeffs[i] = jnp.zeros(op_grad.static_coeffs[i].shape, dtype=op_grad.static_coeffs[i].dtype)

        for j, (op_prod, coeff_grad_callback_) in enumerate(zip(op_term.op_prods, op_term.coeff_grad_callbacks)):
            # We need to set gradient for operator product coefficients only when there is coefficient
            # gradient callback, the coefficient gradient callback has a scalar_grad field, and gradient
            # on this operator product has not been set.
            op_prod_ptr = tuple([base_op._ptr for base_op in op_prod])
            set_op_prod_coeffs_grad = (
                coeff_grad_callback_ is not None and
                hasattr(coeff_grad_callback_.callback, op_prod_grad_attr) and
                op_prod_ptr not in op_prod_ptrs
            )

            if op_term.static_coeffs[j] is None:  # non-batched
                if set_op_prod_coeffs_grad:
                    op_grad.coeffs[j] = getattr(coeff_grad_callback_.callback, op_prod_grad_attr)
                    op_prod_ptrs.append(op_prod_ptr)
                # else:
                #     op_grad.coeffs[j] = 0.0j
            else:  # batched
                if set_op_prod_coeffs_grad:
                    op_prod_ptrs.append(op_prod_ptr)
                    op_term.static_coeffs[j] = getattr(coeff_grad_callback_.callback, op_prod_grad_attr)
                else:
                    op_term.static_coeffs[j] = jnp.zeros(
                        op_term.static_coeffs[j].shape,
                        dtype=op_term.static_coeffs[j].dtype
                    )

            for base_op in op_prod:
                # We need to set gradient for base operator data only when there is tensor
                # gradient callback, the tensor gradient callback has a tensor_grad field,
                # and gradient on this base operator has not been set.
                set_base_op_data_grad = (
                    base_op.grad_callback is not None and
                    hasattr(base_op.grad_callback.callback, base_op_grad_attr) and
                    base_op._ptr not in base_op_ptrs
                )

                if set_base_op_data_grad:
                    base_op.data = getattr(base_op.grad_callback.callback, base_op_grad_attr)
                    base_op_ptrs.append(base_op._ptr)
                else:
                    base_op.data = jnp.zeros(base_op.data.shape, dtype=base_op.dtype)

    return op_grad, 0.0, state_in_adj_bufs, params_grad


_operator_action.defvjp(_operator_action_fwd, _operator_action_bwd)
