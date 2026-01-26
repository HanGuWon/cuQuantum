# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator action primitive.
"""

import logging
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax.interpreters import mlir

from cuquantum.bindings import cudensitymat as cudm

from .base import BasePrimitive, register_primitive
from .context import CudensitymatContext
from .operator import Operator


logger = logging.getLogger("cudensitymat-jax.operator_action")


class OperatorActionPrimitive(BasePrimitive):
    """
    JAX primitive for operator action.
    """

    name = "operator_action"
    inner_multiple_results = True
    outer_multiple_results = True
    inner_primitive = None
    outer_primitive = None
    operator_ptr = None  # Store only pointer to avoid leaking JAX tracers

    logger = logging.getLogger(f"cudensitymat-jax.OperatorActionPrimitive")

    @staticmethod
    def abstract(t_aval, params_aval, *other_in_buf_avals, op_ptr, other_out_shape_dtypes):
        """
        Abstract evaluation of the inner primitive of operator action.
        """
        OperatorActionPrimitive.logger.info(f"Calling abstract evaluation of the inner primitive")

        # TODO: Other output buffers, to be moved elsewhere.
        other_out_buf_avals = [
            jax.core.ShapedArray(shape_dtype.shape, shape_dtype.dtype)
            for shape_dtype in other_out_shape_dtypes
        ]

        # Get context using the stored pointer
        op_act_ctx = CudensitymatContext._contexts[OperatorActionPrimitive.operator_ptr]

        # Create abstract arrays for the output state buffers.
        num_state_components = op_act_ctx.num_state_components

        state_out_buf_avals = [
            jax.core.ShapedArray(other_in_buf_avals[i].shape, other_in_buf_avals[i].dtype)
            for i in range(num_state_components)
        ]

        # Obtain workspace limit and stream from the device.
        workspace_limit = (
            op_act_ctx.device.memory_stats()['bytes_limit'] -
            op_act_ctx.device.memory_stats()['bytes_in_use']
        )
        stream = op_act_ctx.device.get_stream_for_external_ready_events()

        # Prepare operator action.
        cudm.operator_prepare_action(
            CudensitymatContext._handle,
            op_act_ctx._operator,
            op_act_ctx._state_in,
            op_act_ctx._state_out,
            op_act_ctx._compute_type,
            workspace_limit,
            CudensitymatContext._workspace_desc,
            stream)

        # Query the required buffer size for the workspace.
        required_buffer_size = cudm.workspace_get_memory_size(
            CudensitymatContext._handle,
            CudensitymatContext._workspace_desc,
            cudm.Memspace.DEVICE,
            cudm.WorkspaceKind.WORKSPACE_SCRATCH)

        if required_buffer_size > op_act_ctx._required_buffer_size:
            op_act_ctx._required_buffer_size = required_buffer_size

        # Create abstract workspace array.
        # NOTE: Memory buffers from cudaMalloc is automatically 256-aligned, which is not 
        # the case for JAX. 255 is added to the buffer size to ensure workspace is 256-aligned.
        workspace_aval = jax.core.ShapedArray((op_act_ctx._required_buffer_size + 255,), jnp.uint8)
        return workspace_aval, *state_out_buf_avals, *other_out_buf_avals

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        Abstract evaluation of the outer primitive of operator action.
        """
        OperatorActionPrimitive.logger.info(f"Calling abstract evaluation of the outer primitive")
        _, *out = OperatorActionPrimitive.abstract(*args, **kwargs)
        return out

    @staticmethod
    def lowering(ctx, t, params, *other_in_bufs, op_ptr, other_out_shape_dtypes):
        """
        Lowering rule of the operator action primitive.
        """
        OperatorActionPrimitive.logger.info(f"Calling lowering rule")

        # Get context using the stored pointer
        op_act_ctx = CudensitymatContext._contexts[OperatorActionPrimitive.operator_ptr]

        # Revert indices in input and output states. Note the layout is specified as
        # minor-to-major axis order.
        operand_layouts = [None] * len(ctx.avals_in)
        for i in range(2, len(ctx.avals_in)):  # 2 is for skipping t and params
            # 0 (the batch dimension) is the most major axis in input buffers and also when passed
            # to the cuQuantum library. Other dimensions (Hilbert space modes) need to be reversed.
            layout = tuple(range(1, ctx.avals_in[i].ndim)) + (0,)
            operand_layouts[i] = layout

        result_layouts = [None] * len(ctx.avals_out)
        for i in range(1, len(ctx.avals_out)):  # 1 is for skipping workspace
            # 0 (the batch dimension) is the most major axis in input buffers and also when passed
            # to the cuQuantum library. Other dimensions (Hilbert space modes) need to be reversed.
            layout = tuple(range(1, ctx.avals_out[i].ndim)) + (0,)
            result_layouts[i] = layout

        # Lower to the XLA FFI handler.
        outputs = jax.ffi.ffi_lowering(
            OperatorActionPrimitive.name,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts
        )(
            ctx,
            t,
            params,
            *other_in_bufs,
            other_in_types=mlir.dense_int_elements(op_act_ctx.other_in_types),
            other_in_ptrs=mlir.dense_int_elements(op_act_ctx.other_in_ptrs),
            other_out_types=mlir.dense_int_elements(op_act_ctx.other_out_types),
            other_out_ptrs=mlir.dense_int_elements(op_act_ctx.other_out_ptrs),
            batch_size=op_act_ctx.batch_size,
            num_state_components=op_act_ctx.num_state_components,
            handle=CudensitymatContext._handle,
            operator=op_ptr,
            state_in=op_act_ctx._state_in,
            state_out=op_act_ctx._state_out
        )
        return outputs

    @staticmethod
    def impl(t, params, *other_in_bufs, op_ptr, other_out_shape_dtypes):
        """
        Primal evaluation of the operator action primitive.
        """
        OperatorActionPrimitive.logger.info(f"Calling primal evaluation")

        assert OperatorActionPrimitive.inner_primitive is not None
        _, *out = OperatorActionPrimitive.inner_primitive.bind(
            t,
            params,
            *other_in_bufs,
            op_ptr=op_ptr,
            other_out_shape_dtypes=tuple(other_out_shape_dtypes)
        )
        return out

    @staticmethod
    def batcher(batched_args, batch_dims, **kwargs):
        """
        Batching rule of the operator action primitive.
        """
        OperatorActionPrimitive.logger.info(f"Calling batcher")

        out = OperatorActionPrimitive.outer_primitive.bind(*batched_args, **kwargs)
        out_batched_dims = (0,) * len(out)
        return out, out_batched_dims


register_primitive(OperatorActionPrimitive)


def operator_action_prim(op: Operator,
                         t: float,
                         state_in_bufs: Tuple[jax.Array, ...],
                         params: jax.Array
                         ) -> List[jax.Array]:
    """
    Function wrapper around OperatorActionPrimitive.
    """
    logger.info(f"Calling operator_action_prim")

    # Extract unique batched coefficient buffers for operator terms and operator products.
    op_act_ctx = CudensitymatContext.get_context(op)

    other_in_bufs = []

    # Extract temporary batched coefficient buffers for operator terms.
    for i in op_act_ctx.op_term_coeffs_indices:
        other_in_bufs.append(op.static_coeffs[i])

    # Extract temporary batched coefficient buffers for operator products.
    static_coeffs = [x for op_term in op.op_terms for x in op_term.static_coeffs]
    for i in op_act_ctx.op_prod_coeffs_indices:
        other_in_bufs.append(static_coeffs[i])

    # Assign certain static attributes to the operator action context.
    base_ops = [base_op for op_term in op.op_terms for op_prod in op_term.op_prods for base_op in op_prod]
    for base_op in base_ops:
        if not base_op._to_be_allocated and base_op._ptr is not None:
            other_in_bufs.append(base_op.data)

    out = OperatorActionPrimitive.outer_primitive.bind(
        t,
        params,
        *state_in_bufs,
        *other_in_bufs,
        op_ptr=op._ptr,
        other_out_shape_dtypes=tuple(op_act_ctx.other_out_shape_dtypes)
    )

    state_out_bufs = out[:op_act_ctx.num_state_components]
    return state_out_bufs


class OperatorActionBackwardDiffPrimitive(BasePrimitive):
    """
    JAX primitive for operator action backward differentiation.
    """

    name = "operator_action_backward_diff"
    inner_multiple_results = True
    outer_multiple_results = True
    inner_primitive = None
    outer_primitive = None
    operator_ptr = None  # Store only pointer to avoid leaking JAX tracers

    logger = logging.getLogger("cudensitymat-jax.OperatorActionBackwardDiffPrimitive")

    @staticmethod
    def abstract(t_aval, params_aval, *other_in_buf_avals, op_ptr, other_out_shape_dtypes):
        """
        Abstract evaluation of the inner primitive of operator action backward differentiation.
        """
        OperatorActionBackwardDiffPrimitive.logger.info(f"Calling abstract evaluation of the inner primitive")

        other_out_buf_avals = [
            jax.core.ShapedArray(shape_dtype.shape, shape_dtype.dtype)
            for shape_dtype in other_out_shape_dtypes
        ]

        # Get context using the stored pointer
        op_act_ctx = CudensitymatContext._contexts[OperatorActionBackwardDiffPrimitive.operator_ptr]

        # Obtain number of state components.
        num_state_components = op_act_ctx.num_state_components

        # Create abstract arrays for the output quantities.
        params_grad_aval = jax.core.ShapedArray(params_aval.shape, params_aval.dtype)
        
        # Extract state input adjoint buffer shapes from state input buffers.
        state_in_adj_buf_avals = [
            jax.core.ShapedArray(other_in_buf_avals[i].shape, other_in_buf_avals[i].dtype)
            for i in range(num_state_components)
        ]

        # Obtain workspace limit and stream from the device.
        workspace_limit = (
            op_act_ctx.device.memory_stats()['bytes_limit'] -
            op_act_ctx.device.memory_stats()['bytes_in_use']
        )
        stream = op_act_ctx.device.get_stream_for_external_ready_events()

        # Prepare operator action backward differentiation.
        cudm.operator_prepare_action_backward_diff(
            CudensitymatContext._handle,
            op_act_ctx._operator,
            op_act_ctx._state_in,
            op_act_ctx._state_out_adj,
            op_act_ctx._compute_type,
            workspace_limit,
            CudensitymatContext._workspace_desc,
            stream)

        # Query the required buffer size for the workspace.
        required_buffer_size = cudm.workspace_get_memory_size(
            CudensitymatContext._handle,
            CudensitymatContext._workspace_desc,
            cudm.Memspace.DEVICE,
            cudm.WorkspaceKind.WORKSPACE_SCRATCH)
        if required_buffer_size > op_act_ctx._required_buffer_size:
            op_act_ctx._required_buffer_size = required_buffer_size

        # Create abstract workspace array.
        # NOTE: Memory buffers from cudaMalloc is automatically 256-aligned, which is not 
        # the case for JAX. 255 is added to the buffer size to ensure workspace is 256-aligned.
        workspace_aval = jax.core.ShapedArray((op_act_ctx._required_buffer_size + 255,), jnp.uint8)
        return workspace_aval, params_grad_aval, *state_in_adj_buf_avals, *other_out_buf_avals

    @staticmethod
    def outer_abstract(*args, **kwargs):
        """
        Abstract evaluation of the outer primitive of operator action backward differentiation.
        """
        OperatorActionBackwardDiffPrimitive.logger.info(f"Calling abstract evaluation of the outer primitive")
        _, *out = OperatorActionBackwardDiffPrimitive.abstract(*args, **kwargs)
        return out

    @staticmethod
    def lowering(ctx, t, params, *other_in_bufs, op_ptr, other_out_shape_dtypes):
        """
        Lowering rule of the operator action backward differentiation primitive.
        """
        OperatorActionBackwardDiffPrimitive.logger.info(f"Calling lowering rule")

        # Get context using the stored pointer
        op_act_ctx = CudensitymatContext._contexts[OperatorActionBackwardDiffPrimitive.operator_ptr]

        # Revert indices in input and output states. Note the layout is specified as
        # minor-to-major axis order.
        operand_layouts = [None] * len(ctx.avals_in)
        for i in range(2, len(ctx.avals_in)):  # 2 is for skipping t and params
            # 0 (the batch dimension) is the most major axis in input buffers and also when passed
            # to the cuQuantum library. Other dimensions (Hilbert space modes) need to be reversed.
            layout = tuple(range(1, ctx.avals_in[i].ndim)) + (0,)
            operand_layouts[i] = layout

        result_layouts = [None] * len(ctx.avals_out)
        for i in range(1, len(ctx.avals_out)):  # 1 is for skipping workspace and params_grad
            # 0 (the batch dimension) is the most major axis in output buffers and also when passed
            # to the cuQuantum library. Other dimensions (Hilbert space modes) need to be reversed.
            layout = tuple(range(1, ctx.avals_out[i].ndim)) + (0,)
            result_layouts[i] = layout

        # Lower to the XLA FFI handler.
        outputs = jax.ffi.ffi_lowering(
            OperatorActionBackwardDiffPrimitive.name,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts
        )(
            ctx,
            t,
            params,
            *other_in_bufs,
            other_in_types=mlir.dense_int_elements(op_act_ctx.other_in_types),
            other_in_ptrs=mlir.dense_int_elements(op_act_ctx.other_in_ptrs),
            other_out_types=mlir.dense_int_elements(op_act_ctx.other_out_types),
            other_out_ptrs=mlir.dense_int_elements(op_act_ctx.other_out_ptrs),
            batch_size=op_act_ctx.batch_size,
            num_state_components=op_act_ctx.num_state_components,
            handle=CudensitymatContext._handle,
            operator=op_ptr,
            state_in=op_act_ctx._state_in,
            state_out_adj=op_act_ctx._state_out_adj,
            state_in_adj=op_act_ctx._state_in_adj
        )
        return outputs

    @staticmethod
    def impl(t, params, *other_in_bufs, op_ptr, other_out_shape_dtypes):
        """
        Primal evaluation of the operator action backward differentiation primitive.
        """
        OperatorActionBackwardDiffPrimitive.logger.info(f"Calling primal evaluation")
        assert OperatorActionBackwardDiffPrimitive.inner_primitive is not None
        _, *out = OperatorActionBackwardDiffPrimitive.inner_primitive.bind(
            t, params, *other_in_bufs, op_ptr=op_ptr, other_out_shape_dtypes=tuple(other_out_shape_dtypes))
        return out

    @staticmethod
    def batcher(batched_args, batch_dims, **kwargs):
        """
        Batching rule of the operator action backward differentiation primitive.
        """
        raise NotImplementedError("Batched operator action backward differentiation is not implemented.")


register_primitive(OperatorActionBackwardDiffPrimitive)


def operator_action_backward_diff_prim(op: Operator,
                                       t: float,
                                       state_in_bufs: Tuple[jax.Array, ...],
                                       state_out_adj_bufs: Tuple[jax.Array, ...],
                                       params: jax.Array
                                       ) -> Tuple[jax.Array, ...]:
    """
    Wrapper around the outer primitive of OperatorActionBackwardDiffPrimitive.
    """
    logger.info(f"Calling operator_action_backward_diff_prim")

    # Extract buffers from context.
    op_act_ctx = CudensitymatContext.get_context(op)

    other_in_bufs = []

    # Extract temporary batched coefficient buffers for operator terms.
    for i in op_act_ctx.op_term_coeffs_indices:
        other_in_bufs.append(op.static_coeffs[i])

    # Extract temporary batched coefficient buffers for operator products.
    static_coeffs = [x for op_term in op.op_terms for x in op_term.static_coeffs]
    for i in op_act_ctx.op_prod_coeffs_indices:
        other_in_bufs.append(static_coeffs[i])

    # Assign certain static attributes to the operator action context.
    base_ops = [base_op for op_term in op.op_terms for op_prod in op_term.op_prods for base_op in op_prod]
    for base_op in base_ops:
        if not base_op._to_be_allocated and base_op._ptr is not None:
            other_in_bufs.append(base_op.data)

    out = OperatorActionBackwardDiffPrimitive.outer_primitive.bind(
        t,
        params,
        *state_in_bufs,
        *state_out_adj_bufs,
        *other_in_bufs,
        op_ptr=op._ptr,
        other_out_shape_dtypes=tuple(op_act_ctx.other_out_shape_dtypes)
    )

    params_grad = out[0]
    state_in_adj_bufs = out[1:1 + op_act_ctx.num_state_components]
    return params_grad, *state_in_adj_bufs
