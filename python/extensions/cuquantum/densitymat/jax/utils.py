# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions for cuQuantum Python JAX.
"""

from jax.interpreters.batching import BatchTracer

from cuquantum.bindings import cudensitymat as cudm


def maybe_expand_dim(bufs, ndim):
    """
    Expand to a leading batch dimension when jax.vmap is applied or when the state is non-batched.
    """
    if isinstance(bufs[0], BatchTracer) or len(bufs[0].shape) in [ndim, 2 * ndim]:
        bufs = tuple([buf.reshape((1, *buf.shape)) for buf in bufs])
    return bufs


def maybe_squeeze_dim(bufs, ndim):
    """
    Squeeze the leading batch dimension when jax.vmap is applied or when the state is non-batched.
    """
    if isinstance(bufs[0], BatchTracer) or (
        len(bufs[0].shape) in [ndim + 1, 2 * ndim + 1] and bufs[0].shape[0] == 1
    ):
        bufs = tuple([buf.reshape(*buf.shape[1:]) for buf in bufs])
    return bufs


def get_state_batch_size_and_purity(state_in_bufs, ndim, default_batch_size):
    """
    Get the batch size and purity of the state.
    """
    # First check all state buffer shapes.
    shape = state_in_bufs[0].shape
    for buf in state_in_bufs[1:]:
        if buf.shape != shape:
            raise ValueError("All input state buffers must have the same shape.")
    if len(shape) not in [ndim, 2 * ndim, ndim + 1, 2 * ndim + 1]:
        raise ValueError("The dimensions of the input state do not match the dimensions of the operator.")

    # Extract batch size based on the four values ndim, 2 * ndim, ndim + 1, 2 * ndim + 1.
    if isinstance(state_in_bufs[0], BatchTracer):
        batch_size = default_batch_size
    else:
        if len(shape) % ndim == 0:
            batch_size = 1
        else:  # ndim + 1 or 2 * ndim + 1
            batch_size = shape[0]

    # Extract purity based on the four values ndim, 2 * ndim, ndim + 1, 2 * ndim + 1.
    if len(shape) // ndim == 1:
        purity = cudm.StatePurity.PURE
    else:  # 2 * ndim or 2 * ndim + 1
        purity = cudm.StatePurity.MIXED

    return batch_size, purity


def check_and_return_final_batch_size(state_in_bufs, state_batch_size, op_batch_size, default_batch_size):
    """
    Check and return the final batch size.
    """
    if isinstance(state_in_bufs[0], BatchTracer):
        return default_batch_size
    else:
        if len({1, state_batch_size, op_batch_size}) > 2:  # the set should be either {1} or {1, N}
            raise ValueError("The batch size of the input state does not match the batch size of the operator.")
        return max(state_batch_size, op_batch_size)
