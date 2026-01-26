# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Elementary operator class in cuDensityMat.
"""

import logging
from typing import Tuple

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm
from nvmath.internal import typemaps


@jax.tree_util.register_pytree_node_class
class ElementaryOperator:
    """
    PyTree class for cuDensityMat's elementary operator.
    """

    logger = logging.getLogger("cudensitymat-jax.ElementaryOperator")

    def __init__(self,
                 data: jax.Array | jax.ShapeDtypeStruct,
                 callback: cudm.WrappedTensorCallback | None = None,
                 grad_callback: cudm.WrappedTensorGradientCallback | None = None,
                 diag_offsets: Tuple[int, ...] = ()
                 ) -> None:
        """
        Initialize an ElementaryOperator object.

        Args:
            data: Data specification of the elementary operator. If ``callback`` is ``None``, ``data``
                should be a ``jax.Array``; otherwise, ``data`` should be a ``jax.ShapeDtypeStruct``.
            callback: Forward callback for the elementary operator.
            grad_callback: Gradient callback for the elementary operator.
            diag_offsets: Diagonal offsets of the elementary operator.
        """
        if type(data) is object:
            # TODO: data becomes object() during AD tracing. tree_unflatten should be written properly
            # to handle this case.
            pass
        elif callback is not None and not isinstance(data, jax.ShapeDtypeStruct):
            raise RuntimeError("data must be a jax.ShapeDtypeStruct when the data buffer "
                               "is dynamically constructed from a callback.")

        # Check consistency of tensor data.
        if isinstance(data, (jax.Array, jax.ShapeDtypeStruct)):

            if len(diag_offsets) == 0:  # dense elementary operator

                # Set batch size and data.
                if data.ndim % 2 == 0:
                    # Expanding to a leading dimension 1 is necessary since we are taking 0 as the batch
                    # dimension when passing to ffi_lowering.
                    if isinstance(data, jax.Array):
                        self.data = jnp.expand_dims(data, 0)
                        self._to_be_allocated = False
                    else:  # jax.ShapeDtypeStruct
                        # Due to how JAX handles PyTrees, self.data needs to be a jax.Array.
                        self.data = jnp.zeros((1, *data.shape), data.dtype)
                        self._to_be_allocated = True
                    self.batch_size = 1
                else:  # batched elementary operator
                    # TODO: Batch dimension is assumed to be dimension 0. This constraint could be
                    # relaxed in the future.
                    if isinstance(data, jax.Array):
                        self.data = data
                        self._to_be_allocated = False
                    else:  # jax.ShapeDtypeStruct
                        # Due to how JAX handles PyTrees, self.data needs to be a jax.Array.
                        self.data = jnp.zeros(data.shape, data.dtype)
                        self._to_be_allocated = True
                    self.batch_size = data.shape[0]

                self.sparsity = cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_NONE

            else:  # multidiagonal elementary operator

                # Set batch size and data.
                if data.ndim == 2:
                    # Expanding to a leading dimension 1 is necessary since we are taking 0 as the batch
                    # dimension when passing to ffi_lowering.
                    if isinstance(data, jax.Array):
                        self.data = jnp.expand_dims(data, 0)
                        self._to_be_allocated = False
                    else:  # jax.ShapeDtypeStruct
                        # Due to how JAX handles PyTrees, self.data needs to be a jax.Array.
                        self.data = None
                        self._to_be_allocated = True
                    self.batch_size = 1
                elif data.ndim == 3:
                    # TODO: Batch dimension is assumed to be dimension 0. This constraint could be
                    # relaxed in the future.
                    if isinstance(data, jax.Array):
                        self.data = data
                        self._to_be_allocated = False
                    else:  # jax.ShapeDtypeStruct
                        # Due to how JAX handles PyTrees, self.data needs to be a jax.Array.
                        self.data = None
                        self._to_be_allocated = True
                    self.batch_size = data.shape[0]
                else:
                    raise ValueError("Only single-mode multidiagonal elementary operator is supported.")

                # Check diagonal offsets.
                if len(diag_offsets) != len(set(diag_offsets)):
                    raise ValueError("Diagonal offsets cannot contain duplicate elements.")
                if data.shape[-1] != len(diag_offsets):
                    raise ValueError("Number of columns in data does not match length of diagonal offsets.")

                self.sparsity = cudm.ElementaryOperatorSparsity.OPERATOR_SPARSITY_MULTIDIAGONAL

            # The following attributes are set in the same way for dense and multidiagonal elementary 
            # operators. Also check that bra and ket modes have the same shape.
            self.num_modes = data.ndim // 2
            if len(diag_offsets) == 0:  # only check on dense elementary operators
                bra_modes = self.data.shape[-2 * self.num_modes:-self.num_modes]
                ket_modes = self.data.shape[-self.num_modes:]
                if bra_modes != ket_modes:
                    raise ValueError("Dense elementary operator data must have the same shape on the bra and ket modes.")
            self.mode_extents = self.data.shape[1:self.num_modes + 1]  # skip the batch dimension.
            self.dtype: jnp.dtype = self.data.dtype

        elif type(data) is object:  # data is object() during AD tracing.
            # Dummy variables for derived attributes.
            self.batch_size: int = 1
            self.data: object = data
            self._to_be_allocated: bool = False
            self.num_modes: int = 0
            self.mode_extents: Tuple[int, ...] = ()
            self.dtype: jnp.dtype = jnp.dtype(float)

        else:
            # We don't put object() in the error message (like data can also be object())
            # because that case is in internal JAX implementation.
            raise TypeError("data must be a jax.Array or jax.ShapeDtypeStruct.")

        # Callbacks and diagonal offsets.
        self.callback: cudm.WrappedTensorCallback | None = callback
        self.grad_callback: cudm.WrappedTensorGradientCallback | None = grad_callback
        self.diag_offsets: Tuple[int, ...] = diag_offsets

        self._ptr: int | None = None
        self._is_elementary: int | None = None

    def copy(self) -> "ElementaryOperator":
        """
        Copy the elementary operator.
        """
        if self._to_be_allocated and type(self.data) is not object:
            data = jax.ShapeDtypeStruct(self.data.shape, self.data.dtype)
        else:
            data = jnp.copy(self.data)

        elem_op = ElementaryOperator(data, self.callback, self.grad_callback, self.diag_offsets)
        
        elem_op._ptr = self._ptr
        elem_op._is_elementary = self._is_elementary
        elem_op._to_be_allocated = self._to_be_allocated
        return elem_op

    def tree_flatten(self):
        """
        Flatten the elementary operator PyTree.
        """
        children = (self.data, self._ptr, self._is_elementary)
        aux_data = (self.callback, self.grad_callback, self.diag_offsets, self._to_be_allocated)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the elementary operator PyTree.
        """
        data, _ptr, _is_elementary = children
        callback, grad_callback, diag_offsets, _to_be_allocated = aux_data

        if _to_be_allocated and type(data) is not object:
            data = jax.ShapeDtypeStruct(data.shape, data.dtype)

        inst = cls(data, callback, grad_callback, diag_offsets)
        inst._ptr = _ptr
        inst._is_elementary = _is_elementary
        inst._to_be_allocated = _to_be_allocated
        return inst

    def _create(self, handle):
        """
        Create opaque handle to the elementary operator.
        """
        # NOTE: This is to catch the gradient attachment case where grad_callback is provided
        # but callback is not.
        if self.callback is None and self.grad_callback is not None:
            def f(t, args, storage):
                pass
            callback = cudm.WrappedTensorCallback(f, cudm.CallbackDevice.GPU)
        else:
            callback = self.callback

        # Create opaque handle to the elementary operator.
        if self._ptr is None:
            if self.batch_size == 1:
                self._ptr = cudm.create_elementary_operator(
                    handle,
                    self.num_modes,
                    self.mode_extents,
                    self.sparsity,
                    len(self.diag_offsets),
                    self.diag_offsets,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    callback,
                    self.grad_callback
                )
            else:
                self._ptr = cudm.create_elementary_operator_batch(
                    handle,
                    self.num_modes,
                    self.mode_extents,
                    self.batch_size,
                    self.sparsity,
                    len(self.diag_offsets),
                    self.diag_offsets,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    callback,
                    self.grad_callback,
                )

            self.logger.debug(f"Created elementary operator at {hex(self._ptr)}")

        # Set in _create to prevent tracing.
        if self._is_elementary is None:
            self._is_elementary = 1

    def _destroy(self):
        """
        Destroy opaque handle to the elementary operator.
        """
        if self._ptr is not None:
            cudm.destroy_elementary_operator(self._ptr)
            self.logger.debug(f"Destroyed elementary operator at {hex(self._ptr)}")
            self._ptr = None
