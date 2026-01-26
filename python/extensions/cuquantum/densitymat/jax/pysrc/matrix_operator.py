# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Matrix operator class in cuDensityMat.
"""

import logging
from typing import Tuple

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm
from nvmath.internal import typemaps


@jax.tree_util.register_pytree_node_class
class MatrixOperator:
    """
    PyTree class for cuDensityMat's matrix operator.
    """

    logger = logging.getLogger("cudensitymat-jax.MatrixOperator")

    def __init__(self,
                 data: jax.Array | jax.ShapeDtypeStruct,
                 callback: cudm.WrappedTensorCallback | None = None,
                 grad_callback: cudm.WrappedTensorGradientCallback | None = None
                 ) -> None:
        """
        Initialize a MatrixOperator object.

        Args:
            data: Data specification of the matrix operator. If ``callback`` is ``None``, ``data``
                should be a ``jax.Array``; otherwise, ``data`` should be a ``jax.ShapeDtypeStruct``.
            callback: Forward callback for the matrix operator.
            grad_callback: Gradient callback for the matrix operator.
        """
        if type(data) is object:
            # TODO: data becomes object() during AD tracing. tree_unflatten should be written properly
            # to handle this case.
            pass
        elif callback is not None and not isinstance(data, jax.ShapeDtypeStruct):
            raise RuntimeError("data must be a jax.ShapeDtypeStruct when the data buffer "
                               "is dynamically constructed from a callback.")

        if isinstance(data, (jax.Array, jax.ShapeDtypeStruct)):

            # Set data and batch size.
            if data.ndim % 2 == 0:
                # Expanding to a leading dimension 1 is necessary since we are taking 0 as the batch
                # dimension when passing to ffi_lowering.
                if isinstance(data, jax.Array):
                    self.data = jnp.expand_dims(data, 0)
                    self._to_be_allocated = False
                else:
                    self.data = jnp.zeros((1, *data.shape), dtype=data.dtype)
                    self._to_be_allocated = True
                self.batch_size = 1
            else:  # batched
                if isinstance(data, jax.Array):
                    self.data = data
                    self._to_be_allocated = False
                else:
                    self.data = jnp.zeros(data.shape, dtype=data.dtype)
                    self._to_be_allocated = True
                self.batch_size = data.shape[0]

            # Set other attributes derived from data.
            self.num_modes: int = len(self.data.shape) // 2
            if self.data.shape[-2 * self.num_modes:-self.num_modes] != self.data.shape[-self.num_modes:]:
                raise ValueError("Data must have the same shape on the bra and ket modes.")
            self.mode_extents: Tuple[int, ...] = self.data.shape[-self.num_modes:]
            self.dtype: jnp.dtype = self.data.dtype

        elif type(data) is object:  # data is object() during AD tracing.
            # Dummy variables for derived attributes.
            self.data: object = data
            self.batch_size: int = 1
            self.num_modes: int = 0
            self.mode_extents: Tuple[int, ...] = ()
            self.dtype: jnp.dtype = jnp.dtype(float)
            self._to_be_allocated: bool = False
        else:
            # We don't put object() in the error message (like data can also be object())
            # because that case is in internal JAX implementation.
            raise TypeError("data must be a jax.Array or jax.ShapeDtypeStruct.")

        # Callbacks.
        self.callback: cudm.WrappedTensorCallback | None = callback
        self.grad_callback: cudm.WrappedTensorGradientCallback | None = grad_callback

        self._ptr: int | None = None
        self._is_elementary: int | None = None

    def copy(self) -> "MatrixOperator":
        """
        Copy the matrix operator.
        """
        if self._to_be_allocated and type(self.data) is not object:
            data = jax.ShapeDtypeStruct(self.data.shape, self.data.dtype)
        else:
            data = jnp.copy(self.data)

        mat_op = MatrixOperator(data, self.callback, self.grad_callback)
        mat_op._ptr = self._ptr
        mat_op._is_elementary = self._is_elementary
        mat_op._to_be_allocated = self._to_be_allocated
        return mat_op

    def tree_flatten(self):
        """
        Flatten the matrix operator PyTree.
        """
        children = (self.data, self._ptr, self._is_elementary)
        aux_data = (self.callback, self.grad_callback, self._to_be_allocated)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the matrix operator PyTree.
        """
        data, _ptr, _is_elementary = children
        callback, grad_callback, _to_be_allocated = aux_data

        if _to_be_allocated and type(data) is not object:
            data = jax.ShapeDtypeStruct(data.shape, data.dtype)

        inst = cls(data, callback, grad_callback)
        inst._ptr = _ptr
        inst._is_elementary = _is_elementary
        inst._to_be_allocated = _to_be_allocated
        return inst

    def _create(self, handle):
        """
        Create opaque handle to the matrix operator.
        """
        # NOTE: This is to catch the gradient attachment case where grad_callback is provided
        # but callback is not.
        if self.callback is None and self.grad_callback is not None:
            def f(t, args, storage):
                pass
            callback = cudm.WrappedTensorCallback(f, cudm.CallbackDevice.GPU)
        else:
            callback = self.callback

        # Create opaque handle to the matrix operator.
        if self._ptr is None:
            if self.batch_size == 1:
                self._ptr = cudm.create_matrix_operator_dense_local(
                    handle,
                    self.num_modes,
                    self.mode_extents,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    callback,
                    self.grad_callback,
                )
            else:
                self._ptr = cudm.create_matrix_operator_dense_local_batch(
                    handle,
                    self.num_modes,
                    self.mode_extents,
                    self.batch_size,
                    typemaps.NAME_TO_DATA_TYPE[self.dtype.name],
                    0,  # buffer pointer to be attached in the XLA layer
                    callback,
                    self.grad_callback,
                )

            self.logger.debug(f"Created matrix operator at {hex(self._ptr)}")

        # Set in _create to prevent tracing.
        if self._is_elementary is None:
            self._is_elementary = 0

    def _destroy(self):
        """
        Destroy opaque handle to the matrix operator.
        """
        if self._ptr is not None:
            cudm.destroy_matrix_operator(self._ptr)
            self.logger.debug(f"Destroyed matrix operator at {hex(self._ptr)}")
            self._ptr = None
