# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator class in cuDensityMat.
"""

import ctypes
import logging
from typing import List, Tuple, Sequence

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm

from .operator_term import OperatorTerm


@jax.tree_util.register_pytree_node_class
class Operator:
    """
    PyTree class for cuDensityMat's operator.
    """

    logger = logging.getLogger("cudensitymat-jax.Operator")

    def __init__(self, dims: Sequence[int]) -> None:
        """
        Initialize an Operator object.

        Args:
            dims: Hilbert space dimensions.
        """
        self.dims: Tuple[int, ...] = tuple(dims)

        self.op_terms: List[OperatorTerm] = []
        self.duals: List[bool] = []

        self.coeffs: List[float | complex] = []
        self.static_coeffs: List[jax.Array | None] = []
        self.total_coeffs: List[jax.ShapeDtypeStruct | None] = []
        self.coeff_callbacks: List[cudm.WrappedScalarCallback | None] = []
        self.coeff_grad_callbacks: List[cudm.WrappedScalarGradientCallback | None] = []

        self.batch_size: int = 1
        self.dtype: jnp.dtype | None = None

        self.static_coeffs_ptrs: List[int] = []
        self.total_coeffs_ptrs: List[int] = []
        self._static_coeffs_ptr_objs: List[ctypes.c_short] = []
        self._total_coeffs_ptr_objs: List[ctypes.c_short] = []

        self._batch_sizes: List[int] = []
        self._ptr: int | None = None

        self.op_terms_ids: List[int] = []

    def copy(self) -> "Operator":
        """
        Copy the operator.
        """
        op = Operator(self.dims)
        for (
            op_term,
            dual,
            coeff,
            static_coeffs,
            total_coeffs,
            coeff_callback,
            coeff_grad_callback
        ) in zip(
            self.op_terms,
            self.duals,
            self.coeffs,
            self.static_coeffs,
            self.total_coeffs,
            self.coeff_callbacks,
            self.coeff_grad_callbacks
        ):
            op_term_ = op_term.copy()
            static_coeffs_ = jnp.copy(static_coeffs) if static_coeffs is not None else None
            total_coeffs_ = total_coeffs if total_coeffs is not None else None
            op.append(op_term_,
                      dual=dual,
                      coeff=coeff,
                      static_coeffs=static_coeffs_,
                      total_coeffs=total_coeffs_,
                      coeff_callback=coeff_callback,
                      coeff_grad_callback=coeff_grad_callback)
        
        # Preserve pointer information to maintain PyTree structure consistency.
        # This is critical for custom VJP: the backward output must have the same
        # PyTree structure as the forward input, including pointer values.
        op.static_coeffs_ptrs = self.static_coeffs_ptrs.copy()
        op.total_coeffs_ptrs = self.total_coeffs_ptrs.copy()
        op._ptr = self._ptr
        op.op_terms_ids = self.op_terms_ids

        return op

    def tree_flatten(self):
        """
        Flatten the operator PyTree.
        """
        children = (self.op_terms, self.static_coeffs, self.static_coeffs_ptrs)
        aux_data = (
            self.dims,
            self.batch_size,
            self.duals,
            self.coeffs,
            self.total_coeffs,
            self.total_coeffs_ptrs,
            self.coeff_callbacks,
            self.coeff_grad_callbacks,
            self.dtype,
            self._batch_sizes,
            self._ptr,
            self.op_terms_ids
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the operator PyTree.
        """
        op_terms, static_coeffs, static_coeffs_ptrs = children
        (
            dims,
            batch_size,
            duals,
            coeffs,
            total_coeffs,
            total_coeffs_ptrs,
            coeff_callbacks,
            coeff_grad_callbacks,
            dtype,
            batch_sizes,
            ptr,
            op_terms_ids
        ) = aux_data

        inst = cls(dims)

        inst.op_terms = op_terms
        inst.static_coeffs = static_coeffs
        inst.static_coeffs_ptrs = static_coeffs_ptrs
        inst.total_coeffs = total_coeffs
        inst.total_coeffs_ptrs = total_coeffs_ptrs

        inst.batch_size = batch_size
        inst.duals = duals
        inst.coeffs = coeffs
        inst.coeff_callbacks = coeff_callbacks
        inst.coeff_grad_callbacks = coeff_grad_callbacks
        inst.dtype = dtype
        inst._batch_sizes = batch_sizes
        inst._ptr = ptr
        inst.op_terms_ids = op_terms_ids
        return inst

    def _check_dtype(self, op_term: OperatorTerm):
        """
        Check if all operator terms have the same data type.
        """
        if op_term.dtype is not None:  # for empty operator term, skip the check.
            if self.dtype is None:
                # If the data type is not set, set it to the data type of the first operator term.
                self.dtype = op_term.dtype
            else:
                # If the data type is set, check if the operator term has the same data type as the operator.
                if op_term.dtype != self.dtype:
                    raise ValueError("All operator terms must have the same data type.")

    def _check_and_append_batch_size(self,
                                     op_term: OperatorTerm,
                                     static_coeffs: jax.Array | None,
                                     total_coeffs: jax.Array | None
                                     ) -> None:
        """
        Check if all operator terms have batch size 1 or N.
        """
        # Extract the batch sizes of the static and total coefficients.
        if static_coeffs is not None and total_coeffs is None:
            coeffs_batch_size = static_coeffs.shape[0]
        elif static_coeffs is None and total_coeffs is not None:
            coeffs_batch_size = total_coeffs.shape[0]
        elif static_coeffs is not None and total_coeffs is not None:
            if static_coeffs.shape[0] != total_coeffs.shape[0]:
                raise ValueError("Static and total coefficients must have the same batch size if both are provided.")
            coeffs_batch_size = static_coeffs.shape[0]
        else:
            coeffs_batch_size = 1
        self._batch_sizes.append(coeffs_batch_size)

        # Possibly update the batch size of this operator and check consistency.
        batch_size = max(op_term.batch_size, coeffs_batch_size)
        if self.batch_size == 1:
            self.batch_size = batch_size
        else:
            if batch_size not in (1, self.batch_size):
                raise ValueError("Batch size in this operator term does not match batch size of this operator.")

    def append(self,
               op_term: OperatorTerm,
               *,
               dual: bool = False,
               coeff: float | complex = 1.0,
               static_coeffs: jax.Array | None = None,
               total_coeffs: jax.ShapeDtypeStruct | None = None,
               coeff_callback: cudm.WrappedScalarCallback | None = None,
               coeff_grad_callback: cudm.WrappedScalarGradientCallback | None = None
               ) -> None:
        """
        Append an operator term to an operator.

        Args:
            op_term: Operator term to be appended.
            dual: Duality of the operator term.
            coeff: Coefficient of the operator term.
            static_coeffs: Static batched coefficients of the operator term.
            total_coeffs: Total batched coefficients of the operator term.
            coeff_callback: Forward callback for the coefficients.
            coeff_grad_callback: Gradient callback for the coefficient.
        """
        if self._ptr is not None:
            raise RuntimeError("Cannot modify operator after it has been used in an operator action.")

        self.op_terms_ids.append(id(op_term))
    
        # Only require total_coeffs when using batched coefficients (static_coeffs provided)
        if coeff_callback is not None and static_coeffs is not None:
            if not isinstance(total_coeffs, jax.ShapeDtypeStruct):
                raise RuntimeError("For dynamically constructed batched coefficients, "
                                   "total_coeffs must be a jax.ShapeDtypeStruct.")

        # Check if the operator term has the same data type as the operator.
        self._check_dtype(op_term)
        self._check_and_append_batch_size(op_term, static_coeffs, total_coeffs)

        # Populate inst attributes.
        self.op_terms.append(op_term)
        self.duals.append(dual)

        self.coeffs.append(coeff)
        self.static_coeffs.append(static_coeffs)
        self.total_coeffs.append(total_coeffs)
        self.static_coeffs_ptrs.append(None)
        self.total_coeffs_ptrs.append(None)
        self.coeff_callbacks.append(coeff_callback)
        self.coeff_grad_callbacks.append(coeff_grad_callback)

    def __getitem__(self, index: int) -> OperatorTerm:
        """
        Get an operator term from the operator.
        """
        return self.op_terms[index]

    def _create(self, handle):
        """
        Create opaque handle to the operator.
        """
        # Create dependent operator terms.
        first_indices = {}
        for idx, val in enumerate(self.op_terms_ids):
            if val not in first_indices:
                first_indices[val] = idx
        mapping = {i: first_indices[val] for i, val in enumerate(self.op_terms_ids)}
        for index in set(mapping.values()):
            self.op_terms[index]._create(handle)

        # Create the current operator.
        if self._ptr is None:
            self._ptr = cudm.create_operator(
                handle,
                len(self.dims),
                self.dims
            )
            self.logger.debug(f"Created operator at {hex(self._ptr)}")

            for i in range(len(self.op_terms)):
                if self._batch_sizes[i] == 1:
                    cudm.operator_append_term(
                        handle,
                        self._ptr,
                        self.op_terms[mapping[i]]._ptr,
                        self.duals[i],
                        self.coeffs[i],
                        self.coeff_callbacks[i],
                        self.coeff_grad_callbacks[i]
                    )
                else:
                    if self.static_coeffs[i] is not None:
                        # Create and store object for temporary static coefficients.
                        static_coeffs_ptr_obj = ctypes.c_short()
                        self._static_coeffs_ptr_objs.append(static_coeffs_ptr_obj)

                        # Extract pointer of the object for temporary static coefficients.
                        static_coeffs_ptr = ctypes.addressof(static_coeffs_ptr_obj) + 1
                        assert static_coeffs_ptr % 2 == 1, "Temporary static coefficients pointer must be an odd number."
                        self.static_coeffs_ptrs[i] = static_coeffs_ptr
                    else:
                        static_coeffs_ptr = 0

                    if self.total_coeffs[i] is not None:
                        # Create and store object for temporary total coefficients.
                        total_coeffs_ptr_obj = ctypes.c_short()
                        self._total_coeffs_ptr_objs.append(total_coeffs_ptr_obj)

                        # Extract pointer of the object for temporary total coefficients.
                        total_coeffs_ptr = ctypes.addressof(total_coeffs_ptr_obj) + 1
                        assert total_coeffs_ptr % 2 == 1, "Temporary total coefficients pointer must be an odd number."
                        self.total_coeffs_ptrs[i] = total_coeffs_ptr
                    else:
                        total_coeffs_ptr = 0

                    cudm.operator_append_term_batch(
                        handle,
                        self._ptr,
                        self.op_terms[mapping[i]]._ptr,
                        self.duals[i],
                        self._batch_sizes[i],
                        static_coeffs_ptr,
                        total_coeffs_ptr,
                        self.coeff_callbacks[i],
                        self.coeff_grad_callbacks[i],
                    )
            self.logger.debug(f"Appended operator terms to operator at {hex(self._ptr)}")
    
    def _destroy(self):
        """
        Destroy opaque handle to the operator.
        """
        self._static_coeffs_ptr_objs.clear()
        self._total_coeffs_ptr_objs.clear()

        if self._ptr is not None:
            # Destroy the current operator.
            cudm.destroy_operator(self._ptr)
            self.logger.debug(f"Destroyed operator at {hex(self._ptr)}")
            self._ptr = None
