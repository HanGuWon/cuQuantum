# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Operator term class in cuDensityMat.
"""

import ctypes
import logging
from typing import List, Tuple, Sequence, Type

import jax
import jax.numpy as jnp

from cuquantum.bindings import cudensitymat as cudm

from .elementary_operator import ElementaryOperator
from .matrix_operator import MatrixOperator


@jax.tree_util.register_pytree_node_class
class OperatorTerm:
    """
    PyTree class for cuDensityMat's operator term.
    """

    logger = logging.getLogger("cudensitymat-jax.OperatorTerm")

    def __init__(self, dims: Sequence[int]) -> None:
        """
        Initialize an OperatorTerm object.

        Args:
            dims: Hilbert space dimensions.
        """
        # Input argument.
        self.dims: Tuple[int, ...] = tuple(dims)

        # Attributes for handling arguments in append.
        self.op_prods: List[Tuple[ElementaryOperator | MatrixOperator, ...]] = []
        self.modes: List[Tuple[int, ...]] = []
        self.conjs: List[Tuple[bool, ...]] = []
        self.duals: List[Tuple[bool, ...]] = []

        self.coeffs: List[float | complex] = []
        self.static_coeffs: List[jax.Array | None] = []
        self.total_coeffs: List[jax.ShapeDtypeStruct | None] = []
        self.coeff_callbacks: List[cudm.WrappedScalarCallback | None] = []
        self.coeff_grad_callbacks: List[cudm.WrappedScalarGradientCallback | None] = []

        self.batch_size: int = 1
        self.dtype: jnp.dtype | None = None

        # Temporary pointers to be replaced by real pointers at execution time.
        self.static_coeffs_ptrs: List[int] = []
        self.total_coeffs_ptrs: List[int] = []
        self._static_coeffs_ptr_objs: List[ctypes.c_short] = []
        self._total_coeffs_ptr_objs: List[ctypes.c_short] = []

        self._batch_sizes: List[int] = []
        self._op_prod_types: List[Type[ElementaryOperator] | Type[MatrixOperator]] = []
        self._ptr: int | None = None

    def copy(self) -> "OperatorTerm":
        """
        Copy the operator term.
        """
        op_term = OperatorTerm(self.dims)
        for (
            op_prod,
            modes,
            conjs,
            duals,
            coeff,
            static_coeffs,
            total_coeffs,
            coeff_callback,
            coeff_grad_callbacks
        ) in zip(
            self.op_prods,
            self.modes,
            self.conjs,
            self.duals,
            self.coeffs,
            self.static_coeffs,
            self.total_coeffs,
            self.coeff_callbacks,
            self.coeff_grad_callbacks
        ):
            # This is due to empty tuple appended to preserve length.
            if conjs == ():
                conjs = None
            
            # This is due to modes set to all Hilbert space modes internally
            # for reference implementation.
            if isinstance(op_prod[0], MatrixOperator):
                modes = None

            op_prod_ = tuple(base_op.copy() for base_op in op_prod)
            static_coeffs_ = jnp.copy(static_coeffs) if static_coeffs is not None else None
            total_coeffs_ = total_coeffs if total_coeffs is not None else None
            op_term.append(op_prod_,
                           modes=modes,
                           conjs=conjs,
                           duals=duals,
                           coeff=coeff,
                           static_coeffs=static_coeffs_,
                           total_coeffs=total_coeffs_,
                           coeff_callback=coeff_callback,
                           coeff_grad_callback=coeff_grad_callbacks)
        
        # Preserve pointer information to maintain PyTree structure consistency.
        # This is critical for custom VJP: the backward output must have the same
        # PyTree structure as the forward input, including pointer values.
        op_term.static_coeffs_ptrs = self.static_coeffs_ptrs.copy()
        op_term.total_coeffs_ptrs = self.total_coeffs_ptrs.copy()
        op_term._ptr = self._ptr
        
        return op_term

    def tree_flatten(self):
        """
        Flatten the operator term into a PyTree.
        """
        children = (self.op_prods, self.static_coeffs, self.static_coeffs_ptrs)
        aux_data = (
            self.dims,
            self.batch_size,
            self.modes,
            self.conjs,
            self.duals,
            self.coeffs,
            self.total_coeffs,
            self.total_coeffs_ptrs,
            self.coeff_callbacks,
            self.coeff_grad_callbacks,
            self.dtype,
            self._batch_sizes,
            self._op_prod_types,
            self._ptr
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the operator term from a PyTree.
        """
        op_prods, static_coeffs, static_coeffs_ptrs = children
        (
            dims,
            batch_size,
            modes,
            conjs,
            duals,
            coeffs,
            total_coeffs,
            total_coeffs_ptrs,
            coeff_callbacks,
            coeff_grad_callbacks,
            dtype,
            _batch_sizes,
            _op_prod_types,
            _ptr
        ) = aux_data

        # Create the instance.
        inst = cls(dims)

        # Populate traced attributes.
        inst.op_prods = op_prods
        inst.static_coeffs = static_coeffs
        inst.static_coeffs_ptrs = static_coeffs_ptrs
        inst.total_coeffs = total_coeffs
        inst.total_coeffs_ptrs = total_coeffs_ptrs

        # Populate static attributes.
        inst.batch_size = batch_size
        inst.modes = modes
        inst.conjs = conjs
        inst.duals = duals
        inst.coeffs = coeffs
        inst.coeff_callbacks = coeff_callbacks
        inst.coeff_grad_callbacks = coeff_grad_callbacks
        inst.dtype = dtype
        inst._batch_sizes = _batch_sizes
        inst._op_prod_types = _op_prod_types
        inst._ptr = _ptr
        return inst

    def _check_dtype(self, op_prod):
        """
        Check if all elementary or matrix operators have the same data type.
        """
        if self.dtype is None:
            # If the data type is not set, set it to the data type of the first operator.
            self.dtype = op_prod[0].dtype
            for op in op_prod[1:]:
                if op.dtype != self.dtype:
                    raise ValueError("All elementary or matrix operators must have the same data type.")
        else:
            # If the data type is set, check if all elementary or matrix operators
            # have the same data type.
            for op in op_prod:
                if op.dtype != self.dtype:
                    raise ValueError("All elementary or matrix operators must have the same data type.")

    def _check_and_append_op_prod_type(self, op_prod):
        """
        Check if all terms in an operator product are of the same type.
        """
        op_prod_type = type(op_prod[0])
        for op in op_prod[1:]:
            if not isinstance(op, op_prod_type):
                raise ValueError("All terms in an operator product must be of the same type.")
        self._op_prod_types.append(op_prod_type)

    def _check_and_append_batch_size(self,
                                     op_prod: Sequence[ElementaryOperator | MatrixOperator],
                                     static_coeffs: jax.Array | None,
                                     total_coeffs: jax.Array | None
                                     ) -> None:
        """
        Check if all terms in an operator product have batch size 1 or N.
        """
        # Extract the batch size for the operator product.
        op_prod_batch_size = max([base_op.batch_size for base_op in op_prod])
        for base_op in op_prod:
            if base_op.batch_size not in (1, op_prod_batch_size):
                raise ValueError("All basic operators in an operator product must have batch size 1 or N.")

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

        batch_size = max(op_prod_batch_size, coeffs_batch_size)
        if self.batch_size == 1:
            self.batch_size = batch_size
        else:
            if batch_size not in (1, self.batch_size):
                raise ValueError("Batch size in this operator product does not match batch size of this operator term.")

    def append(self,
               op_prod: Sequence[ElementaryOperator | MatrixOperator],
               *,
               modes: Sequence[int] | None = None,
               conjs: Sequence[bool] | None = None,
               duals: Sequence[bool] | None = None,
               coeff: float | complex = 1.0,
               static_coeffs: jax.Array | None = None,
               total_coeffs: jax.ShapeDtypeStruct | None = None,
               coeff_callback: cudm.WrappedScalarCallback | None = None,
               coeff_grad_callback: cudm.WrappedScalarGradientCallback | None = None
               ) -> None:
        """
        Append an elementary or matrix product to an operator term.

        Args:
            op_prod: Product of elementary or matrix operators to be appended.
            modes: Modes acted on by the operator product.
            conjs: Conjugations in the operator product. Only applies to MatrixOperators.
            duals: Dualities of the operator product.
            coeff: Coefficient of the operator product.
            static_coeffs: Static batched coefficients of the operator product.
            total_coeffs: Total batched coefficients of the operator product.
            coeff_callback: Forward callback for the coefficients.
            coeff_grad_callback: Gradient callback for the coefficients.
        """
        # Only require total_coeffs when using batched coefficients (static_coeffs provided)
        if coeff_callback is not None and static_coeffs is not None:
            if not isinstance(total_coeffs, jax.ShapeDtypeStruct):
                raise RuntimeError("For dynamically constructed batched coefficients, "
                                   "total_coeffs must be a jax.ShapeDtypeStruct.")

        # Check if all elementary or matrix operators have the same data type.
        self._check_dtype(op_prod)
        self._check_and_append_op_prod_type(op_prod)
        self._check_and_append_batch_size(op_prod, static_coeffs, total_coeffs)

        # Check consistency and append modes, conjs and duals.
        if self._op_prod_types[-1] is ElementaryOperator:
            # Modes have to specified for elementary operators.
            if modes is None:
                raise ValueError("Modes acted on must be specified for elementary operators.")

            # Check all modes are in Hilbert space.
            if not set(modes) <= set(range(len(self.dims))):
                raise ValueError("Modes acted on must be in the Hilbert space, i.e. between 0 and len(self.dims) - 1")

            # Check length of modes acted on are the same as combined number of modes in the operator product.
            if len(modes) != sum([elem_op.num_modes for elem_op in op_prod]):
                raise ValueError(f"Number of modes acted on {len(modes)} does not match combined number of modes in the operator product.")

            # Check mode extents of each elementary operator match corresponding qubit dimensions.
            modes_index = 0
            for elem_op in op_prod:
                if elem_op.mode_extents != tuple(
                    [self.dims[modes[i]] for i in range(modes_index, modes_index + elem_op.num_modes)]
                ):
                    raise ValueError("Mode extents of each elementary operator must match corresponding qubit dimensions.")
                modes_index += elem_op.num_modes

            # Check that matrix conjugations cannot be specified for elementary operators.
            if conjs is not None:
                raise ValueError("Matrix conjugations cannot be specified for elementary operators.")

            # Check that number of duals matches number of modes.
            if duals is None:
                duals = (False,) * len(modes)
            else:
                if len(duals) != len(modes):
                    raise ValueError("Number of duals must match number of modes acted on for elementary operator product.")

            # For elementary operator product, we only need modes and duals.
            self.modes.append(tuple(modes))
            self.conjs.append(())  # empty tuple is appended here to preserve length
            self.duals.append(tuple(duals))

        else:  # matrix operator product
            # Check that mode extents match Hilbert space dimensions.
            for matrix_op in op_prod:
                if matrix_op.mode_extents != self.dims:
                    raise ValueError("Mode extents must match Hilbert space dimensions for matrix operators.")

            # Check that modes acted on cannot be specified for matrix operators.
            if modes is not None:
                raise ValueError("Modes acted on cannot be specified for matrix operators.")

            # Check consistency of conjs.
            if conjs is None:
                conjs = (False,) * len(op_prod)
            else:
                if len(conjs) != len(op_prod):
                    raise ValueError("Number of matrix conjugations must match number of operator products.")

            # Check that number of duals matches number of matrix operators.
            if duals is None:
                duals = (False,) * len(op_prod)
            else:
                if len(duals) != len(op_prod):
                    raise ValueError("Number of duals must match number of matrix operators.")

            # For matrix operator product, we only need conjs and duals.
            self.modes.append(tuple(range(len(self.dims))))  # used in reference implementation during testing
            self.conjs.append(tuple(conjs))
            self.duals.append(tuple(duals))

        # Populate instance attributes.
        self.op_prods.append(tuple(op_prod))
        self.coeffs.append(coeff)
        self.static_coeffs.append(static_coeffs)
        self.total_coeffs.append(total_coeffs)
        self.static_coeffs_ptrs.append(None)
        self.total_coeffs_ptrs.append(None)
        self.coeff_callbacks.append(coeff_callback)
        self.coeff_grad_callbacks.append(coeff_grad_callback)

    def __getitem__(self, index: int) -> Tuple[ElementaryOperator | MatrixOperator, ...]:
        """
        Get an operator product from the operator term.
        """
        return self.op_prods[index]

    def _create(self, handle):
        """
        Create opaque handle to the operator term.
        """
        # Create opaque handle to dependent elementary or matrix operators.
        for op_prod in self.op_prods:
            for elem_op in op_prod:
                elem_op._create(handle)

        # Create opaque handle to the operator term.    
        if self._ptr is None:
            self._ptr = cudm.create_operator_term(
                handle,
                len(self.dims),
                self.dims
            )
            self.logger.debug(f"Created operator term at {hex(self._ptr)}")

            for i in range(len(self.op_prods)):
                if self._op_prod_types[i] is ElementaryOperator:
                    if self._batch_sizes[i] == 1:
                        cudm.operator_term_append_elementary_product(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [elem_op._ptr for elem_op in self.op_prods[i]],
                            self.modes[i],
                            self.duals[i],
                            self.coeffs[i],
                            self.coeff_callbacks[i],
                            self.coeff_grad_callbacks[i]
                        )
                    else:  # batched
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
                            # Create and store object for temporary pointer.
                            total_coeffs_ptr_obj = ctypes.c_short()
                            self._total_coeffs_ptr_objs.append(total_coeffs_ptr_obj)

                            # Extract pointer of the object for temporary total coefficients.
                            total_coeffs_ptr = ctypes.addressof(total_coeffs_ptr_obj) + 1
                            assert total_coeffs_ptr % 2 == 1, "Temporary total coefficients pointer must be an odd number."
                            self.total_coeffs_ptrs[i] = total_coeffs_ptr
                        else:
                            total_coeffs_ptr = 0

                        cudm.operator_term_append_elementary_product_batch(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [elem_op._ptr for elem_op in self.op_prods[i]],
                            self.modes[i],
                            self.duals[i],
                            self._batch_sizes[i],
                            static_coeffs_ptr,
                            total_coeffs_ptr,
                            self.coeff_callbacks[i],
                            self.coeff_grad_callbacks[i],
                        )
                else:  # MatrixOperator
                    if self._batch_sizes[i] == 1:
                        cudm.operator_term_append_matrix_product(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [mat_op._ptr for mat_op in self.op_prods[i]],
                            self.conjs[i],
                            self.duals[i],
                            self.coeffs[i],
                            self.coeff_callbacks[i],
                            self.coeff_grad_callbacks[i]
                        )
                    else:  # batched
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

                        cudm.operator_term_append_matrix_product_batch(
                            handle,
                            self._ptr,
                            len(self.op_prods[i]),
                            [mat_op._ptr for mat_op in self.op_prods[i]],
                            self.conjs[i],
                            self.duals[i],
                            self._batch_sizes[i],
                            static_coeffs_ptr,
                            total_coeffs_ptr,
                            self.coeff_callbacks[i],
                            self.coeff_grad_callbacks[i],
                        )

            self.logger.debug(f"Appended operator products to operator term at {hex(self._ptr)}")

    def _destroy(self):
        """
        Destroy opaque handle to the operator term.
        """
        self._static_coeffs_ptr_objs.clear()
        self._total_coeffs_ptr_objs.clear()

        # Destroy opaque handle to the operator term.
        if self._ptr is not None:
            cudm.destroy_operator_term(self._ptr)
            self.logger.debug(f"Destroyed operator term at {hex(self._ptr)}")
            self._ptr = None
