# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using NumPy ndarrays with implicit Einstein summation.

The contraction result is also a NumPy ndarray.
"""
import numpy as np

from cuquantum.tensornet import contract


a = np.ones((3,2))
b = np.ones((2,4))

r = contract("ij,jh", a, b)    # output modes = "hi" (lexicographically sorted in implicit form).
print(r)

n = np.einsum("ij,jh", a, b)
assert np.allclose(r, n), 'Incorrect results for "ij,jh".'
