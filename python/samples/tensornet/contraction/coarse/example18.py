# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating ellipsis broadcasting.
"""
import numpy as np

from cuquantum.tensornet import contract


a = np.random.rand(3,1)
b = np.random.rand(3,3)

# Elementwise product of two matrices.
expr = "...,..."

r = contract(expr, a, b)
s = np.einsum(expr, a, b)
assert np.allclose(r, s), "Incorrect results."
