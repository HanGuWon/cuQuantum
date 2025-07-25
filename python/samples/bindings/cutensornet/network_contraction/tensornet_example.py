# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np

import cuquantum
from cuquantum.bindings import cutensornet as cutn


print("cuTensorNet-vers:", cutn.get_version())
dev = cp.cuda.Device()  # get current device
props = cp.cuda.runtime.getDeviceProperties(dev.id)
print("===== device info ======")
print("GPU-local-id:", dev.id)
print("GPU-name:", props["name"].decode())
print("GPU-clock:", props["clockRate"])
print("GPU-memoryClock:", props["memoryClockRate"])
print("GPU-nSM:", props["multiProcessorCount"])
print("GPU-major:", props["major"])
print("GPU-minor:", props["minor"])
print("========================")

######################################################################################
# Computing: R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}
######################################################################################

print("Include headers and define data types.")

data_type = cuquantum.cudaDataType.CUDA_R_32F
compute_type = cuquantum.ComputeType.COMPUTE_32F
num_inputs = 4

# Create an array of modes
modes_A = [ord(c) for c in ('a','b','c','d','e','f')]
modes_B = [ord(c) for c in ('b','g','h','e','i','j')]
modes_C = [ord(c) for c in ('m','a','g','f','i','k')]
modes_D = [ord(c) for c in ('l','c','h','d','j','m')]
modes_R = [ord(c) for c in ('k','l')]

# Create an array of extents (shapes) for each tensor
dim = 8
extent_A = (dim,) * 6 
extent_B = (dim,) * 6 
extent_C = (dim,) * 6 
extent_D = (dim,) * 6 
extent_R = (dim,) * 2

print("Define network, modes, and extents.")

#################
# Initialize data
#################

A_d = cp.random.random((np.prod(extent_A),), dtype=np.float32)
B_d = cp.random.random((np.prod(extent_B),), dtype=np.float32)
C_d = cp.random.random((np.prod(extent_C),), dtype=np.float32)
D_d = cp.random.random((np.prod(extent_D),), dtype=np.float32)
R_d = cp.zeros((np.prod(extent_R),), dtype=np.float32)
raw_data_in_d = (A_d.data.ptr, B_d.data.ptr, C_d.data.ptr, D_d.data.ptr)

#############
# cuTensorNet
#############

stream = cp.cuda.Stream()
handle = cutn.create()

nmode_A = len(modes_A)
nmode_B = len(modes_B)
nmode_C = len(modes_C)
nmode_D = len(modes_D)
nmode_R = len(modes_R)

###############################
# Create Contraction Descriptor
###############################

modes_in = (modes_A, modes_B, modes_C, modes_D)
extents_in = (extent_A, extent_B, extent_C, extent_D)
num_modes_in = (nmode_A, nmode_B, nmode_C, nmode_D)

# Strides are optional; if no stride (0) is provided, then cuTensorNet assumes a generalized column-major data layout
strides_in = (0, 0, 0, 0)

# Set up the tensor qualifiers for all input tensors
qualifiers_in = np.zeros(num_inputs, dtype=cutn.tensor_qualifiers_dtype)

# Set up tensor network
desc_net = cutn.create_network_descriptor(handle,
    num_inputs, num_modes_in, extents_in, strides_in, modes_in, qualifiers_in,  # inputs
    nmode_R, extent_R, 0, modes_R,  # output
    data_type, compute_type)

print("Initialize the cuTensorNet library and create a network descriptor.")

#####################################################
# Choose workspace limit based on available resources
#####################################################

free_mem, total_mem = dev.mem_info
workspace_limit = int(free_mem * 0.9)

##############################################
# Find "optimal" contraction order and slicing
##############################################

optimizer_config = cutn.create_contraction_optimizer_config(handle)

# Set the value of the partitioner imbalance factor to 30 (if desired)
imbalance_dtype = cutn.contraction_optimizer_config_get_attribute_dtype(
    cutn.ContractionOptimizerConfigAttribute.GRAPH_IMBALANCE_FACTOR)
imbalance_factor = np.asarray((30,), dtype=imbalance_dtype)
cutn.contraction_optimizer_config_set_attribute(
    handle, optimizer_config, cutn.ContractionOptimizerConfigAttribute.GRAPH_IMBALANCE_FACTOR,
    imbalance_factor.ctypes.data, imbalance_factor.dtype.itemsize)

optimizer_info = cutn.create_contraction_optimizer_info(handle, desc_net)

cutn.contraction_optimize(handle, desc_net, optimizer_config, workspace_limit, optimizer_info)

num_slices_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.NUM_SLICES)
num_slices = np.zeros((1,), dtype=num_slices_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.NUM_SLICES,
    num_slices.ctypes.data, num_slices.dtype.itemsize)
num_slices = num_slices.item()

assert num_slices > 0

print("Find an optimized contraction path with cuTensorNet optimizer.")
 
###########################################################
# Initialize all pair-wise contraction plans (for cuTENSOR)
###########################################################

work_desc = cutn.create_workspace_descriptor(handle)
cutn.workspace_compute_contraction_sizes(handle, desc_net, optimizer_info, work_desc)
required_workspace_size = cutn.workspace_get_memory_size(
    handle, work_desc,
    cutn.WorksizePref.MIN,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.SCRATCH)
work = cp.cuda.alloc(required_workspace_size)
cutn.workspace_set_memory(
    handle, work_desc,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.SCRATCH,
    work.ptr, required_workspace_size)
plan = cutn.create_contraction_plan(handle, desc_net, optimizer_info, work_desc)

###################################################################################
# Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
###################################################################################

pref = cutn.create_contraction_autotune_preference(handle)

num_autotuning_iterations = 5 # may be 0
n_iter_dtype = cutn.contraction_autotune_preference_get_attribute_dtype(
    cutn.ContractionAutotunePreferenceAttribute.MAX_ITERATIONS)
num_autotuning_iterations = np.asarray([num_autotuning_iterations], dtype=n_iter_dtype)
cutn.contraction_autotune_preference_set_attribute(
    handle, pref,
    cutn.ContractionAutotunePreferenceAttribute.MAX_ITERATIONS,
    num_autotuning_iterations.ctypes.data, num_autotuning_iterations.dtype.itemsize)

# Modify the plan again to find the best pair-wise contractions
cutn.contraction_autotune(
    handle, plan, raw_data_in_d, R_d.data.ptr,
    work_desc, pref, stream.ptr)

cutn.destroy_contraction_autotune_preference(pref)
 
print("Create a contraction plan for cuTENSOR and optionally auto-tune it.")
 
###########
# Execution
###########

minTimeCUTENSOR = 1e100
num_runs = 3  # to get stable perf results
e1 = cp.cuda.Event()
e2 = cp.cuda.Event()
slice_group = cutn.create_slice_group_from_id_range(handle, 0, num_slices, 1)

for i in range(num_runs):
    # Contract over all slices.
    # A user may choose to parallelize over the slices across multiple devices.
    e1.record()
    cutn.contract_slices(
        handle, plan, raw_data_in_d, R_d.data.ptr, False,
        work_desc, slice_group, stream.ptr)
    e2.record()

    # Synchronize and measure timing
    e2.synchronize()
    time = cp.cuda.get_elapsed_time(e1, e2) / 1000  # ms -> s
    minTimeCUTENSOR = minTimeCUTENSOR if minTimeCUTENSOR < time else time

print("Contract the network, each slice uses the same contraction plan.")

# free up the workspace
del work

# Recall that we set strides to null (0), so the data are in F-contiguous layout
A_d = A_d.reshape(extent_A, order='F')
B_d = B_d.reshape(extent_B, order='F')
C_d = C_d.reshape(extent_C, order='F')
D_d = D_d.reshape(extent_D, order='F')
R_d = R_d.reshape(extent_R, order='F')
path, _ = cuquantum.tensornet.einsum_path("abcdef,bgheij,magfik,lchdjm->kl", A_d, B_d, C_d, D_d)
out = cp.einsum("abcdef,bgheij,magfik,lchdjm->kl", A_d, B_d, C_d, D_d, optimize=path)
if not cp.allclose(out, R_d):
    raise RuntimeError("result is incorrect")
print("Check cuTensorNet result against that of cupy.einsum().")

#######################################################

flops_dtype = cutn.contraction_optimizer_info_get_attribute_dtype(
    cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT)
flops = np.zeros((1,), dtype=flops_dtype)
cutn.contraction_optimizer_info_get_attribute(
    handle, optimizer_info, cutn.ContractionOptimizerInfoAttribute.FLOP_COUNT,
    flops.ctypes.data, flops.dtype.itemsize)
flops = flops.item()

print(f"num_slices: {num_slices}")
print(f"{minTimeCUTENSOR * 1000 / num_slices} ms / slice")
print(f"{flops / 1e9 / minTimeCUTENSOR} GFLOPS/s")

cutn.destroy_slice_group(slice_group)
cutn.destroy_contraction_plan(plan)
cutn.destroy_contraction_optimizer_info(optimizer_info)
cutn.destroy_contraction_optimizer_config(optimizer_config)
cutn.destroy_network_descriptor(desc_net)
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy(handle)

print("Free resource and exit.")
