# Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import os
import time
import warnings
import numpy as np
try:
    import pennylane
except ImportError:
    pennylane = None

try:
    from .. import _internal_utils
except ImportError:
    _internal_utils = None
from .backend import Backend
from ..constants import LOGGER_NAME
from .._utils import get_mpi_size


# set up a logger
logger = logging.getLogger(LOGGER_NAME)


class Pennylane(Backend):

    def __init__(self, ngpus, ncpu_threads, precision, *args, identifier=None, **kwargs):
        if pennylane is None:
            raise RuntimeError("pennylane is not installed")
        self.dtype = np.complex64 if precision == "single" else np.complex128
        self.identifier = identifier
        self.ngpus = ngpus
        self.ncpu_threads = ncpu_threads
        self.nqubits = kwargs.pop('nqubits')
        self.version = self.find_version(identifier) 
        self.meta = {}
        self.meta['ncputhreads'] = ncpu_threads

    def find_version(self, identifier):
        if self.ngpus > 1:
            raise ValueError("To specify the number of GPUs for pennylane-lightning-gpu and pennylane-lightning-kokkos, run with MPI support. \
                The total number of MPI processes determines how many GPUs will be used; \
                for example, launching with 4 MPI processes (mpirun -np 4 ...) utilizes 4 GPUs, it should be power of 2.")

        if identifier == "pennylane-lightning-gpu":
            try:
                from pennylane_lightning.lightning_gpu import LightningGPU # version >= 0.33.0
                return LightningGPU.version
            except ImportError:
                raise RuntimeError("PennyLane-Lightning-GPU plugin is not installed")
        elif identifier == "pennylane-lightning-kokkos":
            try:
                from pennylane_lightning.lightning_kokkos import LightningKokkos # version >= 0.33.0
                return LightningKokkos.version
            except ImportError:
                raise RuntimeError("PennyLane-Lightning-Kokkos plugin is not installed")
        elif identifier == "pennylane-lightning-qubit":
            try:
                from pennylane_lightning.lightning_qubit import LightningQubit # version >= 0.33.0
                return LightningQubit.version
            except ImportError as e:
                raise RuntimeError("PennyLane-Lightning plugin is not installed") from e
        else: # identifier == "pennylane"
            return pennylane.__version__

    def preprocess_circuit(self, circuit, *args, **kwargs):
        if _internal_utils is not None:
            _internal_utils.preprocess_circuit(self.identifier, circuit, *args, **kwargs)
        
        self.circuit = circuit
        self.compute_mode = kwargs.pop('compute_mode')
        valid_choices = ['statevector', 'sampling']
        if self.compute_mode not in valid_choices:
            raise ValueError(f"The '{self.compute_mode}' computation mode is not supported for this backend. Supported modes are: {valid_choices}")
        
        nshots = kwargs.get('nshots', 1024)
        t1 = time.perf_counter()
        if self.compute_mode == 'statevector':
            self.device = self._make_device(nshots=None, **kwargs)
        elif self.compute_mode == 'sampling':
            self.device = self._make_device(nshots=nshots, **kwargs)
        t2 = time.perf_counter()
        time_make_device = t2 - t1
        
        self.meta['compute-mode'] = f'{self.compute_mode}()'
        self.meta['make-device time:'] = f'{time_make_device} s'
        logger.info(f'data: {self.meta}')

        pre_data = self.meta
        return pre_data

    def _make_device(self, nshots=None, **kwargs):
        if self.identifier == "pennylane-lightning-gpu":
            if get_mpi_size() > 1: # for new versions, such as versions >= 0.42.0
                dev = pennylane.device("lightning.gpu", wires=self.nqubits, shots=nshots, c_dtype=self.dtype, mpi=True)
            else:
                dev = pennylane.device("lightning.gpu", wires=self.nqubits, shots=nshots, c_dtype=self.dtype)
        elif self.identifier == "pennylane-lightning-kokkos":
            if get_mpi_size() > 1: # for new versions, such as versions >= 0.42.0
                dev = pennylane.device("lightning.kokkos", wires=self.nqubits, shots=nshots, c_dtype=self.dtype, mpi=True)
            else:
                dev = pennylane.device("lightning.kokkos", wires=self.nqubits, shots=nshots, c_dtype=self.dtype) 
        elif self.identifier == "pennylane-lightning-qubit":
            if self.ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {self.identifier}")
            if self.ncpu_threads > 1 and self.ncpu_threads != int(os.environ.get("OMP_NUM_THREADS", "-1")):
                warnings.warn(f"--ncputhreads is ignored, for {self.identifier} please set the env var OMP_NUM_THREADS instead",
                              stacklevel=2)
            dev = pennylane.device("lightning.qubit", wires=self.nqubits, shots=nshots, c_dtype=self.dtype)
        elif self.identifier == "pennylane":
            if self.ngpus != 0:
                raise ValueError(f"cannot specify --ngpus for the backend {self.identifier}")
            if self.dtype == np.complex64:
                raise ValueError("As of version 0.33.0, Pennylane's default.qubit device only supports double precision.")
            dev = pennylane.device("default.qubit", wires=self.nqubits, shots=nshots)
        else:
            raise ValueError(f"the backend {self.identifier} is not recognized")
        
        return dev

    def state_vector_qnode(self):
        @pennylane.qnode(self.device)
        def circuit():
            self.circuit()
            return pennylane.state()
        return circuit()

    def sampling_qnode(self):
        @pennylane.qnode(self.device)
        def circuit():
            self.circuit()
            return pennylane.counts(wires=range(self.nqubits))
        return circuit()

    def run(self, circuit, nshots=1024):
        if self.compute_mode == 'sampling':
            samples = self.sampling_qnode() 
        elif self.compute_mode == 'statevector':
            sv = self.state_vector_qnode() 

        return {'results': None, 'post_results': None, 'run_data': {}}


PnyLightningGpu = functools.partial(Pennylane, identifier='pennylane-lightning-gpu')
PnyLightningCpu = functools.partial(Pennylane, identifier='pennylane-lightning-qubit')
PnyLightningKokkos = functools.partial(Pennylane, identifier='pennylane-lightning-kokkos')
Pny = functools.partial(Pennylane, identifier='pennylane')
