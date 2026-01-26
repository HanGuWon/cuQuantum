# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

import cuquantum.bindings.cupauliprop as cupp
from ._internal import typemaps
from ._internal.utils import register_finalizer

if TYPE_CHECKING:
    from .handles import LibraryHandle

__all__ = ["QuantumOperator", "PauliNoiseChannel", "PauliRotationGate", "CliffordGate", "AmplitudeDampingChannel"]


class _QuantumOperator(ABC):
    """Abstract base class for quantum operators."""
    
    _ptr: int | None
    _library_handle: "LibraryHandle | None"
    
    def __init__(self):
        self._ptr = None
        self._library_handle = None

    def __int__(self) -> int:
        """Return the underlying C pointer as an integer, or 0 if not initialized."""
        return self._ptr if self._ptr is not None else 0

    @abstractmethod
    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        """
        Return the C API create function and its arguments (excluding library handle).
        
        Returns:
            A tuple of (create_function, args) where create_function is called as
            create_function(library_handle, *args).
        """
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable string representation of the operator."""
        ...

    def _maybe_initialize(self, library_handle: "LibraryHandle") -> None:
        """Initialize the operator with the given library handle if not already initialized."""
        if self._ptr is not None:
            if self._library_handle is not library_handle:
                raise ValueError(f"This {self.__class__.__name__} has already been initialized with a different library handle.")
        else:
            self._library_handle = library_handle
            create_func, args = self._get_create_args()
            self._ptr = create_func(int(self._library_handle), *args)
            library_handle.logger.debug(f"C API {create_func.__name__} returned ptr={self._ptr}")
            library_handle.logger.info(f"QuantumOperator initialized: {self}")
            self._finalizer = register_finalizer(self, cupp.destroy_operator, self._ptr, library_handle.logger, f"QuantumOperator({self.__class__.__name__})") 

class PauliNoiseChannel(_QuantumOperator):
    """A Pauli noise channel acting on 1 or 2 qubits."""
    
    @staticmethod
    def _build_noise_paulis(num_qubits: int) -> tuple[str, ...]:
        """Build the reference Pauli ordering from typemaps for consistency with bindings."""
        paulis = []
        for i in range(4**num_qubits):
            if num_qubits == 1:
                paulis.append(typemaps.PAULI_MAP_INV[i])
            else:  # num_qubits == 2
                # C API convention: prob[i] corresponds to PauliKind(i%4), PauliKind(i/4)
                paulis.append(f"{typemaps.PAULI_MAP_INV[i % 4]}{typemaps.PAULI_MAP_INV[i // 4]}")
        return tuple(paulis)

    # Reference Pauli orderings for single and two-qubit channels (derived from bindings)
    _SINGLE_QUBIT_PAULIS = _build_noise_paulis.__func__(1)
    _TWO_QUBIT_PAULIS = _build_noise_paulis.__func__(2)

    def __init__(self, qubit_indices: Sequence[int], noise_probabilities: Mapping[str, float]):
        """
        Create a Pauli noise channel.

        Args:
            qubit_indices: The qubit indices the channel acts on (1 or 2 qubits).
            noise_probabilities: A dictionary mapping Pauli strings to their probabilities.
                Pauli strings not present in the dictionary are assumed to have zero probability.
                For single-qubit channels, valid keys are "I", "X", "Y", "Z".
                For two-qubit channels, valid keys are "II", "XI", "YI", "ZI", "IX", etc.
        """
        super().__init__()
        self._qubit_indices: Sequence[int] = qubit_indices
        self._num_qubits: int = len(qubit_indices)
        if self._num_qubits not in (1, 2):
            raise ValueError(f"Number of qubits must be 1 or 2, got {self._num_qubits}")
        # Build probabilities in reference order, defaulting to 0.0 for unspecified Paulis
        self._noise_probabilities = tuple(
            noise_probabilities.get(pauli, 0.0) for pauli in self.noise_paulis
        )

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        return cupp.create_pauli_noise_channel_operator, (self._num_qubits, self._qubit_indices, self._noise_probabilities)

    @property
    def qubit_indices(self) -> Sequence[int]:
        """The qubit indices this channel acts on."""
        return self._qubit_indices

    @property
    def noise_probabilities(self) -> tuple[float, ...]:
        """The noise probabilities in reference Pauli order, see :attr:`noise_paulis` for the corresponding Pauli strings."""
        return self._noise_probabilities
    
    @property
    def noise_paulis(self) -> tuple[str, ...]:
        """The Pauli strings in reference order corresponding to each probability, see :attr:`noise_probabilities` for the corresponding probabilities."""
        return self._SINGLE_QUBIT_PAULIS if self._num_qubits == 1 else self._TWO_QUBIT_PAULIS

    def __str__(self) -> str:
        # Show only non-zero probabilities for readability
        nonzero = {p: prob for p, prob in zip(self.noise_paulis, self.noise_probabilities) if prob != 0.0}
        return f"PauliNoiseChannel(qubit indices={list(self._qubit_indices)}, noise probabilities={nonzero})"


class PauliRotationGate(_QuantumOperator):
    """A Pauli rotation gate exp(-i * angle/2 * P) where P is a Pauli string."""
    
    def __init__(self, angle: float, pauli_string: str | Sequence[str], qubit_indices: None | Sequence[int]):
        super().__init__()
        self.angle: float = angle
        self._num_qubits: int = len(pauli_string)
        self._pauli_string: Sequence[str] = list(pauli_string) if isinstance(pauli_string, str) else pauli_string
        self._pauli_string_enums: Sequence[int] = [typemaps.PAULI_MAP[pauli] for pauli in self.pauli_string]
        self._qubit_indices: None | Sequence[int] = qubit_indices
    
    @property
    def num_qubits(self) -> int:
        """The number of qubits this gate acts on."""
        return self._num_qubits
    
    @property
    def pauli_string(self) -> Sequence[str]:
        """The Pauli string defining the rotation axis (e.g., ['X', 'Y', 'Z'])."""
        return self._pauli_string
    
    @property
    def qubit_indices(self) -> Sequence[int]:
        """The qubit indices this gate acts on."""
        return self._qubit_indices if self._qubit_indices is not None else list(range(self._num_qubits))

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        return cupp.create_pauli_rotation_gate_operator, (
            self.angle,
            self.num_qubits,
            self._qubit_indices if self._qubit_indices else 0,
            self._pauli_string_enums
        )

    def __str__(self) -> str:
        return f"PauliRotationGate(angle={self.angle}, pauli string={self.pauli_string}, qubit indices={self.qubit_indices})"

class CliffordGate(_QuantumOperator):
    """A Clifford gate (I, X, Y, Z, H, S, CX, CY, CZ, SWAP, iSWAP, etc.)."""
    
    # Supported Clifford gate names (case-insensitive)
    SUPPORTED_GATES: frozenset[str] = frozenset(typemaps.CLIFFORD_MAP.keys())
    
    def __init__(self, name: str, qubit_indices: Sequence[int]):
        super().__init__()
        if name.upper() not in self.SUPPORTED_GATES:
            raise ValueError(
                f"Unsupported Clifford gate '{name}'. "
                f"Supported gates: {sorted(self.SUPPORTED_GATES)}"
            )
        self._name: str = name
        self._qubit_indices: Sequence[int] = qubit_indices

    @property
    def name(self) -> str:
        """The name of the Clifford gate (e.g., 'H', 'CX', 'SWAP')."""
        return self._name
    
    @property
    def qubit_indices(self) -> Sequence[int]:
        """The qubit indices this gate acts on."""
        return self._qubit_indices

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        return cupp.create_clifford_gate_operator, (typemaps.CLIFFORD_MAP[self.name], self._qubit_indices)

    def __str__(self) -> str:
        return f"CliffordGate(which_clifford='{self.name}', qubit_indices={list(self._qubit_indices)})"


class AmplitudeDampingChannel(_QuantumOperator):
    """An amplitude damping channel with damping and excitation probabilities."""
    
    def __init__(self, damping_probability: float, excitation_probability: float, qubit_index: int):
        super().__init__()
        self._damping_probability: float = damping_probability
        self._excitation_probability: float = excitation_probability
        self._qubit_index: int = qubit_index

    @property
    def damping_probability(self) -> float:
        """The damping probability."""
        return self._damping_probability
    
    @property
    def excitation_probability(self) -> float:
        """The excitation probability."""
        return self._excitation_probability
    
    @property
    def qubit_index(self) -> int:
        """The qubit index this channel acts on."""
        return self._qubit_index

    def _get_create_args(self) -> tuple[Callable[..., int], tuple[Any, ...]]:
        return cupp.create_amplitude_damping_channel_operator, (self.qubit_index, self.damping_probability, self.excitation_probability)

    def __str__(self) -> str:
        return f"AmplitudeDampingChannel(damping probability={self.damping_probability}, excitation probability={self.excitation_probability}, qubit index={self.qubit_index})"

QuantumOperator = PauliNoiseChannel | PauliRotationGate | CliffordGate | AmplitudeDampingChannel