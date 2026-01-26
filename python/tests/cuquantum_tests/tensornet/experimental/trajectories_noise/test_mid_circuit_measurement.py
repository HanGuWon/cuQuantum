# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectories based simulation of noisy quantum channels.

This test uses TrajectorySim API to simulate a 3-qubit repetition code
with mid-circuit measurement and error correction
"""

import numpy as np
import pytest
from .quantum_channels import QuantumGates, bitflip_channel


np.random.seed(10)

# This test file uses `trajectory_sim` fixture defined in conftest.py


# Mid-circuit measurement is only supported for MPS for now
MID_CIRCUIT_CONFIGS = [
    # n_qubits  channel_method  state_algo  bitflip_p  dtype        init_state   correction_rounds
    (5,         "general",      "mps",      0.2,       "complex64", 0,                3),
    (5,         "general",      "mps",      0.6,       "complex128",0,                3),
    (5,         "general",      "mps",      0.35,      "complex128",1,                2),
    (5,         "unitary",      "mps",      0.20,      "complex128",1,                4),
]
@pytest.mark.parametrize("n_qubits,channel_method,state_algo,bitflip_p,dtype,init_state,correction_rounds", MID_CIRCUIT_CONFIGS)
def test_3qubit_parity(trajectory_sim, n_qubits, channel_method, bitflip_p, init_state, correction_rounds):
    """
    Test 3-qubit error correction with mid-circuit measurement

    Probability of X rotation is `p`
    Probability of I rotation is `1-p`
    """
    init_state = QuantumGates.I[init_state]
    n_trajectories = 300
    # Logical qubits
    n_qubits = 3

    # Probability of getting more than 1 bitflip
    error_prob_one_round = 3 * bitflip_p**2 * (1 - bitflip_p) + bitflip_p**3
    # odd number of channel errors will give us a bit-flip
    # for G(x) = \sum_n^K (K \choose n)p^n (1-p)*{K-n}x^n, the G(-1) flips the sign of even terms.
    # The value of (G(1) - G(-1))/2 is equal to sum of odd terms
    # G(1) = 1
    # G(-1) = (1 - 2 * error_prob_one_round) ** correction_rounds
    error_prob = (1 - (1 - 2 * error_prob_one_round) ** correction_rounds) / 2
    detected_error_prob = 3 * (1 - bitflip_p)**2 * bitflip_p # single bitflip out of three qubits
    prob_an_error_detected = 1 - (1 - detected_error_prob) ** correction_rounds
    print("Eror probability for one error correction round:", error_prob_one_round)
    print("Error probability:", error_prob)
    print("Probability of at least one error being detected:", prob_an_error_detected)

    init_state /= np.square(init_state).sum()
    orth_state = np.array([init_state[1], -init_state[0]])
    init_gate = np.stack((init_state, orth_state)).astype("complex128").T
    channel = bitflip_channel(bitflip_p)
    if channel_method == "general":
        channel.set_general()

    ensemble_probs = []
    num_trajectories_with_error = 0
    for sim in trajectory_sim.iterate_trajectories(n_trajectories):
        error_detected = False
        cx = lambda i, j: sim.apply_gate(
            (j,),
            QuantumGates.X,
            control_modes=(i,),
            control_values=(1,),
        )
        # Encode initial state
        sim.apply_gate((0,), init_gate)
        # init_dm = state.compute_reduced_density_matrix((0, ))
        cx(0, 1)
        cx(0, 2)
        for j in range(correction_rounds):
            # -- Apply noise channel to each qubit
            for i in range(n_qubits):
                sim.apply_channel((i,), channel)

            # -- Apply error correction

            cx(0, 3)
            cx(1, 3)
            cx(1, 4)
            cx(2, 4)

            # -- Measure error syndromes
            sim.capture_state()
            probs = sim.probs((3, 4))
            assert np.allclose(probs.max(), 1)
            syndrome = np.argmax(probs)
            if syndrome != 0 and not error_detected:
                error_detected = True
                num_trajectories_with_error += 1
            if syndrome == 0:
                pass
            elif syndrome == 2:
                sim.apply_gate((0,), QuantumGates.X)
            elif syndrome == 1:
                sim.apply_gate((2,), QuantumGates.X)
            elif syndrome == 3:
                sim.apply_gate((1,), QuantumGates.X)

            # -- reset ancillas
            if syndrome == 0:
                pass
            elif syndrome == 2:
                sim.apply_gate((3,), QuantumGates.X)
            elif syndrome == 1:
                sim.apply_gate((4,), QuantumGates.X)
            elif syndrome == 3:
                sim.apply_gate((3,), QuantumGates.X)
                sim.apply_gate((4,), QuantumGates.X)

            # -- Continue to the next round of error correction

        probs = sim.probs((0, 1, 2))
        ensemble_probs.append(probs)

    fraction_of_trajectories_with_error = num_trajectories_with_error / n_trajectories
    print("Fraction of trajectories with error:", fraction_of_trajectories_with_error)
    print("Expected fraction of trajectories with error:", prob_an_error_detected)

    ensemble_probs = np.stack(ensemble_probs).mean(axis=0)
    print("Final state probs:")
    for i, v in enumerate(ensemble_probs):
        print(i, v)
    state_probs = [ensemble_probs[0], ensemble_probs[-1]]
    sigma_ = 1 / np.sqrt(n_trajectories)
    _T = np.array([[1 - error_prob, error_prob], [error_prob, 1 - error_prob]])
    expected_probs = _T.dot(init_state)
    print("Expected probabilities:", expected_probs)
    assert np.abs(state_probs[1] - expected_probs[1]) < 2.5 * sigma_
    assert np.abs(state_probs[0] - expected_probs[0]) < 2.5 * sigma_
