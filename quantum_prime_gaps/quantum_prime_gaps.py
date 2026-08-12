# Copyright 2026 Otis Ranson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encode the prime gap sequence onto qubits and read its spectrum through a QFT.

Pipeline: take the first 50 primes, compute the 49 gaps between consecutive
primes, and normalize them to [0, pi] rotation angles. A small qubit register
(7 by default -- the same `--qubits` register size the prediction pathway below
uses) is repeatedly re-loaded with `Ry` rotations from successive
chunks of that angle sequence, with a ring of entangling `CX` gates between
chunks -- "data re-uploading" (Perez-Salinas et al., 2020), the standard way
to angle-encode a classical sequence longer than the qubit count into a fixed
register. A Quantum Fourier Transform is then applied to the fully loaded
register, and the resulting amplitude landscape (the Born-rule probabilities
of the final statevector) is read out as the "frequency portrait" of the gap
sequence.

Two things are checkable without touching hardware, which is the point of
running this before wiring up prediction:

1. The 50 hardcoded primes are independently re-derived with a sieve, so a
   typo in the literal list would be caught immediately.
2. The whole circuit (rotations, entanglers, QFT) is re-implemented from
   scratch as dense linear algebra in plain numpy, with no dependency on
   Qiskit's simulator, and asserted to match Qiskit's `Statevector` output to
   floating point precision. That confirms the reported amplitude landscape
   really is what the circuit computes, not an artifact of one library's
   internals.

A third, softer check compares the real (ordered) gap sequence's amplitude
landscape against landscapes from random shuffles of the same 49 gap values.
Re-uploading is order-sensitive -- each chunk's rotations and the entangler
that follows depend on which values land together -- so if the circuit is
doing something with the sequence's order rather than just its value
histogram, the real sequence's entropy should differ, on average, from
shuffled controls built from the same numbers. This is reported as evidence,
not asserted as pass/fail.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector

matplotlib.use("Agg")

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "prime"

# Notable events for this run (top_k substitutions, hardware skip/execute decisions,
# hard-check failures, any captured warnings) -- collected here so `write_results_report`
# can include them verbatim without every function needing to know about the report.
RUN_EVENTS: list[str] = []


def _log_event(message: str) -> None:
    RUN_EVENTS.append(message)
    print(message)


FIRST_50_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
]


def sieve_primes(count: int) -> list[int]:
    """Independently derive the first `count` primes with a sieve of Eratosthenes,
    to cross-check against the hardcoded FIRST_50_PRIMES list."""
    limit = 10
    while True:
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False
        primes = [i for i, prime in enumerate(is_prime) if prime]
        if len(primes) >= count:
            return primes[:count]
        limit *= 2


def prime_gaps(primes: list[int]) -> np.ndarray:
    return np.diff(np.array(primes, dtype=float))


def normalize_to_angles(values: np.ndarray) -> np.ndarray:
    """Min-max normalize `values` to the [0, pi] range used for Ry rotation angles."""
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.zeros_like(values)
    return np.pi * (values - lo) / (hi - lo)


def build_encoding_circuit(angles: np.ndarray, n_qubits: int) -> QuantumCircuit:
    """Angle-encode `angles` onto `n_qubits` via data re-uploading: chunk the
    sequence into groups of `n_qubits`, Ry-rotate qubit i by chunk[i], then
    entangle the register with a ring of CX gates before the next chunk."""
    qc = QuantumCircuit(n_qubits, name="encode")
    for start in range(0, len(angles), n_qubits):
        chunk = angles[start : start + n_qubits]
        for i, theta in enumerate(chunk):
            qc.ry(theta, i)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
    return qc


def build_full_circuit(angles: np.ndarray, n_qubits: int) -> QuantumCircuit:
    qc = build_encoding_circuit(angles, n_qubits)
    qc.append(QFTGate(n_qubits), range(n_qubits))
    return qc


# --- Independent reference simulator (no Qiskit), used only to verify the
# circuit above computes what it's supposed to. Every gate is applied directly
# to the state vector by bit-indexing, matching Qiskit's little-endian
# convention where qubit q is bit q of the basis state index. ---


def _ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    idx = np.arange(len(state))
    mask = 1 << qubit
    zero_idx = idx[(idx & mask) == 0]
    one_idx = zero_idx | mask
    a, b = state[zero_idx], state[one_idx]
    new_state = state.copy()
    new_state[zero_idx] = gate[0, 0] * a + gate[0, 1] * b
    new_state[one_idx] = gate[1, 0] * a + gate[1, 1] * b
    return new_state


def _apply_cx(state: np.ndarray, control: int, target: int) -> np.ndarray:
    idx = np.arange(len(state))
    control_on = idx[(idx & (1 << control)) != 0]
    swapped = control_on ^ (1 << target)
    new_state = state.copy()
    new_state[swapped] = state[control_on]
    return new_state


def _qft_matrix(n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    j = np.arange(dim).reshape(-1, 1)
    k = np.arange(dim).reshape(1, -1)
    return np.exp(2j * np.pi * j * k / dim) / np.sqrt(dim)


def reference_statevector(angles: np.ndarray, n_qubits: int) -> np.ndarray:
    """Bit-for-bit reimplementation of `build_full_circuit`'s action on |0...0>,
    built from raw numpy linear algebra with no dependency on Qiskit's simulator."""
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    for start in range(0, len(angles), n_qubits):
        chunk = angles[start : start + n_qubits]
        for i, theta in enumerate(chunk):
            state = _apply_single_qubit_gate(state, _ry_matrix(theta), i)
        for i in range(n_qubits):
            state = _apply_cx(state, i, (i + 1) % n_qubits)
    return _qft_matrix(n_qubits) @ state


def shannon_entropy(probabilities: np.ndarray) -> float:
    nonzero = probabilities[probabilities > 1e-12]
    return float(-np.sum(nonzero * np.log2(nonzero)))


@dataclass
class VerificationResult:
    primes_match_sieve: bool
    probabilities_sum_to_one: bool
    matches_reference_simulator: bool
    real_entropy: float
    mean_shuffled_entropy: float
    shuffled_entropies: list[float]

    def ordering_looks_structured(self) -> bool:
        return self.real_entropy < self.mean_shuffled_entropy

    def all_hard_checks_passed(self) -> bool:
        return self.primes_match_sieve and self.probabilities_sum_to_one and self.matches_reference_simulator


def verify(gaps: np.ndarray, n_qubits: int, n_shuffles: int = 50, seed: int = 0) -> VerificationResult:
    primes_match_sieve = FIRST_50_PRIMES == sieve_primes(50)

    angles = normalize_to_angles(gaps)
    qiskit_state = Statevector(build_full_circuit(angles, n_qubits)).data
    reference_state = reference_statevector(angles, n_qubits)
    matches_reference_simulator = bool(np.allclose(qiskit_state, reference_state, atol=1e-9))

    real_probs = np.abs(qiskit_state) ** 2
    probabilities_sum_to_one = bool(abs(real_probs.sum() - 1.0) < 1e-9)
    real_entropy = shannon_entropy(real_probs)

    rng = np.random.default_rng(seed)
    shuffled_entropies = []
    for _ in range(n_shuffles):
        shuffled_gaps = gaps.copy()
        rng.shuffle(shuffled_gaps)
        shuffled_angles = normalize_to_angles(shuffled_gaps)
        shuffled_state = Statevector(build_full_circuit(shuffled_angles, n_qubits)).data
        shuffled_probs = np.abs(shuffled_state) ** 2
        shuffled_entropies.append(shannon_entropy(shuffled_probs))

    return VerificationResult(
        primes_match_sieve=primes_match_sieve,
        probabilities_sum_to_one=probabilities_sum_to_one,
        matches_reference_simulator=matches_reference_simulator,
        real_entropy=real_entropy,
        mean_shuffled_entropy=float(np.mean(shuffled_entropies)),
        shuffled_entropies=shuffled_entropies,
    )


# --- Prediction phase --------------------------------------------------------------
#
# Everything above treats the gap sequence as a wave (time domain) and reads its QFT
# as that wave's spectrum (frequency domain) -- but the re-upload encoding used for
# the landscape/portrait above is a lossy, nonlinear feature map (the same small
# register is repeatedly overwritten and entangled across chunks), so there is no
# meaningful inverse QFT back through it to the original gap values. Prediction needs
# a genuinely invertible mapping, so it gets its own encoding: amplitude encoding
# (the classical vector, once L2-normalized, literally *is* the statevector's
# amplitudes), which makes the following two things well-defined:
#
# 1. Gate-level pathway (dim = 2**args.qubits, zero-padded, same register size as the
#    landscape/portrait pathway above): a real Quantum Fourier Transform of real
#    time-domain samples, exactly invertible. `quantum_fft` below extracts the actual
#    Qiskit-simulated (or hardware-measured) statevector as the frequency
#    representation used for prediction -- this is not a side sanity-check, it is
#    where the spectrum used for extrapolation actually comes from.
# 2. Extrapolation pathway (classical post-processing on whichever spectrum was
#    supplied -- numpy's for the classical path, `quantum_fft`'s for the quantum
#    path): reads the frequency domain as a continuous function of time and
#    evaluates it *past* the known window (`fourier_extrapolate` /
#    `quantum_fourier_extrapolate`). This is what "time evolution forward" actually
#    means here. No fixed-size QFT/IQFT gate can itself produce values past the
#    register it was given -- that's a mathematical fact about finite-dimensional
#    unitaries, not a limitation of this implementation -- so evaluating *past* the
#    known window is necessarily classical math regardless of where the spectrum
#    came from. Wiring in the quantum circuit changes WHERE the spectrum numbers come
#    from, not what happens to them afterward.
#
# `predict()`/`predict_quantum()` are called identically for the forward run and for
# backward verification (predicting known-but-withheld gaps from earlier ones), so a
# bad backward-verification score reflects the mechanism's real predictive power, not
# a quirk of two different code paths.
#
# Fairness consequence for comparing the two pathways: `quantum_fourier_extrapolate`
# always truncates to a handful of frequencies (see `QUANTUM_DEFAULT_TOP_K` below --
# the padded register makes a full-spectrum reconstruction degenerate), while the
# classical path's own default is "keep everything." Comparing them at their
# respective defaults would blend "which pathway computed the FFT" with "how much of
# the spectrum was kept" -- two different questions. So `backward_verify` and
# `backward_verify_quantum` are always called at the SAME `top_k` for the paired
# comparison report in `main()`, isolating exactly the thing being tested.


def amplitude_encode(values: np.ndarray, n_qubits: int) -> tuple[np.ndarray, float]:
    """Zero-pad `values` to 2**n_qubits and L2-normalize for amplitude encoding.

    This is a *different* normalization than `normalize_to_angles` above: quantum
    state amplitudes must have unit L2 norm, not values bounded to [0, pi]. The norm
    is returned so real-valued magnitudes could be recovered from the encoded state.
    """
    dim = 2**n_qubits
    padded = np.zeros(dim, dtype=float)
    padded[: len(values)] = values
    norm = float(np.linalg.norm(padded))
    if norm == 0:
        return padded, 0.0
    return padded / norm, norm


def build_amplitude_circuit(normalized_vector: np.ndarray) -> QuantumCircuit:
    """Literal amplitude encoding: the register is initialized so the statevector's
    real amplitudes *are* `normalized_vector`, unlike the angle/re-upload encoding
    used for the landscape/portrait pipeline above."""
    n_qubits = int(np.log2(len(normalized_vector)))
    qc = QuantumCircuit(n_qubits, name="amplitude_encode")
    qc.initialize(normalized_vector, range(n_qubits))
    return qc


@dataclass
class AmplitudeVerificationResult:
    matches_numpy_reference: bool
    roundtrip_matches: bool

    def all_passed(self) -> bool:
        return self.matches_numpy_reference and self.roundtrip_matches


def verify_amplitude_qft_roundtrip(normalized_vector: np.ndarray) -> AmplitudeVerificationResult:
    """Sanity-check the inverse QFT (checkpoint before any prediction is trusted):
    confirm the gate-level QFT of an amplitude-encoded state matches the from-scratch
    numpy DFT reference (`_qft_matrix`, already used by `verify()` above), and that
    appending its inverse recovers the original amplitudes to floating-point
    precision -- i.e. the round trip preserves both the values and the normalization.
    """
    n_qubits = int(np.log2(len(normalized_vector)))

    forward = build_amplitude_circuit(normalized_vector)
    forward.append(QFTGate(n_qubits), range(n_qubits))
    qft_state = Statevector(forward).data
    reference = _qft_matrix(n_qubits) @ normalized_vector.astype(complex)
    matches_numpy_reference = bool(np.allclose(qft_state, reference, atol=1e-9))

    roundtrip = forward.copy()
    roundtrip.append(QFTGate(n_qubits).inverse(), range(n_qubits))
    roundtrip_state = Statevector(roundtrip).data
    roundtrip_matches = bool(np.allclose(roundtrip_state, normalized_vector.astype(complex), atol=1e-9))

    return AmplitudeVerificationResult(matches_numpy_reference, roundtrip_matches)


def quantum_fft(values: np.ndarray, n_qubits: int) -> np.ndarray:
    """The actual quantum-circuit frequency representation used for prediction:
    amplitude-encode `values` (zero-padded to 2**n_qubits), run it through a real
    Qiskit `QFTGate`, and extract the resulting statevector as the spectrum.

    Qiskit's `QFTGate` uses the OPPOSITE sign convention from `np.fft.fft` (verified
    directly: the plain forward gate matches `sqrt(dim) * np.fft.ifft`, not
    `np.fft.fft`). To get a spectrum that's a drop-in match for `np.fft.fft` on the
    zero-padded array -- which is what `_dft_reconstruct` and everything built on top
    of it assumes -- this appends `QFTGate(n_qubits).inverse()`, not the forward gate,
    and rescales by `norm * sqrt(dim)` to undo both the L2-normalization
    `amplitude_encode` applied and Qiskit's own 1/sqrt(dim) QFT normalization.
    Cross-checked against `np.fft.fft` directly by `verify_quantum_fft_matches_padded_numpy`.
    """
    normalized, norm = amplitude_encode(values, n_qubits)
    circuit = build_amplitude_circuit(normalized)
    circuit.append(QFTGate(n_qubits).inverse(), range(n_qubits))
    statevector = Statevector(circuit).data
    dim = 2**n_qubits
    return statevector * norm * np.sqrt(dim)


def verify_quantum_fft_matches_padded_numpy(values: np.ndarray, n_qubits: int) -> bool:
    """Hard check: `quantum_fft` must match `np.fft.fft` on the zero-padded array to
    floating-point precision. This is what actually proves the sign-convention
    rescaling in `quantum_fft` is correct in the shipped code, rather than trusted by
    reasoning about Qiskit's gate convention alone -- getting it backwards would
    silently produce a conjugated/reversed spectrum that still runs without error."""
    dim = 2**n_qubits
    padded = np.zeros(dim, dtype=float)
    padded[: len(values)] = values
    return bool(np.allclose(quantum_fft(values, n_qubits), np.fft.fft(padded), atol=1e-6))


def _frequency_components(n: int) -> list[tuple[int, ...]]:
    """Group DFT bin indices for a length-`n` real-valued signal by frequency,
    pairing each bin `k` with its complex-conjugate mirror `n - k` (a real signal's
    spectrum is conjugate-symmetric, so a bin and its mirror always carry the same
    "frequency" -- unpaired only for the DC bin and, when `n` is even, the Nyquist
    bin)."""
    components = []
    for k in range(n // 2 + 1):
        mirror = (n - k) % n
        components.append((k,) if mirror == k else (k, mirror))
    return components


def _ranked_frequency_components(spectrum: np.ndarray, n: int) -> list[tuple[tuple[int, ...], float]]:
    """Pair each frequency component from `_frequency_components` with its Parseval
    power (sum |X_k|^2 over its bins) and rank them descending -- this is the same
    quantity plotted as the amplitude landscape, used here to decide which frequencies
    "amplitude concentrates" in."""
    components = _frequency_components(n)
    powers = [sum(abs(spectrum[b]) ** 2 for b in bins) for bins in components]
    return sorted(zip(components, powers), key=lambda cp: cp[1], reverse=True)


def _dft_reconstruct(spectrum: np.ndarray, n: int, t_values: np.ndarray, top_k: int | None = None) -> np.ndarray:
    """Evaluate the inverse-DFT sum x(t) = (1/n) * sum_k X_k * exp(2j*pi*k*t/n) at
    arbitrary times `t_values`, including t >= n. This is the step that actually reads
    the frequency domain as a continuous function rather than just the n known sample
    points -- distinct from, and not the same operation as, the fixed-size gate-level
    inverse QFT verified above, which can only map the register back to those same n
    points. Evaluated at integer t in [0, n), this must reproduce the known values
    exactly when `top_k` is None (see `verify_extrapolation_roundtrip`).

    If `top_k` is given, every frequency component except the `top_k` strongest is
    zeroed (mirrors kept together so the result stays real), which changes the
    assumption being tested from "the known window is exactly one period" to "only
    the dominant few periodicities matter."
    """
    if top_k is not None:
        ranked = _ranked_frequency_components(spectrum, n)
        keep_bins = {b for bins, _ in ranked[:top_k] for b in bins}
        mask = np.zeros(n, dtype=bool)
        mask[list(keep_bins)] = True
        spectrum = np.where(mask, spectrum, 0)

    t = np.asarray(t_values, dtype=float)
    k = np.arange(n)
    phase = 2j * np.pi * np.outer(t, k) / n
    return (np.exp(phase) @ spectrum).real / n


def fourier_extrapolate(values: np.ndarray, n_predict: int, top_k: int | None = None) -> np.ndarray:
    """"Time evolution": take the known gap sequence's spectrum and evaluate it past
    the known window to predict the next `n_predict` gaps. Default `top_k=None` keeps
    every frequency, which is equivalent to assuming the known window is exactly one
    period -- the least arbitrary default, since any other choice of which frequencies
    to drop is itself an assumption about the data."""
    n = len(values)
    spectrum = np.fft.fft(values)
    future_t = np.arange(n, n + n_predict)
    return _dft_reconstruct(spectrum, n, future_t, top_k)


def verify_extrapolation_roundtrip(values: np.ndarray) -> bool:
    """Hard check: the full-spectrum continuous-time reconstruction, evaluated back at
    the original known indices, must reproduce the known gap values -- confirms
    `_dft_reconstruct` is a correct inverse DFT and not just a plausible-looking
    formula. (Only meaningful for the full spectrum -- a `top_k`-truncated
    reconstruction is expected to differ from the known values by construction.)"""
    n = len(values)
    spectrum = np.fft.fft(values)
    reconstructed = _dft_reconstruct(spectrum, n, np.arange(n), top_k=None)
    return bool(np.allclose(reconstructed, values, atol=1e-9))


# Amplitude encoding zero-pads the known gaps to a power-of-2 register (`quantum_fft`
# above), so a FULL-spectrum reconstruction from it is an exact IDFT(DFT(x)) = x
# identity over the *padded* array -- it would just reproduce the padding zeros for
# every "predicted" gap until the register wraps around, not a forecast. The quantum
# path therefore always truncates to a modest number of dominant frequency
# components; `top_k=None` substitutes this default rather than meaning "everything,"
# unlike the classical path (which never pads, so "everything" is well-defined there).
QUANTUM_DEFAULT_TOP_K = 5

# NOTE on what "quantum vs classical agreement" actually means here, since it's easy
# to conflate two different questions: (1) "is quantum_fft computing the right thing"
# -- YES, proven to floating-point precision (~1e-13) by
# `verify_quantum_fft_matches_padded_numpy` against np.fft.fft on the SAME (padded)
# array. (2) "does the backward-verification MAE match between pathways at the same
# top_k" -- NOT expected to be floating-point-close, even though (1) holds: the
# quantum path's mandatory zero-padding (49/39 known gaps into a much larger
# power-of-2 register) dilutes spectral power across many more bins, so "top-k"
# selects a genuinely different, coarser set of frequencies than the classical
# path's native unpadded spectrum. A nonzero MAE gap here reflects that padding cost,
# not a computational error -- there's no floating-point threshold to gate on.


def quantum_fourier_extrapolate(
    values: np.ndarray, n_predict: int, n_qubits: int, top_k: int | None = None
) -> np.ndarray:
    """"Time evolution" via the actual quantum-circuit spectrum: `quantum_fft` supplies
    the frequency representation (from a zero-padded, `2**n_qubits`-dimensional
    register), and `_dft_reconstruct` evaluates it past the known window -- the same
    classical continuous-time-evaluation step `fourier_extrapolate` uses, just fed a
    quantum-circuit-derived spectrum instead of `np.fft.fft`'s. See the
    `QUANTUM_DEFAULT_TOP_K` note above for why `top_k=None` is substituted rather than
    left as "keep everything" here specifically.
    """
    if top_k is None:
        top_k = QUANTUM_DEFAULT_TOP_K
        _log_event(
            f"quantum path: --top-k not given, defaulting to top-{QUANTUM_DEFAULT_TOP_K} "
            "frequency components (a full-spectrum reconstruction of the zero-padded "
            "register would just reproduce the padding as \"predicted\" gaps)"
        )
    dim = 2**n_qubits
    spectrum = quantum_fft(values, n_qubits)
    future_t = np.arange(len(values), len(values) + n_predict)
    return _dft_reconstruct(spectrum, dim, future_t, top_k)


@dataclass
class PredictionResult:
    known_gaps: np.ndarray
    predicted_gaps: np.ndarray
    top_k: int | None
    predicted_primes: np.ndarray
    pathway: str = "classical"


def predict(gaps: np.ndarray, n_predict: int, top_k: int | None, start_prime: int) -> PredictionResult:
    """The classical prediction mechanism (numpy FFT), used identically for the
    forward run and for backward verification: only `gaps`, `n_predict`, and
    `start_prime` differ between the two callers."""
    predicted_gaps = fourier_extrapolate(gaps, n_predict, top_k=top_k)
    predicted_primes = start_prime + np.cumsum(predicted_gaps)
    return PredictionResult(gaps, predicted_gaps, top_k, predicted_primes, pathway="classical")


def predict_quantum(
    gaps: np.ndarray, n_predict: int, n_qubits: int, top_k: int | None, start_prime: int
) -> PredictionResult:
    """The quantum-circuit prediction mechanism: identical in shape to `predict()`,
    but sourcing its spectrum from `quantum_fourier_extrapolate` (an actual Qiskit
    `QFTGate`) instead of `np.fft.fft`."""
    predicted_gaps = quantum_fourier_extrapolate(gaps, n_predict, n_qubits, top_k=top_k)
    predicted_primes = start_prime + np.cumsum(predicted_gaps)
    return PredictionResult(gaps, predicted_gaps, top_k, predicted_primes, pathway="quantum")


@dataclass
class PrimeCandidate:
    frequencies_used: int
    power_fraction: float
    predicted_gaps: np.ndarray
    raw_candidates: np.ndarray
    nearest_primes: list[int]
    distances: np.ndarray


def _candidate_zones_from_spectrum(
    spectrum: np.ndarray,
    n: int,
    known_length: int,
    n_predict: int,
    start_prime: int,
    prime_pool: list[int],
    max_levels: int = 5,
) -> list[PrimeCandidate]:
    """Shared zone-sweep core: rank `spectrum`'s frequency components (`n` bins) by
    Parseval power, then reconstruct at truncation levels 1..max_levels via
    `_dft_reconstruct` directly, evaluated from `known_length` onward (the true number
    of known gaps -- may differ from `n` when `spectrum` came from a padded quantum
    register). Used by `spectral_candidate_zones` (classical/quantum) and
    `hardware_informed_candidate_zones` (hybrid) alike, so all three read the same way.
    """
    ranked = _ranked_frequency_components(spectrum, n)
    total_power = float(sum(power for _, power in ranked))
    future_t = np.arange(known_length, known_length + n_predict)

    zones = []
    for level in range(1, min(max_levels, len(ranked)) + 1):
        kept_power = float(sum(power for _, power in ranked[:level]))
        power_fraction = kept_power / total_power if total_power > 0 else 0.0
        predicted_gaps = _dft_reconstruct(spectrum, n, future_t, top_k=level)
        raw_candidates = start_prime + np.cumsum(predicted_gaps)
        nearest_primes = [min(prime_pool, key=lambda c: abs(c - raw)) for raw in raw_candidates]
        distances = np.abs(np.array(nearest_primes) - raw_candidates)
        zones.append(
            PrimeCandidate(
                frequencies_used=level,
                power_fraction=power_fraction,
                predicted_gaps=predicted_gaps,
                raw_candidates=raw_candidates,
                nearest_primes=nearest_primes,
                distances=distances,
            )
        )
    return zones


def spectral_candidate_zones(
    values: np.ndarray,
    n_predict: int,
    start_prime: int,
    prime_pool: list[int],
    max_levels: int = 5,
    n_qubits: int | None = None,
) -> list[PrimeCandidate]:
    """Read where amplitude concentrates in the frequency domain and turn that into
    ranked candidate primes. Each "zone" is a reconstruction built from the top-k
    strongest frequency components (k = 1, 2, 3, ...), weighted by the cumulative
    fraction of Parseval spectral power those components represent -- the same
    quantity plotted as the amplitude landscape, not an invented probability. Weight
    approaches 1.0 as more components are included.

    `n_qubits=None` (default) sources the spectrum classically (`np.fft.fft` on the
    unpadded gaps, `n = len(values)`) -- unchanged from before. Passing `n_qubits`
    sources it from the actual quantum circuit (`quantum_fft`, `n = 2**n_qubits`)
    instead, so the same zone-sweep logic serves both pathways.
    """
    if n_qubits is None:
        n = len(values)
        spectrum = np.fft.fft(values)
    else:
        n = 2**n_qubits
        spectrum = quantum_fft(values, n_qubits)
    return _candidate_zones_from_spectrum(spectrum, n, len(values), n_predict, start_prime, prime_pool, max_levels)


def hardware_informed_candidate_zones(
    hardware_probabilities: np.ndarray,
    sim_statevector: np.ndarray,
    norm: float,
    n_qubits: int,
    known_length: int,
    n_predict: int,
    start_prime: int,
    prime_pool: list[int],
    max_levels: int = 5,
) -> list[PrimeCandidate]:
    """NOT a full hardware reconstruction of the candidate zones -- a clearly-labeled
    hybrid. A single Sampler measurement only yields Born-rule probabilities
    (magnitude squared); phase is destroyed by measurement and recovering it would
    need full state tomography (multiple non-commuting measurement bases, exponential
    in qubit count), out of scope for one run. This combines the *magnitude* actually
    measured on hardware (`sqrt(hardware_probabilities)`, so it does carry real
    hardware noise) with the *phase* from the noiseless simulator (unmeasured, assumed
    ideal) to see what magnitude-only noise alone does to the candidate zones. It says
    nothing about phase noise, which this circuit cannot measure at all -- report this
    everywhere as "magnitude-from-hardware / phase-from-simulator hybrid," never as an
    unqualified hardware result.
    """
    dim = 2**n_qubits
    hybrid_amplitudes = np.sqrt(hardware_probabilities) * np.exp(1j * np.angle(sim_statevector))
    hybrid_spectrum = hybrid_amplitudes * norm * np.sqrt(dim)
    return _candidate_zones_from_spectrum(
        hybrid_spectrum, dim, known_length, n_predict, start_prime, prime_pool, max_levels
    )


@dataclass
class BackwardVerificationResult:
    predicted_gaps: np.ndarray
    actual_gaps: np.ndarray
    mae: float
    baseline_mean_mae: float
    baseline_last_repeat_mae: float
    pathway: str = "classical"

    def beats_baselines(self) -> bool:
        return self.mae < min(self.baseline_mean_mae, self.baseline_last_repeat_mae)


def _backward_split(n_predict: int = 10) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Shared setup for both classical and quantum backward verification: the
    known/actual gap split, the 40th-prime start point, and the two naive baselines
    (identical for both pathways, so a reported MAE difference reflects the pathway,
    not the data split)."""
    all_gaps = prime_gaps(FIRST_50_PRIMES)
    known_gaps = all_gaps[:39]
    actual_gaps = all_gaps[39 : 39 + n_predict]
    start_prime = FIRST_50_PRIMES[39]  # the 40th prime
    baseline_mean_mae = float(np.mean(np.abs(known_gaps.mean() - actual_gaps)))
    baseline_last_repeat_mae = float(np.mean(np.abs(known_gaps[-1] - actual_gaps)))
    return known_gaps, actual_gaps, start_prime, baseline_mean_mae, baseline_last_repeat_mae


def backward_verify(top_k: int | None = None, n_predict: int = 10) -> BackwardVerificationResult:
    """The classical accuracy check: feed in the first 40 primes (39 known gaps),
    predict the next `n_predict` gaps (indices 40..49, i.e. exactly the gaps needed to
    know primes 41..50), and compare against the real values, via the exact same
    `predict()` used for the forward run -- a bad score here means the mechanism, not
    just the specific forward guess, doesn't work. Two naive baselines (repeat the
    mean known gap; repeat the last known gap) are reported alongside so "reasonable
    accuracy" has a reference point, the same role the shuffled-control comparison
    plays in `verify()`. See `backward_verify_quantum` for the quantum-circuit twin.
    """
    known_gaps, actual_gaps, start_prime, baseline_mean_mae, baseline_last_repeat_mae = _backward_split(n_predict)
    result = predict(known_gaps, n_predict, top_k, start_prime)
    mae = float(np.mean(np.abs(result.predicted_gaps - actual_gaps)))
    return BackwardVerificationResult(
        predicted_gaps=result.predicted_gaps,
        actual_gaps=actual_gaps,
        mae=mae,
        baseline_mean_mae=baseline_mean_mae,
        baseline_last_repeat_mae=baseline_last_repeat_mae,
        pathway="classical",
    )


def backward_verify_quantum(n_qubits: int, top_k: int | None = None, n_predict: int = 10) -> BackwardVerificationResult:
    """The quantum-circuit twin of `backward_verify`: identical split, identical
    baselines, but the prediction comes from `predict_quantum` (an actual Qiskit
    `QFTGate`) instead of `predict`. Called at the same `top_k` as `backward_verify`
    for a fair side-by-side MAE comparison -- see the "Fairness consequence" note in
    the module docstring above for why matching `top_k` matters here."""
    known_gaps, actual_gaps, start_prime, baseline_mean_mae, baseline_last_repeat_mae = _backward_split(n_predict)
    result = predict_quantum(known_gaps, n_predict, n_qubits, top_k, start_prime)
    mae = float(np.mean(np.abs(result.predicted_gaps - actual_gaps)))
    return BackwardVerificationResult(
        predicted_gaps=result.predicted_gaps,
        actual_gaps=actual_gaps,
        mae=mae,
        baseline_mean_mae=baseline_mean_mae,
        baseline_last_repeat_mae=baseline_last_repeat_mae,
        pathway="quantum",
    )


@dataclass
class HardwareRunMetadata:
    backend_name: str
    job_id: str
    pending_jobs: int
    shots: int
    transpiled_depth: int
    transpiled_gate_count: int
    mitigation_applied: bool
    queue_wait_seconds: float | None
    wall_seconds: float
    dynamical_decoupling: bool = False


# Below this queue depth (pending jobs on the selected backend), the adaptive shot
# policy uses the full 4096 shots; at or above it, 1024, to avoid adding to a long
# queue for a shot count beyond what "queue time allows" -- pending_jobs is the only
# pre-submission queue signal the API exposes, there's no direct wait-time estimate.
ADAPTIVE_SHOTS_QUEUE_THRESHOLD = 5
ADAPTIVE_SHOTS_LOW_QUEUE = 4096
ADAPTIVE_SHOTS_HIGH_QUEUE = 1024

# Qiskit transpiles a fixed circuit down to a backend's native gate set and coupling
# map; past this depth, accumulated gate noise tends to dominate the signal on current
# hardware -- flagged as a warning, not a hard stop, since the run still proceeds and
# the noise level is exactly what's being measured.
TRANSPILED_DEPTH_WARNING_THRESHOLD = 50

# Dynamical-decoupling sequence used when --dynamical-decoupling is set: cancels
# dephasing noise during idle windows between gates via timed echo pulses. "XpXm" is
# the sequence most effective against dephasing specifically (vs. "XX"/"XY4", which
# target other noise channels) -- see DynamicalDecouplingOptions in qiskit-ibm-runtime.
DD_SEQUENCE_TYPE = "XpXm"

# The prediction-circuit job from the first hardware run, documented in
# 7QUBIT_HW_RESULTS.md. Retrieved fresh via `fetch_first_hardware_run` for every DD
# comparison rather than re-run, so "first hardware run" in that comparison is
# literally that same measurement -- not a second, possibly hardware-drifted one.
FIRST_HARDWARE_RUN_JOB_ID = "d9tso90u5hac73agdrk0"


def fetch_first_hardware_run(n_qubits: int) -> np.ndarray:
    """Retrieve the first hardware run's actual measured counts from IBM (a free,
    read-only API call -- no quota cost) and return them as a probability array, so
    every DD-vs-first-run comparison uses the exact original measurement."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    job = service.job(FIRST_HARDWARE_RUN_JOB_ID)
    result = job.result()
    counts = dict(result[0].data.meas.get_counts())
    return hardware_counts_to_probabilities(counts, n_qubits)


def run_on_hardware(
    circuit: QuantumCircuit,
    backend_name: str | None,
    shots: int | None,
    dynamical_decoupling: bool = False,
) -> tuple[dict[str, int], HardwareRunMetadata]:
    """Transpile `circuit` (with measurements) for a real IBM Quantum backend and run
    it via the Sampler primitive with readout error mitigation (measurement twirling)
    always enabled, returning (bitstring -> count, run metadata).

    Needs IBM Quantum credentials: either the QISKIT_IBM_TOKEN environment
    variable (see README.md for how to set it), or an account already saved
    locally via QiskitRuntimeService.save_account(channel=..., token=...).

    `shots=None` uses the adaptive policy (see `ADAPTIVE_SHOTS_QUEUE_THRESHOLD`);
    an explicit shot count is used exactly as given. `dynamical_decoupling=True`
    additionally enables DD (`DD_SEQUENCE_TYPE`) -- a server-side scheduling pass
    applied to idle windows in the already-transpiled circuit, so it does NOT change
    the locally-measured `transpiled_depth` below; that's expected, not a sign DD had
    no effect.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()
    backend = service.backend(backend_name) if backend_name else service.least_busy(min_num_qubits=circuit.num_qubits)
    pending_jobs = backend.status().pending_jobs
    print(f"Selected backend: {backend.name} (queue depth: {pending_jobs} pending jobs)")

    if shots is None:
        shots = ADAPTIVE_SHOTS_LOW_QUEUE if pending_jobs <= ADAPTIVE_SHOTS_QUEUE_THRESHOLD else ADAPTIVE_SHOTS_HIGH_QUEUE
        _log_event(
            f"--shots not given: queue depth {pending_jobs} <= {ADAPTIVE_SHOTS_QUEUE_THRESHOLD} threshold, "
            f"using {shots} shots"
            if pending_jobs <= ADAPTIVE_SHOTS_QUEUE_THRESHOLD
            else f"--shots not given: queue depth {pending_jobs} > {ADAPTIVE_SHOTS_QUEUE_THRESHOLD} threshold, "
            f"using {shots} shots to limit added queue load"
        )
    print(f"Shots: {shots}")

    measured = circuit.copy()
    measured.measure_all()
    transpiled = transpile(measured, backend=backend, optimization_level=3)
    transpiled_depth = transpiled.depth()
    transpiled_gate_count = transpiled.size()
    print(f"Transpiled circuit: depth {transpiled_depth}, {transpiled_gate_count} gates")
    if transpiled_depth > TRANSPILED_DEPTH_WARNING_THRESHOLD:
        _log_event(
            f"WARNING: transpiled depth {transpiled_depth} exceeds the "
            f"{TRANSPILED_DEPTH_WARNING_THRESHOLD}-gate threshold -- deep circuits accumulate noise fast, "
            "expect the hardware distribution to diverge more from the simulated one."
        )

    sampler = SamplerV2(mode=backend)
    sampler.options.twirling.enable_measure = True
    mitigation_applied = True
    _log_event("Readout error mitigation: measurement twirling enabled (no added circuit depth).")
    if dynamical_decoupling:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = DD_SEQUENCE_TYPE
        _log_event(
            f"Dynamical decoupling enabled (sequence: {DD_SEQUENCE_TYPE}) -- a server-side scheduling pass on "
            "idle windows, applied on top of the already-transpiled circuit; it will not change the locally "
            "measured transpiled depth printed above."
        )

    start = time.monotonic()
    job = sampler.run([transpiled], shots=shots)
    job_id = job.job_id()
    print(f"Submitted job {job_id}, waiting for results...")
    result = job.result()
    wall_seconds = time.monotonic() - start

    queue_wait_seconds = None
    try:
        timestamps = job.metrics().get("timestamps", {})
        created, running = timestamps.get("created"), timestamps.get("running")
        if created and running:
            queue_wait_seconds = (
                datetime.fromisoformat(running) - datetime.fromisoformat(created)
            ).total_seconds()
    except Exception as exc:  # noqa: BLE001 -- best-effort metadata, must never lose the actual results
        _log_event(f"Could not determine queue wait time from job.metrics() ({exc}); wall time reported instead.")

    counts = result[0].data.meas.get_counts()
    metadata = HardwareRunMetadata(
        backend_name=backend.name,
        job_id=job_id,
        pending_jobs=pending_jobs,
        shots=shots,
        transpiled_depth=transpiled_depth,
        transpiled_gate_count=transpiled_gate_count,
        mitigation_applied=mitigation_applied,
        queue_wait_seconds=queue_wait_seconds,
        wall_seconds=wall_seconds,
        dynamical_decoupling=dynamical_decoupling,
    )
    return dict(counts), metadata


def plot_gap_sequence(gaps: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    indices = np.arange(1, len(gaps) + 1)
    ax.stem(indices, gaps, basefmt=" ")
    ax.set_xlabel("gap index (between prime[i] and prime[i+1])")
    ax.set_ylabel("gap size")
    ax.set_title("Prime gap sequence, first 50 primes")
    fig.tight_layout()
    path = OUTPUT_DIR / "gap_sequence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_amplitude_landscape(probabilities: np.ndarray, phases: np.ndarray, n_qubits: int) -> Path:
    dim = 2**n_qubits
    fig, (ax_prob, ax_phase) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_prob.bar(range(dim), probabilities, color="tab:blue")
    ax_prob.set_ylabel("probability")
    ax_prob.set_title(f"Amplitude landscape after QFT -- SIMULATED ({n_qubits} qubits, {dim} basis states)")

    ax_phase.bar(range(dim), phases, color="tab:orange")
    ax_phase.set_ylabel("phase (rad)")
    ax_phase.set_xlabel("computational basis state index")

    fig.tight_layout()
    path = OUTPUT_DIR / "amplitude_landscape_sim.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_frequency_portrait(probabilities: np.ndarray, n_qubits: int) -> Path:
    """Re-center the amplitude landscape around frequency 0, the way a classical
    FFT magnitude spectrum is usually displayed, since QFT basis state k directly
    corresponds to frequency bin k (with k and dim-k representing +/- frequency)."""
    dim = 2**n_qubits
    freqs = np.fft.fftshift(np.fft.fftfreq(dim, d=1.0)) * dim
    shifted_probs = np.fft.fftshift(probabilities)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, shifted_probs, marker="o", color="tab:green")
    ax.set_xlabel("frequency bin")
    ax.set_ylabel("probability")
    ax.set_title("Frequency portrait of the prime gap wave -- SIMULATED")
    fig.tight_layout()
    path = OUTPUT_DIR / "frequency_portrait_sim.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_extended_wave(known_gaps: np.ndarray, classical_predicted: np.ndarray, quantum_predicted: np.ndarray) -> Path:
    """The gap sequence known so far, plus BOTH pathways' predictions past it, each
    clearly labeled by which one produced it (classical numpy FFT vs. actual quantum
    circuit) -- not just one plotted as if it were the only candidate."""
    fig, ax = plt.subplots(figsize=(10, 4))
    known_idx = np.arange(1, len(known_gaps) + 1)
    predicted_idx = np.arange(len(known_gaps) + 1, len(known_gaps) + len(classical_predicted) + 1)

    ax.plot(known_idx, known_gaps, "o-", color="tab:blue", label="known")
    ax.plot(predicted_idx, classical_predicted, "o--", color="tab:red", label="predicted (classical, numpy FFT)")
    ax.plot(predicted_idx, quantum_predicted, "s--", color="tab:purple", label="predicted (quantum circuit)")
    ax.axvline(len(known_gaps) + 0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("gap index")
    ax.set_ylabel("gap size")
    ax.set_title("Prime gap wave: known vs. classical vs. quantum-circuit prediction")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "extended_wave_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def hardware_counts_to_probabilities(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """Convert a bitstring -> count map from `run_on_hardware` into a probability
    array indexed the same way as `Statevector.probabilities()`: index k has
    qubit 0 as its least significant bit, which is also how Qiskit writes
    measurement bitstrings (leftmost char = highest classical bit), so
    `int(bitstring, 2)` lines up with the statevector index directly."""
    dim = 2**n_qubits
    total = sum(counts.values())
    probabilities = np.zeros(dim)
    for bitstring, count in counts.items():
        probabilities[int(bitstring, 2)] = count / total
    return probabilities


def plot_hardware_overlay(
    sim_probabilities: np.ndarray,
    counts: dict[str, int],
    n_qubits: int,
    backend_name: str,
    label: str | None = None,
) -> Path:
    """Overlay real QUANTUM HARDWARE measured probabilities against the SIMULATED
    statevector probabilities for the same circuit, so noise/decoherence effects
    are visible directly against the ideal result. `label` distinguishes which
    circuit this is for (e.g. "prediction") when more than one `--hardware` overlay
    is written in the same run -- `None` keeps the original landscape-pathway
    filename unchanged."""
    dim = 2**n_qubits
    shots = sum(counts.values())
    hardware_probabilities = hardware_counts_to_probabilities(counts, n_qubits)
    title_suffix = f" ({label})" if label else ""

    x = np.arange(dim)
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, sim_probabilities, width, label="simulated", color="tab:blue")
    ax.bar(x + width / 2, hardware_probabilities, width, label=f"quantum hardware ({backend_name})", color="tab:red")
    ax.set_xlabel("computational basis state index")
    ax.set_ylabel("probability")
    ax.set_title(f"QUANTUM HARDWARE ({backend_name}, {shots} shots) vs SIMULATED amplitude landscape{title_suffix}")
    ax.legend()
    fig.tight_layout()

    backend_slug = "".join(c if c.isalnum() else "_" for c in backend_name)
    filename = f"amplitude_landscape_{label}_quantum_{backend_slug}.png" if label else f"amplitude_landscape_quantum_{backend_slug}.png"
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hardware_amplitude_landscape(hardware_probabilities: np.ndarray, n_qubits: int, backend_name: str) -> Path:
    """The hardware-measured probability distribution alone, the hardware equivalent
    of the simulated statevector's amplitude landscape (magnitude only -- hardware
    measurement gives no phase)."""
    dim = 2**n_qubits
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(dim), hardware_probabilities, color="tab:red")
    ax.set_xlabel("computational basis state index")
    ax.set_ylabel("probability")
    ax.set_title(f"Amplitude landscape after QFT -- QUANTUM HARDWARE ({backend_name}, {n_qubits} qubits, {dim} basis states)")
    fig.tight_layout()
    path = OUTPUT_DIR / "hardware_amplitude_landscape.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hardware_vs_sim_comparison(
    sim_probabilities: np.ndarray, hardware_probabilities: np.ndarray, n_qubits: int, backend_name: str
) -> Path:
    """Simulated and hardware amplitude landscapes as two side-by-side panels (not
    overlaid bars) sharing the same y-axis scale, so the gap between them -- the noise
    floor -- is directly visible as a shape difference between the two panels."""
    dim = 2**n_qubits
    shared_ylim = max(sim_probabilities.max(), hardware_probabilities.max()) * 1.1

    fig, (ax_sim, ax_hw) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ax_sim.bar(range(dim), sim_probabilities, color="tab:blue")
    ax_sim.set_title("Simulated")
    ax_sim.set_xlabel("computational basis state index")
    ax_sim.set_ylabel("probability")
    ax_sim.set_ylim(0, shared_ylim)

    ax_hw.bar(range(dim), hardware_probabilities, color="tab:red")
    ax_hw.set_title(f"Quantum hardware ({backend_name})")
    ax_hw.set_xlabel("computational basis state index")
    ax_hw.set_ylim(0, shared_ylim)

    fig.suptitle("Amplitude landscape: simulated vs. hardware (same axis scale -- the gap is the noise floor)")
    fig.tight_layout()
    path = OUTPUT_DIR / "hardware_vs_sim_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hardware_frequency_portrait(
    sim_probabilities: np.ndarray, hardware_probabilities: np.ndarray, n_qubits: int
) -> Path:
    """Frequency portrait (same fftshift re-centering as `plot_frequency_portrait`)
    with simulated and hardware distributions overlaid on one axes, so it's visible
    directly which peaks survive hardware noise and which collapse into it."""
    dim = 2**n_qubits
    freqs = np.fft.fftshift(np.fft.fftfreq(dim, d=1.0)) * dim
    shifted_sim = np.fft.fftshift(sim_probabilities)
    shifted_hw = np.fft.fftshift(hardware_probabilities)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, shifted_sim, marker="o", color="tab:green", label="simulated")
    ax.plot(freqs, shifted_hw, marker="o", color="tab:red", alpha=0.8, label="quantum hardware")
    ax.set_xlabel("frequency bin")
    ax.set_ylabel("probability")
    ax.set_title("Frequency portrait of the prime gap wave -- simulated vs. hardware")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "hardware_frequency_portrait.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hardware_dd_amplitude_landscape(dd_probabilities: np.ndarray, n_qubits: int, backend_name: str) -> Path:
    """The dynamical-decoupling hardware run's measured probability distribution alone
    -- the DD-mitigated equivalent of `plot_hardware_amplitude_landscape`."""
    dim = 2**n_qubits
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(dim), dd_probabilities, color="tab:orange")
    ax.set_xlabel("computational basis state index")
    ax.set_ylabel("probability")
    ax.set_title(
        f"Amplitude landscape after QFT -- QUANTUM HARDWARE + DYNAMICAL DECOUPLING "
        f"({backend_name}, {n_qubits} qubits, {dim} basis states)"
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "hardware_dd_amplitude_landscape.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hardware_dd_vs_first_run_comparison(
    sim_probabilities: np.ndarray,
    first_hw_probabilities: np.ndarray,
    dd_probabilities: np.ndarray,
    n_qubits: int,
    backend_name: str,
) -> Path:
    """Simulated, first hardware run (no DD), and DD hardware run as three side-by-side
    panels sharing one y-axis scale -- the gap between the first-run and DD panels is
    the mitigation effect, directly comparable to the gap between simulated and
    first-run in `hardware_vs_sim_comparison.png`."""
    dim = 2**n_qubits
    shared_ylim = max(sim_probabilities.max(), first_hw_probabilities.max(), dd_probabilities.max()) * 1.1

    fig, (ax_sim, ax_first, ax_dd) = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    ax_sim.bar(range(dim), sim_probabilities, color="tab:blue")
    ax_sim.set_title("Simulated")
    ax_sim.set_xlabel("computational basis state index")
    ax_sim.set_ylabel("probability")
    ax_sim.set_ylim(0, shared_ylim)

    ax_first.bar(range(dim), first_hw_probabilities, color="tab:red")
    ax_first.set_title("First hardware run (no DD)")
    ax_first.set_xlabel("computational basis state index")
    ax_first.set_ylim(0, shared_ylim)

    ax_dd.bar(range(dim), dd_probabilities, color="tab:orange")
    ax_dd.set_title(f"Hardware + dynamical decoupling ({backend_name})")
    ax_dd.set_xlabel("computational basis state index")
    ax_dd.set_ylim(0, shared_ylim)

    fig.suptitle("Amplitude landscape: simulated vs. first hardware run vs. dynamical decoupling (same axis scale)")
    fig.tight_layout()
    path = OUTPUT_DIR / "hardware_dd_vs_first_run_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hardware_dd_frequency_portrait(
    sim_probabilities: np.ndarray, first_hw_probabilities: np.ndarray, dd_probabilities: np.ndarray, n_qubits: int
) -> Path:
    """Frequency portrait with simulated, first hardware run, and DD hardware run all
    overlaid on one axes, so it's visible directly whether any frequency peaks lost in
    the first run re-emerge under dynamical decoupling."""
    dim = 2**n_qubits
    freqs = np.fft.fftshift(np.fft.fftfreq(dim, d=1.0)) * dim
    shifted_sim = np.fft.fftshift(sim_probabilities)
    shifted_first = np.fft.fftshift(first_hw_probabilities)
    shifted_dd = np.fft.fftshift(dd_probabilities)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, shifted_sim, marker="o", color="tab:green", label="simulated")
    ax.plot(freqs, shifted_first, marker="o", color="tab:red", alpha=0.7, label="first hardware run (no DD)")
    ax.plot(freqs, shifted_dd, marker="o", color="tab:orange", alpha=0.7, label="hardware + dynamical decoupling")
    ax.set_xlabel("frequency bin")
    ax.set_ylabel("probability")
    ax.set_title("Frequency portrait: simulated vs. first hardware run vs. dynamical decoupling")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "hardware_dd_frequency_portrait.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _zone_table(zones: list[PrimeCandidate]) -> str:
    lines = ["| Frequencies kept | Power fraction | Nearest primes | Mean distance |", "|---:|---:|---|---:|"]
    for zone in zones:
        primes_str = ", ".join(str(p) for p in zone.nearest_primes)
        lines.append(
            f"| {zone.frequencies_used} | {zone.power_fraction:.1%} | {primes_str} | {zone.distances.mean():.2f} |"
        )
    return "\n".join(lines)


def write_results_report(
    args: argparse.Namespace,
    effective_top_k: int,
    backward_classical: BackwardVerificationResult,
    backward_quantum: BackwardVerificationResult,
    mae_diff: float,
    verdict: str,
    prediction_classical: PredictionResult,
    prediction_quantum: PredictionResult,
    zones_classical: list[PrimeCandidate],
    zones_quantum: list[PrimeCandidate],
    png_paths: list[tuple[Path, str]],
) -> Path:
    """Write `output/prime/7QUBIT_QUANTUM_PREDICTION.md` summarizing this run -- called
    unconditionally at the end of every run, no manual documentation step. Overwritten
    every time the script runs, so it always reflects the most recent invocation.
    `mae_diff`/`verdict` are computed once in `main()` (and logged to `RUN_EVENTS`
    there) rather than recomputed here, so the console output and this report never
    disagree with each other."""
    png_table = "\n".join(f"| `{path.name}` | {pathway} |" for path, pathway in png_paths)
    events_section = "\n".join(f"- {event}" for event in RUN_EVENTS) if RUN_EVENTS else "- None."

    content = f"""# Quantum prime gap prediction -- run report

Generated automatically by `quantum_prime_gaps.py` at the end of every run -- this
file is overwritten each time, it does not accumulate history (see `NOTES.md` and
`6QUBIT_RESULTS.md` in this directory for hand-written point-in-time snapshots).

- **Timestamp:** {datetime.now().isoformat(timespec="seconds")}
- **Qubits (`--qubits`):** {args.qubits} (dim = {2**args.qubits}, shared by the landscape/portrait and prediction pathways)
- **Effective top-k for the paired backward-verification comparison:** {effective_top_k}
- **`--hardware` used this run:** {"Yes" if args.hardware else "No"}

## Classical vs. quantum backward verification

Predicting gaps 40-49 (i.e. primes 41-50) from only the first 40 primes' 39 known
gaps, both pathways run at the SAME `top_k` above so the comparison isolates the
pathway itself (numpy FFT vs. an actual Qiskit `QFTGate`), not a differing truncation
assumption.

| Pathway | MAE | Baseline (mean gap) | Baseline (repeat last) | Beats both baselines? |
|---|---:|---:|---:|:---:|
| Classical (numpy FFT) | {backward_classical.mae:.4f} | {backward_classical.baseline_mean_mae:.4f} | {backward_classical.baseline_last_repeat_mae:.4f} | {backward_classical.beats_baselines()} |
| Quantum (Qiskit circuit) | {backward_quantum.mae:.4f} | {backward_quantum.baseline_mean_mae:.4f} | {backward_quantum.baseline_last_repeat_mae:.4f} | {backward_quantum.beats_baselines()} |

**Absolute MAE difference:** {mae_diff:.6f}

**Verdict:** {verdict}

## Forward prediction (past gap 49, candidate primes from 229)

Quantum-circuit forecast (primary, per this run's request) alongside the classical
forecast for comparison:

| Step | Classical predicted gap | Classical candidate | Quantum predicted gap | Quantum candidate |
|---:|---:|---:|---:|---:|
{chr(10).join(
    f"| {i + 1} | {cg:.2f} | {cp:.1f} | {qg:.2f} | {qp:.1f} |"
    for i, (cg, cp, qg, qp) in enumerate(
        zip(
            prediction_classical.predicted_gaps,
            prediction_classical.predicted_primes,
            prediction_quantum.predicted_gaps,
            prediction_quantum.predicted_primes,
        )
    )
)}

### Candidate zones -- classical (numpy FFT)

{_zone_table(zones_classical)}

### Candidate zones -- quantum circuit

{_zone_table(zones_quantum)}

## Outputs from this run

| PNG file | Pathway that produced it |
|---|---|
{png_table}

## Console warnings / notable events during this run

{events_section}
"""

    path = OUTPUT_DIR / "7QUBIT_QUANTUM_PREDICTION.md"
    path.write_text(content)
    return path


def write_hardware_report(
    args: argparse.Namespace,
    metadata: HardwareRunMetadata,
    hardware_probabilities: np.ndarray,
    sim_probabilities: np.ndarray,
    zones_classical: list[PrimeCandidate],
    zones_quantum_sim: list[PrimeCandidate],
    zones_hybrid: list[PrimeCandidate],
    png_paths: list[tuple[Path, str]],
) -> Path:
    """Write `output/prime/7QUBIT_HW_RESULTS.md` -- called once the prediction circuit's
    real hardware run completes. Overwritten on every hardware run; prior results live
    in git history, not accumulated here."""
    hardware_mae = float(np.mean(np.abs(hardware_probabilities - sim_probabilities)))
    queue_wait_str = (
        f"{metadata.queue_wait_seconds:.1f}s" if metadata.queue_wait_seconds is not None
        else f"unavailable from job.metrics() -- wall time (submission to result) was {metadata.wall_seconds:.1f}s instead"
    )
    depth_note = (
        f" (exceeds the {TRANSPILED_DEPTH_WARNING_THRESHOLD}-gate warning threshold)"
        if metadata.transpiled_depth > TRANSPILED_DEPTH_WARNING_THRESHOLD
        else ""
    )

    classical_beats_hybrid = zones_classical[-1].distances.mean() > zones_hybrid[-1].distances.mean()
    zones_answer = (
        "The rigorous comparison (simulated quantum vs. classical, both noiseless) is in "
        "`7QUBIT_QUANTUM_PREDICTION.md`. For the hardware-noise-specific question: full hardware "
        "candidate zones cannot be computed from a single measurement setting -- Sampler destroys phase, "
        "and `_dft_reconstruct` needs it; recovering it would need full state tomography, out of scope "
        "for this run. The magnitude-from-hardware/phase-from-simulator hybrid below answers only the "
        "narrower question of whether *readout noise alone* moves the zones -- "
        f"at max frequencies kept, the hybrid's mean nearest-prime distance is "
        f"{zones_hybrid[-1].distances.mean():.2f} vs. classical's {zones_classical[-1].distances.mean():.2f} "
        f"({'closer to actual primes than classical' if classical_beats_hybrid else 'not closer than classical'} "
        "under this magnitude-only noise slice -- NOT a statement about the full hardware-noise question, "
        "which needs tomography to answer properly)."
    )

    png_table = "\n".join(f"| `{path.name}` | {pathway} |" for path, pathway in png_paths)
    events_section = "\n".join(f"- {event}" for event in RUN_EVENTS) if RUN_EVENTS else "- None."

    content = f"""# Quantum prime gap prediction -- real hardware run report

Generated automatically by `quantum_prime_gaps.py --hardware` when the prediction
circuit's hardware job completes. Overwritten on every hardware run -- prior results
live in git history, not accumulated here.

- **Timestamp:** {datetime.now().isoformat(timespec="seconds")}
- **Backend:** {metadata.backend_name} (selected dynamically via `least_busy`, not hardcoded)
- **Job ID:** {metadata.job_id}
- **Queue depth at selection:** {metadata.pending_jobs} pending jobs
- **Shots used:** {metadata.shots}
- **Transpiled circuit depth:** {metadata.transpiled_depth}{depth_note}
- **Transpiled gate count:** {metadata.transpiled_gate_count}
- **Readout error mitigation applied:** {"Yes (measurement twirling)" if metadata.mitigation_applied else "No"}
- **Queue wait time:** {queue_wait_str}

## Hardware vs. simulated amplitude landscape

**Mean absolute difference (hardware vs. simulated probability per basis state):**
{hardware_mae:.5f} -- this is the noise floor: on a noiseless simulator this would be
exactly 0, so this number *is* the measurable effect of real hardware noise on the
prediction circuit's readout.

## Candidate zones under hardware noise

{zones_answer}

### Candidate zones -- classical (numpy FFT, noiseless, for reference)

{_zone_table(zones_classical)}

### Candidate zones -- quantum circuit (noiseless simulator, for reference)

{_zone_table(zones_quantum_sim)}

### Candidate zones -- magnitude-from-hardware / phase-from-simulator HYBRID (NOT a full hardware reconstruction)

{_zone_table(zones_hybrid)}

## Outputs from this run

| PNG file | Pathway that produced it |
|---|---|
{png_table}

## Console warnings / notable events during this run

{events_section}
"""

    path = OUTPUT_DIR / "7QUBIT_HW_RESULTS.md"
    path.write_text(content)
    return path


def write_hardware_dd_report(
    args: argparse.Namespace,
    metadata: HardwareRunMetadata,
    sim_probabilities: np.ndarray,
    first_hw_probabilities: np.ndarray,
    dd_probabilities: np.ndarray,
    zones_classical: list[PrimeCandidate],
    zones_first_hybrid: list[PrimeCandidate],
    zones_dd_hybrid: list[PrimeCandidate],
    png_paths: list[tuple[Path, str]],
) -> Path:
    """Write `output/prime/7QUBIT_HW_DD_RESULTS.md` -- called once the dynamical-decoupling
    hardware run completes. Overwritten on every DD run; prior results live in git
    history. "First hardware run" throughout is the historical job re-fetched by
    `fetch_first_hardware_run`, not a fresh re-run -- the only variable that changed is
    whether DD was enabled."""
    mae_sim_first = float(np.mean(np.abs(first_hw_probabilities - sim_probabilities)))
    mae_sim_dd = float(np.mean(np.abs(dd_probabilities - sim_probabilities)))
    mae_first_dd = float(np.mean(np.abs(dd_probabilities - first_hw_probabilities)))

    depth_note = (
        f" (exceeds the {TRANSPILED_DEPTH_WARNING_THRESHOLD}-gate warning threshold)"
        if metadata.transpiled_depth > TRANSPILED_DEPTH_WARNING_THRESHOLD
        else ""
    )

    first_hybrid_distance = zones_first_hybrid[-1].distances.mean()
    dd_hybrid_distance = zones_dd_hybrid[-1].distances.mean()
    classical_distance = zones_classical[-1].distances.mean()

    mae_improved = mae_sim_dd < mae_sim_first
    zones_improved = dd_hybrid_distance < first_hybrid_distance
    verdict = (
        f"MAE vs. simulated: first run {mae_sim_first:.5f}, DD run {mae_sim_dd:.5f} "
        f"({'improved' if mae_improved else 'did not improve'} -- "
        f"{'lower' if mae_improved else 'not lower'} means closer to the simulated distribution). "
        f"Hybrid candidate-zone mean nearest-prime distance (max frequencies kept, magnitude-from-hardware/"
        f"phase-from-simulator, same caveats as `7QUBIT_HW_RESULTS.md`): first run {first_hybrid_distance:.2f}, "
        f"DD run {dd_hybrid_distance:.2f} ({'improved' if zones_improved else 'did not improve'} -- "
        f"classical baseline is {classical_distance:.2f} for reference). "
        + (
            "Both signals moved the same direction: dynamical decoupling measurably recovered some signal "
            "structure on this circuit."
            if mae_improved and zones_improved
            else "Neither signal improved: on this 907-gate circuit, dynamical decoupling did not measurably "
            "recover signal structure -- plausible given DD only protects idle qubit time, and an "
            "initialize()-heavy circuit at this depth likely keeps most qubits busy rather than idle, leaving "
            "little idle time for DD to protect."
            if not mae_improved and not zones_improved
            else "The two signals disagree (one improved, one didn't) -- treat this as inconclusive rather than "
            "a clean win or loss; the underlying amplitude landscape comparison plot is the more direct evidence "
            "than either single summary number."
        )
    )

    png_table = "\n".join(f"| `{path.name}` | {pathway} |" for path, pathway in png_paths)
    events_section = "\n".join(f"- {event}" for event in RUN_EVENTS) if RUN_EVENTS else "- None."

    content = f"""# Quantum prime gap prediction -- dynamical decoupling hardware run report

Generated automatically by `quantum_prime_gaps.py --hardware --dynamical-decoupling`
when the DD run completes. Overwritten on every DD run -- prior results live in git
history. "First hardware run" below is job `{FIRST_HARDWARE_RUN_JOB_ID}`
(documented in `7QUBIT_HW_RESULTS.md`), re-fetched fresh from IBM rather than re-run,
so dynamical decoupling is the only variable that changed between the two.

- **Timestamp:** {datetime.now().isoformat(timespec="seconds")}
- **Backend:** {metadata.backend_name}
- **DD job ID:** {metadata.job_id}
- **Shots used:** {metadata.shots}
- **Dynamical decoupling sequence:** {DD_SEQUENCE_TYPE}
- **Transpiled circuit depth:** {metadata.transpiled_depth}{depth_note} -- NOTE: dynamical decoupling is a
  server-side scheduling pass applied to idle windows in the already-transpiled circuit, so this number is
  expected to match the first run's exactly; that is not evidence DD had no effect, it's the wrong metric
  to look at for DD's effect (see the amplitude/MAE/candidate-zone comparisons below instead).
- **Readout error mitigation:** {"Yes (measurement twirling, same as the first run)" if metadata.mitigation_applied else "No"}

## Three-way amplitude landscape comparison

| Comparison | MAE |
|---|---:|
| Simulated vs. first hardware run (no DD) | {mae_sim_first:.5f} |
| Simulated vs. DD hardware run | {mae_sim_dd:.5f} |
| First hardware run vs. DD hardware run | {mae_first_dd:.5f} |

## Candidate zones: classical vs. first-run hybrid vs. DD-run hybrid

Same magnitude-from-hardware/phase-from-simulator hybrid methodology as
`7QUBIT_HW_RESULTS.md` (NOT a full hardware reconstruction -- phase is unmeasurable
from a single Sampler setting; see that report for the full explanation), computed
identically for both hardware runs so the comparison isolates DD.

### Candidate zones -- classical (numpy FFT, noiseless, for reference)

{_zone_table(zones_classical)}

### Candidate zones -- first hardware run hybrid (no DD)

{_zone_table(zones_first_hybrid)}

### Candidate zones -- DD hardware run hybrid

{_zone_table(zones_dd_hybrid)}

## Verdict

{verdict}

## Outputs from this run

| PNG file | Pathway that produced it |
|---|---|
{png_table}

## Console warnings / notable events during this run

{events_section}
"""

    path = OUTPUT_DIR / "7QUBIT_HW_DD_RESULTS.md"
    path.write_text(content)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qubits", type=int, default=7, help="size of the encoding register, shared by both pathways (default: 7)"
    )
    parser.add_argument(
        "--predict-steps", type=int, default=10, help="number of gaps to predict past index 49 (default: 10)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="restrict the forward prediction to the top-K dominant frequency components "
        "(default: use all, i.e. assume the known 49-gap window is exactly one period)",
    )
    parser.add_argument("--hardware", action="store_true", help="run the encoded circuit on real IBM Quantum hardware")
    parser.add_argument(
        "--dynamical-decoupling",
        action="store_true",
        help="with --hardware, also submit a second prediction-circuit run with dynamical decoupling "
        f"({DD_SEQUENCE_TYPE}) enabled, and compare it against the first hardware run (job "
        f"{FIRST_HARDWARE_RUN_JOB_ID!r}) -- no-op without --hardware",
    )
    parser.add_argument("--backend", type=str, default=None, help="IBM backend name (default: least busy)")
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="shots for hardware execution (default: adaptive -- "
        f"{ADAPTIVE_SHOTS_LOW_QUEUE} if the selected backend's queue is shallow, "
        f"{ADAPTIVE_SHOTS_HIGH_QUEUE} if it's deep)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        gaps = prime_gaps(FIRST_50_PRIMES)
        print(f"First 50 primes: {FIRST_50_PRIMES[0]}..{FIRST_50_PRIMES[-1]}")
        print(f"Gap sequence ({len(gaps)} gaps): {gaps.astype(int).tolist()}")

        angles = normalize_to_angles(gaps)
        circuit = build_full_circuit(angles, args.qubits)
        print(f"\nEncoding circuit: {args.qubits} qubits, {circuit.size()} gates "
              f"({-(-len(gaps) // args.qubits)} re-upload chunks)")

        print("\nVerifying before touching hardware...")
        result = verify(gaps, args.qubits)
        print(f"  Primes match independent sieve:            {result.primes_match_sieve}")
        print(f"  Statevector probabilities sum to 1:         {result.probabilities_sum_to_one}")
        print(f"  Matches from-scratch numpy reference sim:   {result.matches_reference_simulator}")
        print(f"  Real gap-sequence entropy:                  {result.real_entropy:.4f} bits")
        print(f"  Mean entropy over 50 shuffled controls:     {result.mean_shuffled_entropy:.4f} bits")
        print(f"  Ordering looks structured (real < shuffled): {result.ordering_looks_structured()}")
        dim = 2**args.qubits
        primes_below_dim = [p for p in FIRST_50_PRIMES if p < dim]
        print(
            "\nNote: probability is symmetric about index dim/2 (P(k) ~= P(dim-k)) because the "
            "pre-QFT state is entirely real-valued (only Ry and CX gates, no complex phases) -- that's "
            "a generic property of a QFT applied to any real input, not a sign of prime structure at "
            f"those specific basis-state indices. With dim={dim} indices, {len(primes_below_dim)} of them "
            f"({', '.join(str(p) for p in primes_below_dim)}) landing on primes by coincidence is expected, "
            "not significant on its own."
        )

        if not result.all_hard_checks_passed():
            raise SystemExit("Hard verification checks failed -- refusing to plot or touch hardware.")

        statevector = Statevector(circuit)
        probabilities = statevector.probabilities()
        phases = np.angle(statevector.data)

        gap_path = plot_gap_sequence(gaps)
        landscape_path = plot_amplitude_landscape(probabilities, phases, args.qubits)
        portrait_path = plot_frequency_portrait(probabilities, args.qubits)
        print(f"\nWrote {gap_path}")
        print(f"Wrote {landscape_path}")
        print(f"Wrote {portrait_path}")
        png_paths: list[tuple[Path, str]] = [
            (gap_path, "raw data (neither pathway)"),
            (landscape_path, "landscape re-upload circuit (Qiskit simulator)"),
            (portrait_path, "landscape re-upload circuit (Qiskit simulator)"),
        ]

        print("\nVerifying the prediction pathway (classical and quantum) before predicting anything...")
        normalized_vector, _ = amplitude_encode(gaps, args.qubits)
        amp_result = verify_amplitude_qft_roundtrip(normalized_vector)
        quantum_fft_ok = verify_quantum_fft_matches_padded_numpy(gaps, args.qubits)
        extrap_ok = verify_extrapolation_roundtrip(gaps)
        print(f"  Amplitude-encoded QFT matches numpy DFT reference:         {amp_result.matches_numpy_reference}")
        print(f"  QFT -> inverse QFT round trip recovers amplitudes:         {amp_result.roundtrip_matches}")
        print(f"  quantum_fft matches np.fft.fft on the zero-padded array:   {quantum_fft_ok}")
        print(f"  Full-spectrum extrapolation round trip matches known gaps: {extrap_ok}")
        if not (amp_result.all_passed() and quantum_fft_ok and extrap_ok):
            _log_event("Hard check failed in the prediction pathway -- refusing to predict.")
            raise SystemExit("Prediction-pathway verification failed -- refusing to predict.")

        effective_top_k = args.top_k if args.top_k is not None else QUANTUM_DEFAULT_TOP_K
        if args.top_k is None:
            _log_event(
                f"--top-k not given: using top-{QUANTUM_DEFAULT_TOP_K} for the paired backward-verification "
                "comparison below so classical and quantum are compared at the same truncation level "
                "(classical's own default elsewhere in this run is still \"keep everything\")"
            )

        print(
            "\nBackward verification -- predicting gaps 40-49 (primes 41-50) from only the first 40 primes, "
            f"classical vs. quantum circuit, both at top_k={effective_top_k}:"
        )
        backward_classical = backward_verify(top_k=effective_top_k)
        backward_quantum = backward_verify_quantum(args.qubits, top_k=effective_top_k)
        for step, (c_pred, q_pred, actual) in enumerate(
            zip(backward_classical.predicted_gaps, backward_quantum.predicted_gaps, backward_classical.actual_gaps),
            start=1,
        ):
            print(
                f"  gap {39 + step}: actual {actual:5.0f}  classical {c_pred:6.2f} (err {abs(c_pred - actual):5.2f})  "
                f"quantum {q_pred:6.2f} (err {abs(q_pred - actual):5.2f})"
            )
        print(f"  MAE classical:              {backward_classical.mae:.3f}")
        print(f"  MAE quantum:                {backward_quantum.mae:.3f}")
        print(f"  MAE baseline (mean gap):    {backward_classical.baseline_mean_mae:.3f}")
        print(f"  MAE baseline (repeat last): {backward_classical.baseline_last_repeat_mae:.3f}")
        print(f"  Classical beats baselines:  {backward_classical.beats_baselines()}")
        print(f"  Quantum beats baselines:    {backward_quantum.beats_baselines()}")
        mae_diff = abs(backward_quantum.mae - backward_classical.mae)
        if mae_diff < 1e-9:
            verdict = (
                f"Quantum vs classical MAE differ by {mae_diff:.2e} -- indistinguishable from floating-point "
                "noise between the two computational paths at this top_k."
            )
        else:
            verdict = (
                f"Quantum vs classical MAE differ by {mae_diff:.4f} at matched top_k={effective_top_k}. This is "
                "NOT a sign-convention or rescaling bug -- that's separately proven to floating-point precision "
                "(~1e-13) by the \"quantum_fft matches np.fft.fft on the zero-padded array\" hard check above, "
                "which compares the two pathways' spectra bin-for-bin on the identical (padded) array. This MAE "
                "gap instead reflects that amplitude encoding's mandatory zero-padding "
                f"(39 known gaps into a {2**args.qubits}-slot register) dilutes spectral power "
                f"across many more bins, so \"top-{effective_top_k}\" selects a genuinely coarser, different set "
                "of frequencies than the classical path's native unpadded spectrum -- an inherent cost of "
                "quantum amplitude encoding onto a power-of-2 register, not a computational error."
            )
        _log_event(verdict)

        print(
            f"\nForward prediction: {args.predict_steps} steps past gap 49 "
            "(quantum circuit is primary per this run's request; classical shown for comparison):"
        )
        prediction_classical = predict(gaps, args.predict_steps, args.top_k, start_prime=FIRST_50_PRIMES[-1])
        prediction_quantum = predict_quantum(
            gaps, args.predict_steps, args.qubits, args.top_k, start_prime=FIRST_50_PRIMES[-1]
        )
        for step, (c_gap, c_cand, q_gap, q_cand) in enumerate(
            zip(
                prediction_classical.predicted_gaps,
                prediction_classical.predicted_primes,
                prediction_quantum.predicted_gaps,
                prediction_quantum.predicted_primes,
            ),
            start=1,
        ):
            print(
                f"  gap {49 + step}: classical {c_gap:6.2f} -> candidate {c_cand:7.1f}   "
                f"quantum {q_gap:6.2f} -> candidate {q_cand:7.1f}"
            )

        wave_path = plot_extended_wave(gaps, prediction_classical.predicted_gaps, prediction_quantum.predicted_gaps)
        print(f"\nWrote {wave_path}")
        png_paths.append((wave_path, "both -- classical and quantum series shown together, clearly labeled"))

        prime_pool = sieve_primes(200)
        zones_classical = spectral_candidate_zones(gaps, args.predict_steps, FIRST_50_PRIMES[-1], prime_pool)
        zones_quantum = spectral_candidate_zones(
            gaps, args.predict_steps, FIRST_50_PRIMES[-1], prime_pool, n_qubits=args.qubits
        )
        print("\nCandidate zones, classical (ranked by spectral power fraction retained):")
        for zone in zones_classical:
            primes_str = ", ".join(str(p) for p in zone.nearest_primes)
            print(
                f"  top-{zone.frequencies_used} frequencies (power fraction {zone.power_fraction:.1%}): "
                f"nearest primes [{primes_str}], mean distance from raw candidate {zone.distances.mean():.2f}"
            )
        print("\nCandidate zones, quantum circuit (ranked by spectral power fraction retained):")
        for zone in zones_quantum:
            primes_str = ", ".join(str(p) for p in zone.nearest_primes)
            print(
                f"  top-{zone.frequencies_used} frequencies (power fraction {zone.power_fraction:.1%}): "
                f"nearest primes [{primes_str}], mean distance from raw candidate {zone.distances.mean():.2f}"
            )

        if args.hardware:
            print()
            counts, landscape_hw_meta = run_on_hardware(circuit, args.backend, args.shots)
            total = sum(counts.values())
            print(f"Job {landscape_hw_meta.job_id}, top 10 measured bitstrings out of {total} shots (landscape circuit):")
            for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
                print(f"  {bitstring}: {count} ({count / total:.1%})")
            overlay_path = plot_hardware_overlay(probabilities, counts, args.qubits, landscape_hw_meta.backend_name)
            print(f"\nWrote {overlay_path}")
            png_paths.append((overlay_path, "landscape circuit vs. real IBM hardware"))

            _log_event(
                "--hardware set: the prediction circuit is also submitted to hardware, but its measured "
                "output is Born-rule probabilities only (no phase) -- the prediction MAE/candidates reported "
                "above still come from the noiseless simulator regardless; recovering phase from hardware "
                "would need full state tomography, out of scope."
            )
            prediction_normalized, prediction_norm = amplitude_encode(gaps, args.qubits)
            prediction_circuit = build_amplitude_circuit(prediction_normalized)
            prediction_circuit.append(QFTGate(args.qubits).inverse(), range(args.qubits))
            prediction_sim_statevector = Statevector(prediction_circuit).data
            prediction_sim_probs = np.abs(prediction_sim_statevector) ** 2

            print(f"\nSubmitting the prediction circuit ({args.qubits} qubits) to real IBM Quantum hardware...")
            pred_counts, pred_hw_meta = run_on_hardware(prediction_circuit, args.backend, args.shots)
            pred_total = sum(pred_counts.values())
            print(
                f"\n*** JOB ID: {pred_hw_meta.job_id} *** backend: {pred_hw_meta.backend_name}, "
                f"shots: {pred_hw_meta.shots}, transpiled depth: {pred_hw_meta.transpiled_depth}, "
                f"mitigation: {pred_hw_meta.mitigation_applied}"
            )
            print(f"Top 10 measured bitstrings out of {pred_total} shots (prediction circuit):")
            for bitstring, count in sorted(pred_counts.items(), key=lambda kv: -kv[1])[:10]:
                print(f"  {bitstring}: {count} ({count / pred_total:.1%})")

            hardware_probabilities = hardware_counts_to_probabilities(pred_counts, args.qubits)
            hardware_mae = float(np.mean(np.abs(hardware_probabilities - prediction_sim_probs)))
            print(f"\nHardware vs. simulated amplitude-landscape MAE (the noise floor): {hardware_mae:.5f}")

            hw_landscape_path = plot_hardware_amplitude_landscape(
                hardware_probabilities, args.qubits, pred_hw_meta.backend_name
            )
            hw_comparison_path = plot_hardware_vs_sim_comparison(
                prediction_sim_probs, hardware_probabilities, args.qubits, pred_hw_meta.backend_name
            )
            hw_portrait_path = plot_hardware_frequency_portrait(prediction_sim_probs, hardware_probabilities, args.qubits)
            print(f"Wrote {hw_landscape_path}")
            print(f"Wrote {hw_comparison_path}")
            print(f"Wrote {hw_portrait_path}")

            zones_hybrid = hardware_informed_candidate_zones(
                hardware_probabilities,
                prediction_sim_statevector,
                prediction_norm,
                args.qubits,
                len(gaps),
                args.predict_steps,
                FIRST_50_PRIMES[-1],
                prime_pool,
            )
            print(
                "\nCandidate zones, magnitude-from-hardware/phase-from-simulator HYBRID "
                "(NOT a full hardware reconstruction -- see report for why):"
            )
            for zone in zones_hybrid:
                primes_str = ", ".join(str(p) for p in zone.nearest_primes)
                print(
                    f"  top-{zone.frequencies_used} frequencies (power fraction {zone.power_fraction:.1%}): "
                    f"nearest primes [{primes_str}], mean distance from raw candidate {zone.distances.mean():.2f}"
                )

            hw_png_paths = [
                (hw_landscape_path, "prediction circuit, hardware only"),
                (hw_comparison_path, "prediction circuit, simulated + hardware side by side"),
                (hw_portrait_path, "prediction circuit, simulated + hardware overlay"),
            ]

            warnings_logged_through = len(caught)
            for w in caught[:warnings_logged_through]:
                _log_event(f"Warning: {warnings.formatwarning(w.message, w.category, w.filename, w.lineno).strip()}")

            hw_report_path = write_hardware_report(
                args,
                pred_hw_meta,
                hardware_probabilities,
                prediction_sim_probs,
                zones_classical,
                zones_quantum,
                zones_hybrid,
                hw_png_paths,
            )
            print(f"\nWrote {hw_report_path}")

            if args.dynamical_decoupling:
                print(
                    f"\nSubmitting a second prediction-circuit run with dynamical decoupling "
                    f"({DD_SEQUENCE_TYPE}) enabled, for comparison against first-run job "
                    f"{FIRST_HARDWARE_RUN_JOB_ID}..."
                )
                dd_backend = args.backend or "ibm_kingston"  # match the first run exactly, per this comparison's request
                dd_counts, dd_hw_meta = run_on_hardware(
                    prediction_circuit, dd_backend, args.shots, dynamical_decoupling=True
                )
                dd_total = sum(dd_counts.values())
                print(
                    f"\n*** DD JOB ID: {dd_hw_meta.job_id} *** backend: {dd_hw_meta.backend_name}, "
                    f"shots: {dd_hw_meta.shots}, transpiled depth: {dd_hw_meta.transpiled_depth}, "
                    f"DD sequence: {DD_SEQUENCE_TYPE}"
                )
                print(f"Top 10 measured bitstrings out of {dd_total} shots (DD run):")
                for bitstring, count in sorted(dd_counts.items(), key=lambda kv: -kv[1])[:10]:
                    print(f"  {bitstring}: {count} ({count / dd_total:.1%})")

                dd_probabilities = hardware_counts_to_probabilities(dd_counts, args.qubits)
                first_hw_probabilities = fetch_first_hardware_run(args.qubits)

                mae_sim_first = float(np.mean(np.abs(first_hw_probabilities - prediction_sim_probs)))
                mae_sim_dd = float(np.mean(np.abs(dd_probabilities - prediction_sim_probs)))
                print(
                    f"\nMAE vs. simulated -- first run: {mae_sim_first:.5f}, DD run: {mae_sim_dd:.5f} "
                    f"({'improved' if mae_sim_dd < mae_sim_first else 'did not improve'})"
                )

                dd_landscape_path = plot_hardware_dd_amplitude_landscape(
                    dd_probabilities, args.qubits, dd_hw_meta.backend_name
                )
                dd_comparison_path = plot_hardware_dd_vs_first_run_comparison(
                    prediction_sim_probs, first_hw_probabilities, dd_probabilities, args.qubits, dd_hw_meta.backend_name
                )
                dd_portrait_path = plot_hardware_dd_frequency_portrait(
                    prediction_sim_probs, first_hw_probabilities, dd_probabilities, args.qubits
                )
                print(f"Wrote {dd_landscape_path}")
                print(f"Wrote {dd_comparison_path}")
                print(f"Wrote {dd_portrait_path}")

                zones_first_hybrid = hardware_informed_candidate_zones(
                    first_hw_probabilities,
                    prediction_sim_statevector,
                    prediction_norm,
                    args.qubits,
                    len(gaps),
                    args.predict_steps,
                    FIRST_50_PRIMES[-1],
                    prime_pool,
                )
                zones_dd_hybrid = hardware_informed_candidate_zones(
                    dd_probabilities,
                    prediction_sim_statevector,
                    prediction_norm,
                    args.qubits,
                    len(gaps),
                    args.predict_steps,
                    FIRST_50_PRIMES[-1],
                    prime_pool,
                )
                print(
                    f"\nHybrid candidate-zone mean nearest-prime distance (max frequencies kept) -- "
                    f"classical: {zones_classical[-1].distances.mean():.2f}, "
                    f"first run: {zones_first_hybrid[-1].distances.mean():.2f}, "
                    f"DD run: {zones_dd_hybrid[-1].distances.mean():.2f}"
                )

                dd_png_paths = [
                    (dd_landscape_path, "prediction circuit, DD hardware run only"),
                    (dd_comparison_path, "prediction circuit, simulated + first hardware run + DD hardware run"),
                    (dd_portrait_path, "prediction circuit, simulated + first hardware run + DD hardware run overlay"),
                ]

                for w in caught[warnings_logged_through:]:
                    _log_event(f"Warning: {warnings.formatwarning(w.message, w.category, w.filename, w.lineno).strip()}")
                warnings_logged_through = len(caught)

                dd_report_path = write_hardware_dd_report(
                    args,
                    dd_hw_meta,
                    prediction_sim_probs,
                    first_hw_probabilities,
                    dd_probabilities,
                    zones_classical,
                    zones_first_hybrid,
                    zones_dd_hybrid,
                    dd_png_paths,
                )
                print(f"\nWrote {dd_report_path}")
        else:
            _log_event("--hardware not set: all pathways ran on the noiseless statevector simulator only.")
            if args.dynamical_decoupling:
                _log_event("--dynamical-decoupling was set without --hardware -- ignored, it has no effect without --hardware.")

        if not args.hardware:
            for w in caught:
                _log_event(f"Warning: {warnings.formatwarning(w.message, w.category, w.filename, w.lineno).strip()}")

        report_path = write_results_report(
            args,
            effective_top_k,
            backward_classical,
            backward_quantum,
            mae_diff,
            verdict,
            prediction_classical,
            prediction_quantum,
            zones_classical,
            zones_quantum,
            png_paths,
        )
        print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
