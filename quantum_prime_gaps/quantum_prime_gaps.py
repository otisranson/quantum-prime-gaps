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
(4 by default) is repeatedly re-loaded with `Ry` rotations from successive
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
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector

matplotlib.use("Agg")

OUTPUT_DIR = Path(__file__).parent / "output"

# Register size for the amplitude-encoding prediction pathway below -- 64 dimensions,
# comfortably >= the 49 (forward) or 39 (backward-verification) known gaps, with the
# same size used for both so the two runs are directly comparable.
AMPLITUDE_QUBITS = 6

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
# 1. Gate-level pathway (dim = 2**AMPLITUDE_QUBITS, zero-padded): a real Quantum
#    Fourier Transform of real time-domain samples, exactly invertible. Used only to
#    sanity-check the inverse QFT (`verify_amplitude_qft_roundtrip`) -- it can only
#    reconstruct the same known points it was given, not new ones.
# 2. Extrapolation pathway (plain numpy, unpadded, one period = the true number of
#    known gaps): reads the frequency domain as a continuous function of time and
#    evaluates it *past* the known window (`fourier_extrapolate`). This is what
#    "time evolution forward" actually means here, and it's classical post-processing
#    on the same DFT data -- not a claim that the gate-level IQFT itself produces
#    future values, which no fixed-size inverse transform can do.
#
# The same `predict()` function is called for the forward run and for backward
# verification (predicting known-but-withheld gaps from earlier ones), so a bad
# backward-verification score reflects the mechanism's real predictive power, not a
# quirk of two different code paths.


def amplitude_encode(values: np.ndarray, n_qubits: int = AMPLITUDE_QUBITS) -> tuple[np.ndarray, float]:
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


@dataclass
class PredictionResult:
    known_gaps: np.ndarray
    predicted_gaps: np.ndarray
    top_k: int | None
    predicted_primes: np.ndarray


def predict(gaps: np.ndarray, n_predict: int, top_k: int | None, start_prime: int) -> PredictionResult:
    """The single prediction mechanism used identically for the forward run and for
    backward verification: only `gaps`, `n_predict`, and `start_prime` differ between
    the two callers."""
    predicted_gaps = fourier_extrapolate(gaps, n_predict, top_k=top_k)
    predicted_primes = start_prime + np.cumsum(predicted_gaps)
    return PredictionResult(gaps, predicted_gaps, top_k, predicted_primes)


@dataclass
class PrimeCandidate:
    frequencies_used: int
    power_fraction: float
    predicted_gaps: np.ndarray
    raw_candidates: np.ndarray
    nearest_primes: list[int]
    distances: np.ndarray


def spectral_candidate_zones(
    values: np.ndarray,
    n_predict: int,
    start_prime: int,
    prime_pool: list[int],
    max_levels: int = 5,
) -> list[PrimeCandidate]:
    """Read where amplitude concentrates in the frequency domain and turn that into
    ranked candidate primes. Each "zone" is a reconstruction built from the top-k
    strongest frequency components (k = 1, 2, 3, ...), weighted by the cumulative
    fraction of Parseval spectral power those components represent -- the same
    quantity plotted as the amplitude landscape, not an invented probability. Weight
    approaches 1.0 as more components are included; the last zone (using every
    component) matches the default `top_k=None` prediction."""
    n = len(values)
    spectrum = np.fft.fft(values)
    ranked = _ranked_frequency_components(spectrum, n)
    total_power = float(sum(power for _, power in ranked))

    zones = []
    for level in range(1, min(max_levels, len(ranked)) + 1):
        kept_power = float(sum(power for _, power in ranked[:level]))
        power_fraction = kept_power / total_power if total_power > 0 else 0.0
        predicted_gaps = fourier_extrapolate(values, n_predict, top_k=level)
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


@dataclass
class BackwardVerificationResult:
    predicted_gaps: np.ndarray
    actual_gaps: np.ndarray
    mae: float
    baseline_mean_mae: float
    baseline_last_repeat_mae: float

    def beats_baselines(self) -> bool:
        return self.mae < min(self.baseline_mean_mae, self.baseline_last_repeat_mae)


def backward_verify(top_k: int | None = None, n_predict: int = 10) -> BackwardVerificationResult:
    """The accuracy check: feed in the first 40 primes (39 known gaps), predict the
    next `n_predict` gaps (indices 40..49, i.e. exactly the gaps needed to know primes
    41..50), and compare against the real values, via the exact same `predict()` used
    for the forward run -- a bad score here means the mechanism, not just the specific
    forward guess, doesn't work. Two naive baselines (repeat the mean known gap;
    repeat the last known gap) are reported alongside so "reasonable accuracy" has a
    reference point, the same role the shuffled-control comparison plays in `verify()`.
    """
    all_gaps = prime_gaps(FIRST_50_PRIMES)
    known_gaps = all_gaps[:39]
    actual_gaps = all_gaps[39 : 39 + n_predict]
    start_prime = FIRST_50_PRIMES[39]  # the 40th prime

    result = predict(known_gaps, n_predict, top_k, start_prime)
    mae = float(np.mean(np.abs(result.predicted_gaps - actual_gaps)))

    baseline_mean_mae = float(np.mean(np.abs(known_gaps.mean() - actual_gaps)))
    baseline_last_repeat_mae = float(np.mean(np.abs(known_gaps[-1] - actual_gaps)))

    return BackwardVerificationResult(
        predicted_gaps=result.predicted_gaps,
        actual_gaps=actual_gaps,
        mae=mae,
        baseline_mean_mae=baseline_mean_mae,
        baseline_last_repeat_mae=baseline_last_repeat_mae,
    )


def run_on_hardware(circuit: QuantumCircuit, backend_name: str | None, shots: int) -> tuple[dict[str, int], str]:
    """Transpile `circuit` (with measurements) for a real IBM Quantum backend and
    run it via the Sampler primitive, returning (bitstring -> count, backend name).

    Needs IBM Quantum credentials: either the QISKIT_IBM_TOKEN environment
    variable (see README.md for how to set it), or an account already saved
    locally via QiskitRuntimeService.save_account(channel=..., token=...).
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()
    backend = service.backend(backend_name) if backend_name else service.least_busy(min_num_qubits=circuit.num_qubits)
    print(f"Running on hardware backend: {backend.name}")

    measured = circuit.copy()
    measured.measure_all()
    transpiled = transpile(measured, backend=backend, optimization_level=3)

    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    print(f"Submitted job {job.job_id()}, waiting for results...")
    result = job.result()
    counts = result[0].data.meas.get_counts()
    return dict(counts), backend.name


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


def plot_extended_wave(known_gaps: np.ndarray, predicted_gaps: np.ndarray) -> Path:
    """The gap sequence known so far, plus the spectrally-extrapolated prediction
    past it, with the boundary between the two marked explicitly."""
    fig, ax = plt.subplots(figsize=(10, 4))
    known_idx = np.arange(1, len(known_gaps) + 1)
    predicted_idx = np.arange(len(known_gaps) + 1, len(known_gaps) + len(predicted_gaps) + 1)

    ax.plot(known_idx, known_gaps, "o-", color="tab:blue", label="known")
    ax.plot(predicted_idx, predicted_gaps, "o--", color="tab:red", label="predicted")
    ax.axvline(len(known_gaps) + 0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("gap index")
    ax.set_ylabel("gap size")
    ax.set_title("Prime gap wave: known vs. spectrally-extrapolated prediction")
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


def plot_hardware_overlay(sim_probabilities: np.ndarray, counts: dict[str, int], n_qubits: int, backend_name: str) -> Path:
    """Overlay real QUANTUM HARDWARE measured probabilities against the SIMULATED
    statevector probabilities for the same circuit, so noise/decoherence effects
    are visible directly against the ideal result."""
    dim = 2**n_qubits
    shots = sum(counts.values())
    hardware_probabilities = hardware_counts_to_probabilities(counts, n_qubits)

    x = np.arange(dim)
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, sim_probabilities, width, label="simulated", color="tab:blue")
    ax.bar(x + width / 2, hardware_probabilities, width, label=f"quantum hardware ({backend_name})", color="tab:red")
    ax.set_xlabel("computational basis state index")
    ax.set_ylabel("probability")
    ax.set_title(f"QUANTUM HARDWARE ({backend_name}, {shots} shots) vs SIMULATED amplitude landscape")
    ax.legend()
    fig.tight_layout()

    backend_slug = "".join(c if c.isalnum() else "_" for c in backend_name)
    path = OUTPUT_DIR / f"amplitude_landscape_quantum_{backend_slug}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=4, help="size of the encoding register (default: 4)")
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
    parser.add_argument("--backend", type=str, default=None, help="IBM backend name (default: least busy)")
    parser.add_argument("--shots", type=int, default=4096, help="shots for hardware execution (default: 4096)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

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
    print(
        "\nNote: probability is symmetric about index dim/2 (P(k) ~= P(dim-k)) because the "
        "pre-QFT state is entirely real-valued (only Ry and CX gates, no complex phases) -- that's "
        "a generic property of a QFT applied to any real input, not a sign of prime structure at "
        "those specific basis-state indices. With only dim=16 indices, some of them landing on "
        "small primes (2, 3, 5, 7, 11, 13) by coincidence is expected, not significant on its own."
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

    print("\nVerifying the prediction pathway before predicting anything...")
    normalized_vector, _ = amplitude_encode(gaps)
    amp_result = verify_amplitude_qft_roundtrip(normalized_vector)
    extrap_ok = verify_extrapolation_roundtrip(gaps)
    print(f"  Amplitude-encoded QFT matches numpy DFT reference:        {amp_result.matches_numpy_reference}")
    print(f"  QFT -> inverse QFT round trip recovers amplitudes:        {amp_result.roundtrip_matches}")
    print(f"  Full-spectrum extrapolation round trip matches known gaps: {extrap_ok}")
    if not (amp_result.all_passed() and extrap_ok):
        raise SystemExit("Prediction-pathway verification failed -- refusing to predict.")

    print(f"\nBackward verification -- predicting gaps 40-49 (primes 41-50) from only the first 40 primes:")
    backward = backward_verify(top_k=args.top_k)
    for step, (pred, actual) in enumerate(zip(backward.predicted_gaps, backward.actual_gaps), start=1):
        print(f"  gap {39 + step}: predicted {pred:6.2f}  actual {actual:5.0f}  error {abs(pred - actual):5.2f}")
    print(f"  MAE (model):                 {backward.mae:.3f}")
    print(f"  MAE (baseline, mean gap):    {backward.baseline_mean_mae:.3f}")
    print(f"  MAE (baseline, repeat last): {backward.baseline_last_repeat_mae:.3f}")
    print(f"  Model beats both naive baselines: {backward.beats_baselines()}")

    print(f"\nForward prediction: {args.predict_steps} steps past gap 49 (top_k={args.top_k}):")
    prediction = predict(gaps, args.predict_steps, args.top_k, start_prime=FIRST_50_PRIMES[-1])
    for step, (gap, candidate) in enumerate(zip(prediction.predicted_gaps, prediction.predicted_primes), start=1):
        print(f"  gap {49 + step}: predicted {gap:6.2f}  -> raw candidate {candidate:7.1f}")

    wave_path = plot_extended_wave(gaps, prediction.predicted_gaps)
    print(f"\nWrote {wave_path}")

    prime_pool = sieve_primes(200)
    zones = spectral_candidate_zones(gaps, args.predict_steps, FIRST_50_PRIMES[-1], prime_pool)
    print("\nCandidate zones (ranked by spectral power fraction retained):")
    for zone in zones:
        primes_str = ", ".join(str(p) for p in zone.nearest_primes)
        print(
            f"  top-{zone.frequencies_used} frequencies (power fraction {zone.power_fraction:.1%}): "
            f"nearest primes [{primes_str}], mean distance from raw candidate {zone.distances.mean():.2f}"
        )

    if args.hardware:
        print()
        counts, backend_name = run_on_hardware(circuit, args.backend, args.shots)
        total = sum(counts.values())
        print(f"Top 10 measured bitstrings out of {total} shots:")
        for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {bitstring}: {count} ({count / total:.1%})")

        overlay_path = plot_hardware_overlay(probabilities, counts, args.qubits, backend_name)
        print(f"\nWrote {overlay_path}")


if __name__ == "__main__":
    main()
