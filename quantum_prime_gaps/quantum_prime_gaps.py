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
