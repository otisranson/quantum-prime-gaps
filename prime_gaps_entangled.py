# Copyright 2026 Otis Ranson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Rebuilt prime-gap experiment with explicit gate-based entanglement.

Architecture (2-qubit prototype):
  1. Bell-pair foundation  — H on q0, CNOT q0→q1
  2. Gap encoding          — RY(θ₀) on q0, RY(θ₁) on q1
                             θᵢ = gap[i] * π / max(gaps)
  3. Inverse QFT           — standard 2-qubit iQFT
  4. Measurement

No QuantumCircuit.initialize, no StatePrep — explicit gates only.

Outputs:
  - ASCII circuit diagram
  - Statevector table (AerSimulator)
  - Shot-counts histogram
  - Pairwise correlation r and mutual information (bits)
  - Significance vs permutation null (z-score)

Run on AerSimulator first; hardware section gated behind --hardware flag.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Prime data
# ---------------------------------------------------------------------------

FIRST_50_PRIMES: list[int] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
]

# First 2 gaps: 3-2=1, 5-3=2
_GAPS = [FIRST_50_PRIMES[i + 1] - FIRST_50_PRIMES[i] for i in range(len(FIRST_50_PRIMES) - 1)]
MAX_GAP = max(_GAPS)


def gap_to_angle(gap: int) -> float:
    """Map a prime gap to an RY rotation angle in [0, π]."""
    return gap * math.pi / MAX_GAP


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_circuit(n_qubits: int = 2) -> "QuantumCircuit":
    """Build: Bell pair → gap encoding → inverse QFT."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate

    gaps = _GAPS[:n_qubits]
    angles = [gap_to_angle(g) for g in gaps]

    qc = QuantumCircuit(n_qubits, n_qubits)

    # 1. Bell-pair foundation (explicit entanglement seed)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier(label="Bell pair")

    # 2. Gap encoding via RY rotations
    for i, theta in enumerate(angles):
        qc.ry(theta, i)
    qc.barrier(label=f"gaps {gaps}")

    # 3. Inverse QFT
    iqft_gate = QFTGate(n_qubits).inverse()
    qc.append(iqft_gate, range(n_qubits))
    qc.barrier(label="iQFT")

    # 4. Measure
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ---------------------------------------------------------------------------
# Entanglement metrics
# ---------------------------------------------------------------------------

def counts_to_bits(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """Convert Qiskit counts to an (N_shots, n_qubits) binary array.

    Qiskit bitstrings are MSB-left; qubit 0 is the rightmost character.
    """
    shots = sum(counts.values())
    bits = np.zeros((shots, n_qubits), dtype=np.int8)
    row = 0
    for bitstring, count in counts.items():
        for _ in range(count):
            for q in range(n_qubits):
                bits[row, q] = int(bitstring[n_qubits - 1 - q])
            row += 1
    return bits


def pearson_r(bits: np.ndarray) -> float:
    """Pearson correlation between q0 and q1 columns."""
    a, b = bits[:, 0].astype(float), bits[:, 1].astype(float)
    denom = np.std(a) * np.std(b)
    if denom == 0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / denom)


def _entropy_bits(p: np.ndarray) -> float:
    """Shannon entropy in bits, skipping zero terms."""
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _miller_madow_correction(k_obs: int, n: int) -> float:
    """Miller-Madow additive bias correction term."""
    return (k_obs - 1) / (2 * n * math.log(2))


def mutual_information(bits: np.ndarray) -> float:
    """Miller-Madow corrected MI in bits between q0 and q1."""
    n = len(bits)
    # Marginals
    p0 = np.bincount(bits[:, 0], minlength=2) / n
    p1 = np.bincount(bits[:, 1], minlength=2) / n
    # Joint
    joint_idx = bits[:, 0] * 2 + bits[:, 1]
    p_joint = np.bincount(joint_idx, minlength=4) / n

    h0 = _entropy_bits(p0) + _miller_madow_correction(int(np.sum(p0 > 0)), n)
    h1 = _entropy_bits(p1) + _miller_madow_correction(int(np.sum(p1 > 0)), n)
    h_joint = _entropy_bits(p_joint) + _miller_madow_correction(int(np.sum(p_joint > 0)), n)

    return max(0.0, h0 + h1 - h_joint)


def null_mi_zscore(bits: np.ndarray, observed_mi: float, n_trials: int = 500) -> tuple[float, float]:
    """Permutation null: shuffle q1 column, recompute MI, return (mean_null, z-score)."""
    rng = np.random.default_rng(42)
    null_mis = np.empty(n_trials)
    for i in range(n_trials):
        shuffled = bits.copy()
        rng.shuffle(shuffled[:, 1])
        null_mis[i] = mutual_information(shuffled)
    mean_null = float(null_mis.mean())
    std_null = float(null_mis.std())
    z = (observed_mi - mean_null) / std_null if std_null > 0 else 0.0
    return mean_null, z


# ---------------------------------------------------------------------------
# Simulator run
# ---------------------------------------------------------------------------

def run_simulator(qc: "QuantumCircuit", shots: int = 8192) -> dict[str, int]:
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    from qiskit import transpile

    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    return result.get_counts()


# ---------------------------------------------------------------------------
# Hardware run
# ---------------------------------------------------------------------------

def run_hardware(qc: "QuantumCircuit", shots: int = 8192) -> dict[str, int]:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
    print(f"  Backend: {backend.name}  Status: {backend.status().status_msg}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)

    sampler = Sampler(mode=backend)
    job = sampler.run([isa_qc], shots=shots)
    print(f"  Job ID: {job.job_id()}")
    result = job.result()
    pub_result = result[0]
    bitarray = pub_result.data.c
    counts: dict[str, int] = {}
    for bs in bitarray.get_bitstrings():
        counts[bs] = counts.get(bs, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(label: str, counts: dict[str, int], n_qubits: int) -> None:
    shots = sum(counts.values())
    bits = counts_to_bits(counts, n_qubits)
    r = pearson_r(bits)
    mi = mutual_information(bits)
    mean_null, z = null_mi_zscore(bits, mi)

    print(f"\n{'─'*52}")
    print(f"  {label}")
    print(f"{'─'*52}")
    print(f"  Shots: {shots:,}")
    print()
    print("  Outcome counts:")
    for bs in sorted(counts):
        bar = "█" * int(40 * counts[bs] / shots)
        print(f"    |{bs}⟩  {counts[bs]:5d}  {bar}")
    print()
    print(f"  Pearson r (q0–q1):       {r:+.4f}")
    print(f"  MI corrected (bits):      {mi:.4f}")
    print(f"  Null MI mean (500 perms): {mean_null:.4f}")
    print(f"  z-score vs null:          {z:.2f}  {'★ significant' if z > 2 else '(not significant)'}")


def save_results(label: str, counts: dict[str, int], n_qubits: int, out_dir: Path) -> None:
    """Write a JSON results file alongside the screenshots."""
    import json

    shots = sum(counts.values())
    bits = counts_to_bits(counts, n_qubits)
    r = pearson_r(bits)
    mi = mutual_information(bits)
    mean_null, z = null_mi_zscore(bits, mi)

    slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    data = {
        "label": label,
        "n_qubits": n_qubits,
        "shots": shots,
        "counts": counts,
        "pearson_r": round(r, 6),
        "mi_bits": round(mi, 6),
        "null_mi_mean": round(mean_null, 6),
        "z_score": round(z, 4),
    }
    out_path = out_dir / f"results_{slug}.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", type=int, default=8192, help="Shot count (default 8192)")
    parser.add_argument("--hardware", action="store_true", help="Also run on ibm_kingston after simulator")
    args = parser.parse_args()

    n_qubits = 2
    gaps = _GAPS[:n_qubits]
    angles = [gap_to_angle(g) for g in gaps]

    print("=" * 52)
    print("  Quantum Prime Gaps — Explicit Entanglement v1")
    print("=" * 52)
    print(f"\n  Qubits: {n_qubits}")
    print(f"  First {n_qubits} prime gaps: {gaps}")
    print(f"  RY angles (rad): {[f'{a:.4f}' for a in angles]}")
    print(f"  Max gap (normalizer): {MAX_GAP}")

    qc = build_circuit(n_qubits)

    print("\n── Circuit ──────────────────────────────────────")
    print(qc.draw(output="text", fold=80))

    out_dir = Path("quantum_prime_gaps/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Simulator ────────────────────────────────────")
    sim_counts = run_simulator(qc, shots=args.shots)
    report("AerSimulator", sim_counts, n_qubits)
    save_results("AerSimulator", sim_counts, n_qubits, out_dir)

    if args.hardware:
        print("\n── Hardware (ibm_kingston) ───────────────────────")
        hw_counts = run_hardware(qc, shots=args.shots)
        report("ibm_kingston", hw_counts, n_qubits)
        save_results("ibm_kingston", hw_counts, n_qubits, out_dir)
    else:
        print("\n  [Hardware skipped — run with --hardware to submit to ibm_kingston]")

    print()


if __name__ == "__main__":
    main()
