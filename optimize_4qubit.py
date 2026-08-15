"""optimize_4qubit.py

Reduce 4-qubit Bell+RY+iQFT depth before hardware submission.

Baseline (from prior run, FakeSherbrooke opt=3):
  depth=81, ECR=18, sim root MI=0.373 bits, predicted HW retention ~66%

Strategies compared:
  0. Full iQFT (baseline)     — synth_qft_full, degree=0  (6 CP gates, long-range routing)
  1. Approx degree=1          — synth_qft_full, degree=1  (5 CP, removes CP(π/8))
  2. Approx degree=2          — synth_qft_full, degree=2  (3 CP, nearest-neighbor only ← key)
  3. Approx degree=3          — synth_qft_full, degree=3  (0 CP, just H+SWAP)
  4. Linear iQFT              — synth_qft_line (nearest-neighbor topology synthesis)

Winner criterion: lowest depth with root MI ≥ 0.336 bits (within 10% of 0.373 baseline).

Run:
  python optimize_4qubit.py               # compare all strategies on AerSimulator
  python optimize_4qubit.py --hardware    # also submit winner to ibm_kingston
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis.qft import synth_qft_full, synth_qft_line
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

# ── Prime gap data ─────────────────────────────────────────────────────────────

FIRST_50_PRIMES: list[int] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
]
_ALL_GAPS = [FIRST_50_PRIMES[i + 1] - FIRST_50_PRIMES[i] for i in range(len(FIRST_50_PRIMES) - 1)]
MAX_GAP = max(_ALL_GAPS)
N_QUBITS = 4
GAPS = _ALL_GAPS[:N_QUBITS]        # [1, 2, 2, 4]
ANGLES = [g * math.pi / MAX_GAP for g in GAPS]

SHOTS = 8_192
BASIS_GATES = ["ecr", "sx", "rz", "x"]

# T2_depth calibrated from 2q hardware run: 91% retention at depth=18
# retention = exp(-depth / T2_depth)  →  T2_depth = -18 / ln(0.9126) = 195.8
T2_DEPTH = 195.8
SIM_BASELINE_ROOT_MI = 0.373       # from prior 4q AerSimulator run

# ── Proxy backend ──────────────────────────────────────────────────────────────

BACKEND_PROXY = FakeSherbrooke()

# ── MI utilities ───────────────────────────────────────────────────────────────

def counts_to_bits(counts: dict, n: int) -> np.ndarray:
    """Qiskit bitstrings: MSB-left, qubit 0 is rightmost char."""
    total = sum(counts.values())
    bits = np.zeros((total, n), dtype=np.int8)
    row = 0
    for bs, cnt in counts.items():
        bs = bs.zfill(n)
        for _ in range(cnt):
            for q in range(n):
                bits[row, q] = int(bs[n - 1 - q])
            row += 1
    return bits


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _mm(k_obs: int, n: int) -> float:
    """Miller-Madow correction."""
    return (k_obs - 1) / (2 * n * math.log(2))


def mi_halves(bits: np.ndarray, left_qs: list[int], right_qs: list[int]) -> float:
    """MI (bits) between two groups of qubits treated as composite variables."""
    n = len(bits)
    left = np.zeros(n, dtype=np.int64)
    for i, q in enumerate(left_qs):
        left += bits[:, q].astype(np.int64) << i
    right = np.zeros(n, dtype=np.int64)
    for i, q in enumerate(right_qs):
        right += bits[:, q].astype(np.int64) << i

    ls = 2 ** len(left_qs)
    rs = 2 ** len(right_qs)
    p_l = np.bincount(left, minlength=ls) / n
    p_r = np.bincount(right, minlength=rs) / n
    p_j = np.bincount(left * rs + right, minlength=ls * rs) / n

    h_l = _entropy(p_l) + _mm(int(np.sum(p_l > 0)), n)
    h_r = _entropy(p_r) + _mm(int(np.sum(p_r > 0)), n)
    h_j = _entropy(p_j) + _mm(int(np.sum(p_j > 0)), n)
    return max(0.0, h_l + h_r - h_j)


def root_mi_from_counts(counts: dict) -> float:
    bits = counts_to_bits(counts, N_QUBITS)
    return mi_halves(bits, [0, 1], [2, 3])


def predicted_retention(depth: int) -> float:
    return math.exp(-depth / T2_DEPTH)


# ── Circuit builders ───────────────────────────────────────────────────────────

def bell_ry_prefix(qc: QuantumCircuit) -> None:
    """Append Bell pair + RY gap encoding in-place (no iQFT, no measure)."""
    qc.h(0)
    qc.cx(0, 1)
    for i, theta in enumerate(ANGLES):
        qc.ry(theta, i)


def build_full_iqft(approx_degree: int = 0) -> QuantumCircuit:
    """Bell+RY + synth_qft_full iQFT at given approximation_degree."""
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    bell_ry_prefix(qc)
    iqft = synth_qft_full(N_QUBITS, inverse=True, do_swaps=True,
                          approximation_degree=approx_degree)
    qc.compose(iqft, inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def build_line_iqft() -> QuantumCircuit:
    """Bell+RY + synth_qft_line iQFT (nearest-neighbor topology)."""
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    bell_ry_prefix(qc)
    # synth_qft_line returns a QuantumCircuit (no inverse kwarg); take .inverse()
    iqft = synth_qft_line(N_QUBITS, do_swaps=False).inverse()
    qc.compose(iqft, inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


# ── Transpile + simulate ───────────────────────────────────────────────────────

SIM = AerSimulator()


def transpile_and_score(qc: QuantumCircuit, coupling_map=None) -> tuple[QuantumCircuit, dict]:
    if coupling_map is not None:
        tqc = transpile(qc, coupling_map=coupling_map, basis_gates=BASIS_GATES,
                        optimization_level=3, seed_transpiler=42)
    else:
        tqc = transpile(qc, backend=BACKEND_PROXY, optimization_level=3, seed_transpiler=42)

    result = SIM.run(tqc, shots=SHOTS).result()
    counts = result.get_counts()
    ops = tqc.count_ops()
    ecr = ops.get("ecr", 0) + ops.get("cx", 0)
    return tqc, {
        "depth": tqc.depth(),
        "ecr": ecr,
        "root_mi": root_mi_from_counts(counts),
        "counts": counts,
    }


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class Result:
    label: str
    depth: int
    ecr: int
    root_mi: float
    retention: float
    predicted_hw_mi: float
    note: str = ""
    counts: dict | None = None

    def within_tolerance(self, baseline_mi: float, tol: float = 0.10) -> bool:
        return self.root_mi >= baseline_mi * (1.0 - tol)


# ── Main comparison ────────────────────────────────────────────────────────────

def run_comparison(coupling_map=None) -> list[Result]:
    results = []

    strategies = [
        ("Full iQFT (degree=0, baseline)", build_full_iqft(0), "All 6 CP gates; long-range routing required"),
        ("Approx iQFT degree=1",           build_full_iqft(1), "5/6 CPs; removes CP(π/8)"),
        ("Approx iQFT degree=2",           build_full_iqft(2), "3/6 CPs; nearest-neighbor only → no long-range SWAP"),
        ("Approx iQFT degree=3",           build_full_iqft(3), "0 CPs; H+SWAP only"),
        ("Linear iQFT",                    build_line_iqft(),  "synth_qft_line: nearest-neighbor synthesis (no do_swaps)"),
    ]

    for label, qc, note in strategies:
        print(f"  Transpiling & simulating: {label}...")
        tqc, info = transpile_and_score(qc, coupling_map)
        ret = predicted_retention(info["depth"])
        results.append(Result(
            label=label,
            depth=info["depth"],
            ecr=info["ecr"],
            root_mi=info["root_mi"],
            retention=ret,
            predicted_hw_mi=info["root_mi"] * ret,
            note=note,
            counts=info["counts"],
        ))
        print(f"    depth={info['depth']}, ECR={info['ecr']}, root MI={info['root_mi']:.4f} bits, "
              f"retention={ret:.1%}, pred HW MI={info['root_mi']*ret:.4f}")

    return results


def pick_winner(results: list[Result], baseline_mi: float = SIM_BASELINE_ROOT_MI) -> Result:
    eligible = [r for r in results if r.within_tolerance(baseline_mi)]
    if not eligible:
        print("  WARNING: no strategy meets the 10% MI tolerance; picking best MI anyway")
        eligible = sorted(results, key=lambda r: r.root_mi, reverse=True)[:1]
    return min(eligible, key=lambda r: r.depth)


def print_table(results: list[Result], winner: Result) -> None:
    print()
    print("─" * 90)
    print(f"  {'Strategy':<32} {'Depth':>6} {'ECR':>4} {'Root MI':>9} {'Retention':>9} {'Pred HW MI':>11}  {'OK?':>4}")
    print("─" * 90)
    for r in results:
        ok = "✓" if r.within_tolerance(SIM_BASELINE_ROOT_MI) else "✗"
        flag = "  ◀ WINNER" if r is winner else ""
        print(f"  {r.label:<32} {r.depth:>6} {r.ecr:>4} {r.root_mi:>9.4f} {r.retention:>9.1%} {r.predicted_hw_mi:>11.4f}  {ok:>4}{flag}")
    print("─" * 90)
    print(f"\n  Baseline: depth=81, ECR=18, sim root MI={SIM_BASELINE_ROOT_MI:.3f} bits, 10% floor=0.336 bits")


# ── Hardware run ───────────────────────────────────────────────────────────────

def run_hardware(qc: QuantumCircuit, shots: int = 8_192) -> dict:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
    print(f"  Backend: {backend.name}  status: {backend.status().status_msg}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=42)
    isa_qc = pm.run(qc)

    isa_ops = isa_qc.count_ops()
    hw_ecr = isa_ops.get("ecr", 0)
    hw_depth = isa_qc.depth()
    print(f"  ISA circuit: depth={hw_depth}, ECR={hw_ecr}")

    sampler = Sampler(mode=backend)
    job = sampler.run([isa_qc], shots=shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}  (waiting...)")
    result = job.result()

    pub_result = result[0]
    bitarray = pub_result.data.c
    counts: dict = {}
    for bs in bitarray.get_bitstrings():
        counts[bs] = counts.get(bs, 0) + 1

    return {
        "job_id": job_id,
        "backend": backend.name,
        "hw_depth": hw_depth,
        "hw_ecr": hw_ecr,
        "counts": counts,
    }


# ── Reporting helpers ──────────────────────────────────────────────────────────

def hw_report(hw_info: dict, sim_result: Result) -> None:
    counts = hw_info["counts"]
    bits = counts_to_bits(counts, N_QUBITS)
    hw_mi = mi_halves(bits, [0, 1], [2, 3])
    hw_ret = hw_mi / sim_result.root_mi if sim_result.root_mi > 0 else 0.0

    shots = sum(counts.values())
    print(f"\n{'─'*60}")
    print(f"  Hardware results  (ibm_kingston, job {hw_info['job_id']})")
    print(f"{'─'*60}")
    print(f"  ISA depth: {hw_info['hw_depth']}   ECR: {hw_info['hw_ecr']}")
    print(f"  Shots: {shots:,}")
    print()
    print("  Outcome counts (top 8):")
    for bs, cnt in sorted(counts.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * int(30 * cnt / shots)
        print(f"    |{bs}⟩  {cnt:5d} ({cnt/shots:.1%})  {bar}")
    print()
    print(f"  Sim root MI:       {sim_result.root_mi:.4f} bits")
    print(f"  Hardware root MI:  {hw_mi:.4f} bits")
    print(f"  HW/Sim retention:  {hw_ret:.1%}")
    print(f"  Predicted was:     {sim_result.retention:.1%}")
    print(f"{'─'*60}")

    return hw_mi, hw_ret


def save_json(results: list[Result], winner: Result,
              hw_info: dict | None, hw_mi: float | None, hw_ret: float | None,
              out_dir: Path) -> None:
    data = {
        "n_qubits": N_QUBITS,
        "gaps": GAPS,
        "angles_rad": [round(a, 6) for a in ANGLES],
        "shots": SHOTS,
        "t2_depth_calibrated": T2_DEPTH,
        "baseline_root_mi_sim": SIM_BASELINE_ROOT_MI,
        "mi_tolerance_pct": 10,
        "strategies": [
            {
                "label": r.label,
                "depth": r.depth,
                "ecr": r.ecr,
                "root_mi_sim": round(r.root_mi, 6),
                "retention_predicted": round(r.retention, 4),
                "predicted_hw_mi": round(r.predicted_hw_mi, 6),
                "within_tolerance": r.within_tolerance(SIM_BASELINE_ROOT_MI),
                "note": r.note,
            }
            for r in results
        ],
        "winner": winner.label,
    }
    if hw_info is not None:
        data["hardware"] = {
            "job_id": hw_info["job_id"],
            "backend": hw_info["backend"],
            "hw_depth": hw_info["hw_depth"],
            "hw_ecr": hw_info["hw_ecr"],
            "hw_root_mi": round(hw_mi, 6),
            "hw_sim_retention": round(hw_ret, 4),
            "counts": hw_info["counts"],
        }
    out_path = out_dir / "results_4qubit_optimized.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\n  Saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", action="store_true",
                        help="Submit winner to ibm_kingston after comparison")
    parser.add_argument("--shots", type=int, default=8_192)
    args = parser.parse_args()

    out_dir = Path("quantum_prime_gaps/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  4-Qubit Depth Reduction — Bell+RY+iQFT Optimization")
    print(f"  Gaps: {GAPS}   MAX_GAP={MAX_GAP}   Shots: {SHOTS:,}")
    print(f"  Proxy backend: {BACKEND_PROXY.name}")
    print("=" * 70)

    # Try to get Kingston's real coupling map for transpilation
    coupling_map = None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        real_backend = service.backend("ibm_kingston")
        coupling_map = real_backend.coupling_map
        print(f"\n  Using REAL Kingston coupling map ({len(list(coupling_map))} edges)")
    except Exception as exc:
        print(f"\n  Kingston coupling map unavailable ({exc}); using FakeSherbrooke")

    print()
    results = run_comparison(coupling_map)

    winner = pick_winner(results, SIM_BASELINE_ROOT_MI)
    print_table(results, winner)

    print(f"\n  WINNER: {winner.label}")
    print(f"  depth={winner.depth}, ECR={winner.ecr}, "
          f"root MI={winner.root_mi:.4f} bits, "
          f"predicted retention={winner.retention:.1%}")

    hw_mi = hw_ret = hw_info = None

    if args.hardware:
        print("\n── Hardware submission (ibm_kingston) ─────────────────────")
        # Rebuild winner circuit (clean, no transpile artifacts)
        if "degree=0" in winner.label or "baseline" in winner.label.lower():
            hw_qc = build_full_iqft(0)
        elif "degree=1" in winner.label:
            hw_qc = build_full_iqft(1)
        elif "degree=2" in winner.label:
            hw_qc = build_full_iqft(2)
        elif "degree=3" in winner.label:
            hw_qc = build_full_iqft(3)
        else:
            hw_qc = build_line_iqft()

        hw_info = run_hardware(hw_qc, shots=args.shots)
        hw_mi, hw_ret = hw_report(hw_info, winner)
    else:
        print("\n  [Hardware skipped — run with --hardware to submit winner to ibm_kingston]")

    save_json(results, winner, hw_info, hw_mi, hw_ret, out_dir)
    print()


if __name__ == "__main__":
    main()
