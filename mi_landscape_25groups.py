"""mi_landscape_25groups.py

Layer 3 internal-wave extraction run (hypotheses/regime_internal_wave_structure.md):
25 independent, non-overlapping 4-qubit "v3" circuit runs across the first 100
prime gaps (group 0 = gaps 0-3, group 1 = gaps 4-7, ..., group 24 = gaps
96-99). Each group is its own Bell-pair + RY-gap-encoding + approximated-iQFT
circuit (prime_predictor.py's v3 architecture, degree=1, local-window
normalization) -- there is no per-qubit feedback between groups, since groups
are independent rather than recurrent.

Structured to mirror hardware execution exactly: one AerSimulator instance,
one circuit per group, run sequentially in a plain for loop. Swapping to
hardware later means dropping ibm_kingston in for AerSimulator at the single
`sim = AerSimulator()` line and wiring up --hardware below -- the loop and
circuit construction do not change. --hardware is accepted on the CLI but not
implemented yet (raises NotImplementedError); see CLAUDE.md.

After all 25 runs: Miller-Madow corrected mutual information between each
group's [q0,q1] half and [q2,q3] half, via qubit_hierarchy_core (the same
statistical core used by quantum_prime_gaps/qubit_hierarchy_analysis.py).

Run: python mi_landscape_25groups.py
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis.qft import synth_qft_full
from qiskit_aer import AerSimulator

from qubit_hierarchy_core import counts_to_bits_and_probs, mutual_information

REPO_ROOT = Path(__file__).parent

WINDOW_SIZE = 4
N_GROUPS = 25
N_GAPS = WINDOW_SIZE * N_GROUPS  # 100
SHOTS = 8_192
IQFT_APPROX_DEGREE = 1  # matches the proven v3 circuit (prime_predictor.py)

# Source run for the three known regime changepoints and the window count
# they were found in -- see hypotheses/second_order_gap_structure.md and
# hypotheses/log_summation_regime.md. n_windows is read from that run's own
# config, not re-derived here.
CHANGEPOINT_SOURCE = REPO_ROOT / "output/prime/20260816_010716/terrain_5000primes/results_5000primes.json"
KNOWN_CHANGEPOINTS = [1529, 2501, 4211]

OUT_ROOT = REPO_ROOT / "output" / "prime"


def sieve_primes(count: int) -> list[int]:
    """Sieve of Eratosthenes, doubling the search limit until enough primes
    are found. Independent second implementation exists in
    quantum_prime_gaps/quantum_prime_gaps.py; not imported from there since
    this script only needs the primes, not that module's other machinery."""
    limit = 10
    while True:
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        primes = [i for i, p in enumerate(is_p) if p]
        if len(primes) >= count:
            return primes[:count]
        limit *= 2


def _is_prime_trial_division(n: int) -> bool:
    if n < 2:
        return False
    return all(n % d != 0 for d in range(2, int(n**0.5) + 1))


def build_circuit(gaps: list[int]) -> tuple[QuantumCircuit, int]:
    """Bell pair + RY gap encoding (local-window normalization) + approximated
    iQFT (degree=1) -- the v3 architecture, no per-qubit feedback offset since
    groups are independent, not recurrent."""
    local_max = max(gaps)
    qc = QuantumCircuit(WINDOW_SIZE, WINDOW_SIZE)
    qc.h(0)
    qc.cx(0, 1)
    for i, gap in enumerate(gaps):
        qc.ry(gap * math.pi / local_max, i)
    iqft = synth_qft_full(WINDOW_SIZE, inverse=True, do_swaps=True, approximation_degree=IQFT_APPROX_DEGREE)
    qc.compose(iqft, inplace=True)
    qc.measure(range(WINDOW_SIZE), range(WINDOW_SIZE))
    return qc, local_max


def changepoint_group_positions(n_groups: int) -> list[tuple[int, float]]:
    with open(CHANGEPOINT_SOURCE) as f:
        n_windows = json.load(f)["config"]["n_windows"]
    return [(cp, cp / n_windows * n_groups) for cp in KNOWN_CHANGEPOINTS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", action="store_true",
                         help="(not implemented yet) run on ibm_kingston instead of AerSimulator")
    args = parser.parse_args()
    if args.hardware:
        raise NotImplementedError(
            "Hardware path not implemented yet -- run without --hardware. The loop below "
            "is already structured for a one-flag swap to ibm_kingston later."
        )

    primes = sieve_primes(N_GAPS + 1)
    assert all(_is_prime_trial_division(p) for p in primes), "sieve output failed independent trial-division check"
    all_gaps = [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]
    assert len(all_gaps) == N_GAPS

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  MI landscape -- 25 independent 4-qubit groups, gaps 0-{N_GAPS - 1}")
    print(f"  Run: {ts}")
    print("=" * 65)

    # One simulator instance, 25 sequential runs -- mirrors how a hardware
    # backend session would be used (one connection, many jobs).
    sim = AerSimulator()
    records = []
    for g in range(N_GROUPS):
        gaps = all_gaps[g * WINDOW_SIZE : (g + 1) * WINDOW_SIZE]
        qc, local_max = build_circuit(gaps)
        tqc = transpile(qc, sim)
        counts = sim.run(tqc, shots=SHOTS).result().get_counts()

        bits, probs = counts_to_bits_and_probs(counts, WINDOW_SIZE)
        mi = mutual_information(bits, probs, [0, 1], [2, 3], SHOTS)

        records.append({"group": g, "gaps": gaps, "local_max": local_max, "mi": round(mi, 6)})
        print(f"  group {g:2d}  gaps={gaps}  local_max={local_max}  MI={mi:.4f}")

    mi_values = np.array([r["mi"] for r in records])
    print(f"\n  MI landscape: mean={mi_values.mean():.4f}  std={mi_values.std():.4f}  "
          f"min={mi_values.min():.4f}  max={mi_values.max():.4f}")

    cp_positions = changepoint_group_positions(N_GROUPS)
    print("\n  Regime-changepoint proportional positions in 25-group space:")
    cp_report = []
    for cp, pos in cp_positions:
        nearest_group = max(0, min(N_GROUPS - 1, int(round(pos))))
        nearest_mi = float(mi_values[nearest_group])
        percentile = float(np.mean(mi_values <= nearest_mi) * 100)
        cp_report.append({
            "changepoint_window_5000run": cp,
            "group_position": round(pos, 3),
            "nearest_group": nearest_group,
            "nearest_group_mi": round(nearest_mi, 6),
            "percentile_in_25_group_mi": round(percentile, 1),
        })
        print(f"    window {cp:5d} -> group position {pos:6.3f}  nearest group {nearest_group:2d}  "
              f"MI={nearest_mi:.4f}  (percentile {percentile:.1f} of 25 groups)")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(N_GROUPS), mi_values, color="#4c72b0", marker="o", lw=1.5, markersize=4)
    colors = ["#d1495b", "#e08214", "#2a9d5c"]
    for (cp, pos), color in zip(cp_positions, colors, strict=True):
        ax.axvline(pos, color=color, lw=1.5, ls="--", label=f"changepoint {cp} -> group {pos:.2f}")
    ax.set_xlim(-0.5, N_GROUPS - 0.5)
    ax.set_xlabel("group index (4 gaps each, gaps 0-99)")
    ax.set_ylabel("MI [q0,q1] : [q2,q3]  (Miller-Madow corrected, bits)")
    ax.set_title(f"MI landscape -- 25 independent groups, first 100 prime gaps [{ts}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = out_dir / "mi_landscape.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved figure to {plot_path.relative_to(REPO_ROOT)}")

    results = {
        "timestamp": ts,
        "config": {
            "n_groups": N_GROUPS,
            "window_size": WINDOW_SIZE,
            "n_gaps": N_GAPS,
            "shots": SHOTS,
            "iqft_approx_degree": IQFT_APPROX_DEGREE,
            "backend": "AerSimulator (clean)",
            "changepoint_source": str(CHANGEPOINT_SOURCE.relative_to(REPO_ROOT)),
        },
        "per_group": records,
        "mi_stats": {
            "mean": round(float(mi_values.mean()), 6),
            "std": round(float(mi_values.std()), 6),
            "min": round(float(mi_values.min()), 6),
            "max": round(float(mi_values.max()), 6),
        },
        "changepoints": cp_report,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved results to {json_path.relative_to(REPO_ROOT)}")

    mean_mi = float(mi_values.mean())
    mi_range = float(mi_values.max() - mi_values.min())
    msg = (f"analysis: MI landscape 25 groups {ts} -- mean MI={mean_mi:.4f}, "
           f"range={mi_range:.4f}, August 17 2026")
    subprocess.run(["git", "add", str(out_dir.relative_to(REPO_ROOT))], check=True, cwd=REPO_ROOT)
    commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        print(f"\n  Committed: {out_dir.relative_to(REPO_ROOT)}")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"\n  Git commit skipped: {commit.stdout.strip()}")


if __name__ == "__main__":
    main()
