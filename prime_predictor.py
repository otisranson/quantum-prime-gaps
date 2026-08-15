"""prime_predictor.py

Recurrent quantum circuit for prime gap sequence learning and next-prime prediction.

Architecture:
  - Fixed 4-qubit circuit: Bell pair + RY gap encoding + approximated iQFT (degree=1)
  - Time is the extra dimension: slides a 4-gap window across all 49 known gaps (46 windows)
  - Feedback: each window's measurement mode (most-likely bitstring) adds a small
    angle offset to the next window's RY encoding — classical post-processing that
    imprints sequence history onto the quantum state
  - Final window output distribution → decode → predicted next gap

Ground truth:
  - Last known prime:  #50 = 229
  - Actual next prime: #51 = 233
  - Actual gap:        4

Run: python prime_predictor.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis.qft import synth_qft_full
from qiskit_aer import AerSimulator

# ── Prime gap data ─────────────────────────────────────────────────────────────

FIRST_50_PRIMES: list[int] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
]
ALL_GAPS: list[int] = [
    FIRST_50_PRIMES[i + 1] - FIRST_50_PRIMES[i]
    for i in range(len(FIRST_50_PRIMES) - 1)
]  # 49 gaps
MAX_GAP = max(ALL_GAPS)  # 14

LAST_KNOWN_PRIME = FIRST_50_PRIMES[-1]   # 229
TRUE_NEXT_PRIME  = 233
TRUE_GAP         = TRUE_NEXT_PRIME - LAST_KNOWN_PRIME  # 4

WINDOW_SIZE    = 4
SHOTS          = 8_192
FEEDBACK_SCALE = math.pi / 16   # max modulation ±π/32 ≈ ±0.098 rad

# ── Sliding windows ────────────────────────────────────────────────────────────

WINDOWS: list[list[int]] = [
    ALL_GAPS[i : i + WINDOW_SIZE]
    for i in range(len(ALL_GAPS) - WINDOW_SIZE + 1)
]  # 46 windows

# ── Circuit builder ────────────────────────────────────────────────────────────

def build_circuit(gaps: list[int], feedback_offset: float = 0.0) -> QuantumCircuit:
    """Bell pair + RY gap encoding (+ feedback offset) + approx iQFT (degree=1)."""
    qc = QuantumCircuit(WINDOW_SIZE, WINDOW_SIZE)
    qc.h(0)
    qc.cx(0, 1)
    for i, gap in enumerate(gaps):
        angle = gap * math.pi / MAX_GAP + feedback_offset
        qc.ry(angle, i)
    iqft = synth_qft_full(WINDOW_SIZE, inverse=True, do_swaps=True, approximation_degree=1)
    qc.compose(iqft, inplace=True)
    qc.measure(range(WINDOW_SIZE), range(WINDOW_SIZE))
    return qc

# ── Feedback ───────────────────────────────────────────────────────────────────

def bitstring_to_offset(bs: str) -> float:
    """Map mode bitstring integer → small angle offset centered at 0."""
    val = int(bs, 2)
    max_val = (1 << len(bs)) - 1  # 15 for 4-bit
    return (val / max_val - 0.5) * FEEDBACK_SCALE

# ── MI utilities ───────────────────────────────────────────────────────────────

def counts_to_bits(counts: dict, n: int) -> np.ndarray:
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


def _mm(k: int, n: int) -> float:
    return (k - 1) / (2 * n * math.log(2))


def mi_halves(bits: np.ndarray, left: list[int], right: list[int]) -> float:
    n = len(bits)
    lv = np.zeros(n, dtype=np.int64)
    for i, q in enumerate(left):
        lv += bits[:, q].astype(np.int64) << i
    rv = np.zeros(n, dtype=np.int64)
    for i, q in enumerate(right):
        rv += bits[:, q].astype(np.int64) << i
    ls, rs = 1 << len(left), 1 << len(right)
    pl = np.bincount(lv, minlength=ls) / n
    pr = np.bincount(rv, minlength=rs) / n
    pj = np.bincount(lv * rs + rv, minlength=ls * rs) / n
    hl = _entropy(pl) + _mm(int(np.sum(pl > 0)), n)
    hr = _entropy(pr) + _mm(int(np.sum(pr > 0)), n)
    hj = _entropy(pj) + _mm(int(np.sum(pj > 0)), n)
    return max(0.0, hl + hr - hj)

# ── Prediction decoder ─────────────────────────────────────────────────────────

def decode_prediction(counts: dict) -> tuple[float, int, str, int]:
    """Return (weighted_gap_float, weighted_gap_rounded, mode_bitstring, mode_gap_rounded)."""
    total = sum(counts.values())
    weighted_gap = 0.0
    for bs, cnt in counts.items():
        val = int(bs, 2)
        angle = val * math.pi / 15       # 0–15 → 0–π
        gap = angle * MAX_GAP / math.pi  # angle → gap value
        weighted_gap += (cnt / total) * gap

    mode_bs = max(counts, key=counts.get)
    mode_val = int(mode_bs, 2)
    mode_gap = mode_val * math.pi / 15 * MAX_GAP / math.pi

    return weighted_gap, max(1, round(weighted_gap)), mode_bs, max(1, round(mode_gap))

# ── Simulation ─────────────────────────────────────────────────────────────────

SIM = AerSimulator()


def run_window(qc: QuantumCircuit) -> dict:
    tqc = transpile(qc, SIM)
    return SIM.run(tqc, shots=SHOTS).result().get_counts()

# ── Recurrent loop ─────────────────────────────────────────────────────────────

def run_recurrent() -> list[dict]:
    records = []
    feedback_offset = 0.0

    for w_idx, window in enumerate(WINDOWS):
        qc = build_circuit(window, feedback_offset)
        counts = run_window(qc)
        bits = counts_to_bits(counts, WINDOW_SIZE)
        mi = mi_halves(bits, [0, 1], [2, 3])
        mode_bs = max(counts, key=counts.get)
        next_offset = bitstring_to_offset(mode_bs)

        records.append({
            "window_idx": w_idx,
            "gaps": window,
            "feedback_in": round(feedback_offset, 6),
            "mi": round(mi, 6),
            "mode_bs": mode_bs,
            "mode_prob": round(counts[mode_bs] / SHOTS, 4),
            "counts": {k: v for k, v in sorted(counts.items(), key=lambda x: -x[1])},
        })

        feedback_offset = next_offset
        if w_idx % 10 == 0 or w_idx == len(WINDOWS) - 1:
            print(f"  Window {w_idx:2d}/{len(WINDOWS)-1}  gaps={window}  "
                  f"MI={mi:.4f}  mode={mode_bs}  → next_offset={next_offset:+.4f}")

    return records

# ── Plotting ───────────────────────────────────────────────────────────────────

BG   = "#0d1117"
GRID = "#1e293b"
MUT  = "#94a3b8"
FG   = "#f8fafc"
BLUE = "#7dd3fc"
ORG  = "#fb923c"
GRN  = "#22c55e"
RED  = "#ef4444"


def _dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT)
    ax.spines[:].set_color(GRID)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)


def make_plots(records: list[dict], weighted_gap: float, rounded_gap: int,
               mode_bs: str, mode_gap: int, out_dir: Path) -> None:

    mi_vals  = np.array([r["mi"] for r in records])
    max_gaps = np.array([max(r["gaps"]) for r in records])
    x = np.arange(len(records))

    # ── Plot 1: MI over windows ────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    fig1.patch.set_facecolor(BG)
    _dark_ax(ax1)

    cmap = plt.cm.get_cmap("plasma")
    for i in range(len(records)):
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.20,
                    color=cmap(max_gaps[i] / MAX_GAP), linewidth=0)

    ax1.plot(x, mi_vals, color=BLUE, linewidth=1.5, zorder=3)
    ax1.fill_between(x, mi_vals, alpha=0.15, color=BLUE, zorder=2)
    ax1.axvline(len(records) - 1, color=ORG, linewidth=1.2,
                linestyle="--", label="Prediction window")
    ax1.set_xlabel("Window index")
    ax1.set_ylabel("Root MI  (bits)")
    ax1.set_title("Mutual Information across recurrent windows", color=FG, fontsize=11)
    ax1.legend(framealpha=0, labelcolor=ORG, fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, MAX_GAP))
    sm.set_array([])
    cb = fig1.colorbar(sm, ax=ax1, pad=0.01)
    cb.set_label("Max gap in window", color=MUT)
    cb.ax.yaxis.set_tick_params(color=MUT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUT)

    fig1.tight_layout()
    p1 = out_dir / "predictor_mi.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved → {p1}")

    # ── Plot 2: Final window distribution ─────────────────────────────────────
    final_counts = records[-1]["counts"]
    states = [f"{i:04b}" for i in range(16)]
    probs  = [final_counts.get(s, 0) / SHOTS for s in states]
    gap_labels = [f"{i / 15 * MAX_GAP:.1f}" for i in range(16)]

    true_bin = round(TRUE_GAP / MAX_GAP * 15)
    pred_bin = round(weighted_gap / MAX_GAP * 15)

    bar_colors = [
        GRN if i == true_bin else ORG if i == pred_bin else "#334155"
        for i in range(16)
    ]

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.patch.set_facecolor(BG)
    _dark_ax(ax2)
    bars = ax2.bar(range(16), probs, color=bar_colors,
                   edgecolor=GRID, linewidth=0.5, width=0.8)
    ax2.set_xticks(range(16))
    ax2.set_xticklabels(
        [f"|{s}⟩\n(≈{g})" for s, g in zip(states, gap_labels, strict=True)],
        fontsize=7, color=MUT,
    )
    ax2.set_ylabel("Probability")
    ax2.set_title(
        f"Final window prediction distribution  —  window gaps: {WINDOWS[-1]}\n"
        f"Weighted E[gap] = {weighted_gap:.2f} → {rounded_gap}  |  "
        f"True gap = {TRUE_GAP}  (prime {LAST_KNOWN_PRIME} → {TRUE_NEXT_PRIME})",
        color=FG, fontsize=10,
    )
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    legend_els = [
        Patch(facecolor=GRN, label=f"Ground truth  (gap = {TRUE_GAP})"),
        Patch(facecolor=ORG, label=f"Weighted prediction  ({weighted_gap:.2f} → {rounded_gap})"),
        Patch(facecolor="#334155", label="Other states"),
    ]
    ax2.legend(handles=legend_els, framealpha=0, labelcolor="white", fontsize=9)
    for bar, prob in zip(bars, probs, strict=True):
        if prob > 0.04:
            ax2.text(bar.get_x() + bar.get_width() / 2, prob + 0.003,
                     f"{prob:.1%}", ha="center", va="bottom", fontsize=7, color=FG)
    fig2.tight_layout()
    p2 = out_dir / "predictor_final_dist.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved → {p2}")

    # ── Plot 3: Summary scorecard ──────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    fig3.patch.set_facecolor(BG)
    ax3.set_facecolor(BG)
    ax3.axis("off")

    correct = (rounded_gap == TRUE_GAP)
    verdict_txt   = "✓  CORRECT" if correct else "✗  MISS"
    verdict_color = GRN if correct else RED

    rows = [
        ("Last known prime",        f"#{len(FIRST_50_PRIMES)} = {LAST_KNOWN_PRIME}",          MUT),
        ("Windows processed",       f"{len(records)} (all 49 gaps, window=4)",                MUT),
        ("Final window",            str(WINDOWS[-1]),                                          MUT),
        None,
        ("Weighted prediction",     f"E[gap] = {weighted_gap:.3f}  →  gap = {rounded_gap}", ORG),
        ("Mode prediction",         f"|{mode_bs}⟩  →  gap ≈ {mode_gap}",                    BLUE),
        ("Predicted next prime",    f"{LAST_KNOWN_PRIME} + {rounded_gap} = "
                                    f"{LAST_KNOWN_PRIME + rounded_gap}",                       ORG),
        None,
        ("Ground truth gap",        str(TRUE_GAP),                                             GRN),
        ("Ground truth prime",      str(TRUE_NEXT_PRIME),                                      GRN),
        ("Verdict",                 verdict_txt,                                               verdict_color),
    ]

    y = 0.97
    for row in rows:
        if row is None:
            y -= 0.04
            continue
        label, value, color = row
        ax3.text(0.02, y, f"{label}:", transform=ax3.transAxes,
                 fontsize=10, color="#64748b", va="top")
        ax3.text(0.48, y, value, transform=ax3.transAxes,
                 fontsize=10, color=color, va="top", fontweight="bold")
        y -= 0.088

    ax3.set_title("Quantum Prime Predictor — Result", color=FG, fontsize=13, pad=10)
    fig3.tight_layout()
    p3 = out_dir / "predictor_summary.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved → {p3}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir  = Path("quantum_prime_gaps/screenshots")
    json_dir = Path("output/prime")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Quantum Prime Predictor — Recurrent Gap Learning")
    print(f"  Primes: {len(FIRST_50_PRIMES)}  |  Gaps: {len(ALL_GAPS)}  |  "
          f"Windows: {len(WINDOWS)}  |  MAX_GAP: {MAX_GAP}")
    print(f"  Last prime: {LAST_KNOWN_PRIME}  |  True next: {TRUE_NEXT_PRIME}  "
          f"(gap = {TRUE_GAP})")
    print(f"  Feedback scale: ±{FEEDBACK_SCALE/2:.4f} rad  |  "
          f"Shots/window: {SHOTS:,}")
    print("=" * 65)
    print()

    print("Running recurrent loop (46 windows × 8,192 shots)...")
    records = run_recurrent()

    final_counts = records[-1]["counts"]
    weighted_gap, rounded_gap, mode_bs, mode_gap = decode_prediction(final_counts)
    predicted_prime = LAST_KNOWN_PRIME + rounded_gap
    correct = (rounded_gap == TRUE_GAP)

    print()
    print("─" * 65)
    print(f"  Final window gaps:    {WINDOWS[-1]}")
    print(f"  Weighted prediction:  E[gap] = {weighted_gap:.4f}  →  {rounded_gap}  "
          f"→  prime {predicted_prime}")
    print(f"  Mode prediction:      |{mode_bs}⟩  →  gap ≈ {mode_gap}")
    print(f"  Ground truth:         gap = {TRUE_GAP}  →  prime {TRUE_NEXT_PRIME}")
    print(f"  Error:                |{weighted_gap:.4f} − {TRUE_GAP}| = "
          f"{abs(weighted_gap - TRUE_GAP):.4f}")
    print(f"  Verdict:              {'✓ CORRECT' if correct else '✗ MISS'}")
    print("─" * 65)

    data = {
        "config": {
            "n_qubits": WINDOW_SIZE,
            "shots_per_window": SHOTS,
            "feedback_scale_rad": FEEDBACK_SCALE,
            "iqft_approx_degree": 1,
            "n_windows": len(WINDOWS),
            "max_gap": MAX_GAP,
        },
        "ground_truth": {
            "last_known_prime": LAST_KNOWN_PRIME,
            "true_next_prime": TRUE_NEXT_PRIME,
            "true_gap": TRUE_GAP,
        },
        "prediction": {
            "final_window_gaps": WINDOWS[-1],
            "weighted_gap_float": round(weighted_gap, 6),
            "weighted_gap_rounded": rounded_gap,
            "mode_bitstring": mode_bs,
            "mode_gap_rounded": mode_gap,
            "predicted_prime": predicted_prime,
            "correct": correct,
            "error_abs": round(abs(weighted_gap - TRUE_GAP), 4),
        },
        "per_window": [
            {k: v for k, v in r.items() if k != "counts"}
            for r in records
        ],
        "final_window_counts": records[-1]["counts"],
    }
    json_path = json_dir / "prime_predictor_results.json"
    json_path.write_text(json.dumps(data, indent=2))
    print(f"\n  JSON → {json_path}")

    print("\nGenerating plots...")
    make_plots(records, weighted_gap, rounded_gap, mode_bs, mode_gap, out_dir)

    mean_mi = sum(r["mi"] for r in records) / len(records)
    print(f"\n  Mean MI across all windows: {mean_mi:.4f} bits")
    print(f"  Final window MI:            {records[-1]['mi']:.4f} bits")
    print()


if __name__ == "__main__":
    main()
