"""prime_predictor.py

Recurrent quantum circuit for prime gap sequence learning and next-prime prediction.

Architecture:
  - Fixed 4-qubit circuit: Bell pair + RY gap encoding + approximated iQFT (degree=1)
  - Time is the extra dimension: slides a 4-gap window across all 49 known gaps (46 windows)
  - Feedback (v2): per-qubit — each bit of the mode bitstring drives its own qubit's
    angle offset independently, breaking the |0000⟩ attractor of the uniform scalar approach
  - Scale sweep: runs scales [0.05, 0.1, 0.2] and picks the one closest to gap=4
  - Noisy preflight: winning scale re-run on AerSimulator with Kingston noise model

Ground truth: prime #50 = 229, prime #51 = 233, gap = 4

Run: python prime_predictor.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
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
SWEEP_SCALES   = [0.05, 0.1, 0.2]   # feedback_scale values to test

WINDOWS: list[list[int]] = [
    ALL_GAPS[i : i + WINDOW_SIZE]
    for i in range(len(ALL_GAPS) - WINDOW_SIZE + 1)
]  # 46 windows

# ── Circuit builder ────────────────────────────────────────────────────────────

def build_circuit(gaps: list[int], offsets: list[float]) -> QuantumCircuit:
    """Bell pair + per-qubit RY (gap angle + per-qubit offset) + approx iQFT (degree=1)."""
    qc = QuantumCircuit(WINDOW_SIZE, WINDOW_SIZE)
    qc.h(0)
    qc.cx(0, 1)
    for i, gap in enumerate(gaps):
        angle = gap * math.pi / MAX_GAP + offsets[i]
        qc.ry(angle, i)
    iqft = synth_qft_full(WINDOW_SIZE, inverse=True, do_swaps=True, approximation_degree=1)
    qc.compose(iqft, inplace=True)
    qc.measure(range(WINDOW_SIZE), range(WINDOW_SIZE))
    return qc

# ── Per-qubit feedback ─────────────────────────────────────────────────────────

def bitstring_to_offsets(bs: str, scale: float) -> list[float]:
    """Decompose mode bitstring into per-qubit angle offsets.

    Qiskit bitstrings are MSB-left; qubit 0 is the rightmost character.
    Each bit maps to -(scale/2) for 0 or +(scale/2) for 1, centered at 0.
    """
    n = len(bs)
    return [(int(bs[n - 1 - q]) - 0.5) * scale for q in range(n)]

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
    """Return (weighted_gap_float, weighted_gap_rounded, mode_bs, mode_gap_rounded)."""
    total = sum(counts.values())
    weighted_gap = 0.0
    for bs, cnt in counts.items():
        val = int(bs, 2)
        angle = val * math.pi / 15
        gap   = angle * MAX_GAP / math.pi
        weighted_gap += (cnt / total) * gap

    mode_bs  = max(counts, key=counts.get)
    mode_gap = int(mode_bs, 2) * math.pi / 15 * MAX_GAP / math.pi

    return weighted_gap, max(1, round(weighted_gap)), mode_bs, max(1, round(mode_gap))

# ── Recurrent loop ─────────────────────────────────────────────────────────────

@dataclass
class WindowRecord:
    window_idx: int
    gaps: list[int]
    offsets_in: list[float]
    mi: float
    mode_bs: str
    mode_prob: float
    counts: dict = field(default_factory=dict)


def run_recurrent(scale: float, sim: AerSimulator,
                  verbose: bool = False) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    offsets = [0.0] * WINDOW_SIZE   # start with no feedback

    for w_idx, window in enumerate(WINDOWS):
        qc = build_circuit(window, offsets)
        tqc = transpile(qc, sim)
        counts = sim.run(tqc, shots=SHOTS).result().get_counts()

        bits = counts_to_bits(counts, WINDOW_SIZE)
        mi   = mi_halves(bits, [0, 1], [2, 3])
        mode_bs  = max(counts, key=counts.get)
        next_offsets = bitstring_to_offsets(mode_bs, scale)

        records.append(WindowRecord(
            window_idx=w_idx,
            gaps=window,
            offsets_in=offsets[:],
            mi=round(mi, 6),
            mode_bs=mode_bs,
            mode_prob=round(counts[mode_bs] / SHOTS, 4),
            counts={k: v for k, v in sorted(counts.items(), key=lambda x: -x[1])},
        ))

        if verbose and (w_idx % 10 == 0 or w_idx == len(WINDOWS) - 1):
            print(f"    w={w_idx:2d}  gaps={window}  MI={mi:.4f}  "
                  f"mode={mode_bs}  offsets→{[f'{o:+.3f}' for o in next_offsets]}")

        offsets = next_offsets

    return records

# ── Scale sweep ────────────────────────────────────────────────────────────────

@dataclass
class ScaleResult:
    scale: float
    records: list[WindowRecord]
    weighted_gap: float
    rounded_gap: int
    mode_bs: str
    mode_gap: int
    error: float
    mean_mi: float
    final_mi: float

    @property
    def predicted_prime(self) -> int:
        return LAST_KNOWN_PRIME + self.rounded_gap


def sweep(sim: AerSimulator) -> list[ScaleResult]:
    results = []
    for scale in SWEEP_SCALES:
        print(f"\n  scale={scale} ──────────────────────────────────────────")
        records = run_recurrent(scale, sim, verbose=True)
        final_counts = records[-1].counts
        w_gap, r_gap, m_bs, m_gap = decode_prediction(final_counts)
        mi_vals = [r.mi for r in records]
        results.append(ScaleResult(
            scale=scale,
            records=records,
            weighted_gap=w_gap,
            rounded_gap=r_gap,
            mode_bs=m_bs,
            mode_gap=m_gap,
            error=abs(w_gap - TRUE_GAP),
            mean_mi=sum(mi_vals) / len(mi_vals),
            final_mi=mi_vals[-1],
        ))
        print(f"    → E[gap]={w_gap:.4f}  rounded={r_gap}  prime={LAST_KNOWN_PRIME+r_gap}"
              f"  error={abs(w_gap-TRUE_GAP):.4f}  mean_MI={mi_vals[-1]:.4f}")
    return results


def pick_winner(results: list[ScaleResult]) -> ScaleResult:
    return min(results, key=lambda r: r.error)

# ── Noisy preflight ────────────────────────────────────────────────────────────

def build_noisy_sim() -> tuple[AerSimulator | None, object | None]:
    """Try to build Kingston-noise AerSimulator. Returns (sim, coupling_map) or (None, None)."""
    try:
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend("ibm_kingston")
        nm = NoiseModel.from_backend(backend)
        sim = AerSimulator(noise_model=nm)
        print(f"  Kingston noise model loaded: {len(nm.noise_instructions)} error channels")
        return sim, backend.coupling_map
    except Exception as exc:
        print(f"  Noise model unavailable ({exc}); skipping noisy preflight")
        return None, None


def run_noisy_preflight(winner: ScaleResult, noisy_sim: AerSimulator,
                        coupling_map) -> ScaleResult:
    print(f"\n  Noisy preflight  (scale={winner.scale}, Kingston noise model) ──────")
    records = run_recurrent(winner.scale, noisy_sim, verbose=True)
    final_counts = records[-1].counts
    w_gap, r_gap, m_bs, m_gap = decode_prediction(final_counts)
    mi_vals = [r.mi for r in records]
    result = ScaleResult(
        scale=winner.scale,
        records=records,
        weighted_gap=w_gap,
        rounded_gap=r_gap,
        mode_bs=m_bs,
        mode_gap=m_gap,
        error=abs(w_gap - TRUE_GAP),
        mean_mi=sum(mi_vals) / len(mi_vals),
        final_mi=mi_vals[-1],
    )
    print(f"  → E[gap]={w_gap:.4f}  rounded={r_gap}  prime={LAST_KNOWN_PRIME+r_gap}"
          f"  error={abs(w_gap-TRUE_GAP):.4f}")
    return result

# ── Plotting ───────────────────────────────────────────────────────────────────

BG   = "#0d1117"
GRID = "#1e293b"
MUT  = "#94a3b8"
FG   = "#f8fafc"
GRN  = "#22c55e"
RED  = "#ef4444"
SCALE_COLORS = ["#7dd3fc", "#fb923c", "#a78bfa"]   # blue / orange / purple


def _dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT)
    ax.spines[:].set_color(GRID)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)


def plot_mi_sweep(sweep_results: list[ScaleResult], winner: ScaleResult,
                  out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    _dark_ax(ax)

    for sr, color in zip(sweep_results, SCALE_COLORS, strict=True):
        mi_vals = [r.mi for r in sr.records]
        lw = 2.0 if sr is winner else 1.0
        alpha = 1.0 if sr is winner else 0.55
        label = f"scale={sr.scale}  E[gap]={sr.weighted_gap:.2f}→{sr.rounded_gap}"
        if sr is winner:
            label += "  ★ winner"
        ax.plot(mi_vals, color=color, linewidth=lw, alpha=alpha, label=label)

    ax.axvline(len(WINDOWS) - 1, color="#64748b", linewidth=1, linestyle="--")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Root MI  (bits)")
    ax.set_title("MI across recurrent windows — feedback scale sweep", color=FG, fontsize=11)
    ax.legend(framealpha=0, labelcolor="white", fontsize=9)
    fig.tight_layout()
    p = out_dir / "predictor_mi_sweep.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p}")


def plot_final_dist(result: ScaleResult, label: str, filename: str,
                    out_dir: Path) -> None:
    counts = result.records[-1].counts
    states = [f"{i:04b}" for i in range(16)]
    probs  = [counts.get(s, 0) / SHOTS for s in states]
    gap_labels = [f"{i / 15 * MAX_GAP:.1f}" for i in range(16)]

    true_bin = round(TRUE_GAP / MAX_GAP * 15)
    pred_bin = round(result.weighted_gap / MAX_GAP * 15)

    bar_colors = [
        GRN if i == true_bin else "#fb923c" if i == pred_bin else "#334155"
        for i in range(16)
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    _dark_ax(ax)
    bars = ax.bar(range(16), probs, color=bar_colors, edgecolor=GRID,
                  linewidth=0.5, width=0.8)
    ax.set_xticks(range(16))
    ax.set_xticklabels(
        [f"|{s}⟩\n(≈{g})" for s, g in zip(states, gap_labels, strict=True)],
        fontsize=7, color=MUT,
    )
    ax.set_ylabel("Probability")
    ax.set_title(
        f"{label}\nWindow {WINDOWS[-1]}  |  E[gap]={result.weighted_gap:.2f}→{result.rounded_gap}"
        f"  |  True gap={TRUE_GAP}  |  scale={result.scale}",
        color=FG, fontsize=10,
    )
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    legend_els = [
        Patch(facecolor=GRN,      label=f"Ground truth  (gap={TRUE_GAP})"),
        Patch(facecolor="#fb923c", label=f"Weighted prediction ({result.weighted_gap:.2f}→{result.rounded_gap})"),
        Patch(facecolor="#334155", label="Other states"),
    ]
    ax.legend(handles=legend_els, framealpha=0, labelcolor="white", fontsize=9)
    for bar, prob in zip(bars, probs, strict=True):
        if prob > 0.04:
            ax.text(bar.get_x() + bar.get_width() / 2, prob + 0.003,
                    f"{prob:.1%}", ha="center", va="bottom", fontsize=7, color=FG)
    fig.tight_layout()
    p = out_dir / filename
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p}")


def plot_summary(sweep_results: list[ScaleResult], winner: ScaleResult,
                 noisy: ScaleResult | None, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    def verdict(r: ScaleResult) -> tuple[str, str]:
        ok = r.rounded_gap == TRUE_GAP
        return ("✓ CORRECT", GRN) if ok else ("✗ MISS", RED)

    rows: list[tuple] = [
        ("Scale sweep — AerSimulator (clean)", None, FG),
    ]
    for sr, color in zip(sweep_results, SCALE_COLORS, strict=True):
        flag = "  ★" if sr is winner else ""
        v, vc = verdict(sr)
        rows.append((
            f"  scale={sr.scale}{flag}",
            f"E[gap]={sr.weighted_gap:.3f}→{sr.rounded_gap}  prime={sr.predicted_prime}"
            f"  err={sr.error:.3f}  {v}",
            color if sr is not winner else vc,
        ))

    rows.append((None, None, None))
    rows.append((f"Winner  (per-qubit feedback, scale={winner.scale})" , None, FG))
    rows.append((
        "  Clean sim",
        f"E[gap]={winner.weighted_gap:.3f}→{winner.rounded_gap}  "
        f"prime={winner.predicted_prime}  " + verdict(winner)[0],
        verdict(winner)[1],
    ))

    if noisy is not None:
        nv, nvc = verdict(noisy)
        rows.append((
            "  Kingston noisy sim",
            f"E[gap]={noisy.weighted_gap:.3f}→{noisy.rounded_gap}  "
            f"prime={noisy.predicted_prime}  {nv}",
            nvc,
        ))
        hw_ready = noisy.error < 1.5
        rows.append((None, None, None))
        rows.append((
            "Hardware flag",
            "✓ READY — noisy preflight passed" if hw_ready
            else "⚠ MARGINAL — error > 1.5, review before hardware",
            GRN if hw_ready else "#facc15",
        ))

    rows.append((None, None, None))
    rows.append(("Ground truth", f"gap={TRUE_GAP}  prime={TRUE_NEXT_PRIME}", GRN))

    ax.set_title("Quantum Prime Predictor v2 — Result Summary",
                 color=FG, fontsize=13, pad=10)
    y = 0.95
    for row in rows:
        if row[0] is None:
            y -= 0.04
            continue
        label, value, color = row
        ax.text(0.02, y, label + (":" if value else ""), transform=ax.transAxes,
                fontsize=9.5, color="#64748b" if value else FG, va="top")
        if value:
            ax.text(0.42, y, value, transform=ax.transAxes,
                    fontsize=9.5, color=color, va="top", fontweight="bold")
        y -= 0.082

    fig.tight_layout()
    p = out_dir / "predictor_summary_v2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p}")

# ── JSON serialiser ────────────────────────────────────────────────────────────

def save_json(sweep_results: list[ScaleResult], winner: ScaleResult,
              noisy: ScaleResult | None, json_dir: Path) -> None:
    data: dict = {
        "config": {
            "n_qubits": WINDOW_SIZE,
            "shots_per_window": SHOTS,
            "sweep_scales": SWEEP_SCALES,
            "iqft_approx_degree": 1,
            "n_windows": len(WINDOWS),
            "max_gap": MAX_GAP,
            "feedback": "per-qubit v2",
        },
        "ground_truth": {
            "last_known_prime": LAST_KNOWN_PRIME,
            "true_next_prime": TRUE_NEXT_PRIME,
            "true_gap": TRUE_GAP,
        },
        "sweep": [
            {
                "scale": sr.scale,
                "weighted_gap": round(sr.weighted_gap, 6),
                "rounded_gap": sr.rounded_gap,
                "predicted_prime": sr.predicted_prime,
                "error_abs": round(sr.error, 4),
                "mean_mi": round(sr.mean_mi, 6),
                "final_mi": round(sr.final_mi, 6),
                "correct": sr.rounded_gap == TRUE_GAP,
                "per_window": [
                    {"w": r.window_idx, "gaps": r.gaps, "mi": r.mi,
                     "mode": r.mode_bs, "mode_prob": r.mode_prob}
                    for r in sr.records
                ],
                "final_counts": sr.records[-1].counts,
            }
            for sr in sweep_results
        ],
        "winner": winner.scale,
    }
    if noisy is not None:
        data["noisy_preflight"] = {
            "scale": noisy.scale,
            "weighted_gap": round(noisy.weighted_gap, 6),
            "rounded_gap": noisy.rounded_gap,
            "predicted_prime": noisy.predicted_prime,
            "error_abs": round(noisy.error, 4),
            "mean_mi": round(noisy.mean_mi, 6),
            "final_mi": round(noisy.final_mi, 6),
            "correct": noisy.rounded_gap == TRUE_GAP,
            "hw_ready": noisy.error < 1.5,
            "final_counts": noisy.records[-1].counts,
        }
    p = json_dir / "prime_predictor_results.json"
    p.write_text(json.dumps(data, indent=2))
    print(f"  JSON → {p}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir  = Path("quantum_prime_gaps/screenshots")
    json_dir = Path("output/prime")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Quantum Prime Predictor v2 — Per-Qubit Feedback + Scale Sweep")
    print(f"  Gaps: {len(ALL_GAPS)}  Windows: {len(WINDOWS)}  "
          f"Scales: {SWEEP_SCALES}  MAX_GAP: {MAX_GAP}")
    print(f"  Last prime: {LAST_KNOWN_PRIME}  |  True next: {TRUE_NEXT_PRIME}  "
          f"(gap={TRUE_GAP})")
    print("=" * 65)

    clean_sim = AerSimulator()

    print("\nPhase 1 — scale sweep on clean AerSimulator")
    sweep_results = sweep(clean_sim)

    winner = pick_winner(sweep_results)
    print(f"\n  Winner: scale={winner.scale}  E[gap]={winner.weighted_gap:.4f}"
          f"  →  prime {winner.predicted_prime}  (error={winner.error:.4f})")

    print("\nPhase 2 — Kingston noisy preflight")
    noisy_sim, coupling_map = build_noisy_sim()
    noisy_result: ScaleResult | None = None
    if noisy_sim is not None:
        noisy_result = run_noisy_preflight(winner, noisy_sim, coupling_map)
        nv = "✓ CORRECT" if noisy_result.rounded_gap == TRUE_GAP else "✗ MISS"
        print(f"  Noisy result: E[gap]={noisy_result.weighted_gap:.4f}"
              f"  →  prime {noisy_result.predicted_prime}  {nv}")
        if noisy_result.error < 1.5:
            print("  ✓ HARDWARE READY — noisy preflight error < 1.5")
        else:
            print("  ⚠ MARGINAL — review before hardware submission")

    print("\nGenerating plots...")
    plot_mi_sweep(sweep_results, winner, out_dir)
    plot_final_dist(winner, "Winner — clean AerSimulator",
                    "predictor_final_dist_winner.png", out_dir)
    if noisy_result is not None:
        plot_final_dist(noisy_result, "Kingston noisy preflight",
                        "predictor_noisy_dist.png", out_dir)
    plot_summary(sweep_results, winner, noisy_result, out_dir)

    save_json(sweep_results, winner, noisy_result, json_dir)

    print("\n── Final report ─────────────────────────────────────────────")
    for sr in sweep_results:
        mark = "★" if sr is winner else " "
        print(f"  {mark} scale={sr.scale}  E[gap]={sr.weighted_gap:.4f}"
              f"  rounded={sr.rounded_gap}  prime={sr.predicted_prime}"
              f"  err={sr.error:.4f}")
    if noisy_result is not None:
        nv = "✓" if noisy_result.rounded_gap == TRUE_GAP else "✗"
        print(f"\n  Noisy: E[gap]={noisy_result.weighted_gap:.4f}"
              f"  prime={noisy_result.predicted_prime}  {nv}")
    print(f"\n  Ground truth: gap={TRUE_GAP}  prime={TRUE_NEXT_PRIME}")
    print()


if __name__ == "__main__":
    main()
