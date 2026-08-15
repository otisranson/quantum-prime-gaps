"""classical_baseline.py

Classical baseline comparison for the quantum prime gap predictor.

Runs six deterministic classical predictors against the same target as the
v3 hardware run: gap after prime 229, ground truth gap=4, prime=233.

Outputs → output/prime/{YYYYMMDD_HHMMSS}/:
  classical_baseline_comparison.md   markdown table + narrative
  classical_baseline_comparison.png  bar chart: predicted gap vs ground truth
  classical_baseline_results.json    machine-readable results

Auto-commits and pushes.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats

# ── Data ───────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent

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
]

LAST_PRIME  = FIRST_50_PRIMES[-1]   # 229
TRUE_GAP    = 4
TRUE_PRIME  = 233

# Hardware quantum result (from run 20260815_204703)
QC_HW_EGAP  = 3.9578
QC_HW_RGAP  = 4
QC_HW_PRIME = 233

# ── Baseline predictor dataclass ────────────────────────────────────────────────

@dataclass
class Baseline:
    name: str
    raw: float            # raw floating-point predicted gap
    rounded: int          # round(raw), clipped to ≥1
    predicted_prime: int
    error: float          # |raw - TRUE_GAP|
    note: str = ""


def make(name: str, raw: float, note: str = "") -> Baseline:
    rounded = max(1, round(raw))
    return Baseline(
        name=name,
        raw=raw,
        rounded=rounded,
        predicted_prime=LAST_PRIME + rounded,
        error=abs(raw - TRUE_GAP),
        note=note,
    )

# ── Six classical predictors ───────────────────────────────────────────────────

def predict_mean() -> Baseline:
    raw = float(np.mean(ALL_GAPS))
    return make("Mean gap", raw, f"mean of all {len(ALL_GAPS)} gaps")


def predict_last() -> Baseline:
    raw = float(ALL_GAPS[-1])
    return make("Last gap", raw, f"gap[{len(ALL_GAPS)-1}] = {ALL_GAPS[-1]}")


def predict_moving_average(window: int = 4) -> Baseline:
    raw = float(np.mean(ALL_GAPS[-window:]))
    return make(
        f"Moving avg (w={window})",
        raw,
        f"mean of last {window} gaps: {ALL_GAPS[-window:]}",
    )


def predict_median() -> Baseline:
    raw = float(np.median(ALL_GAPS))
    return make("Median gap", raw, f"median of all {len(ALL_GAPS)} gaps")


def predict_linear_regression() -> Baseline:
    x = np.arange(len(ALL_GAPS), dtype=float)
    y = np.array(ALL_GAPS, dtype=float)
    slope, intercept, r, p, _ = stats.linregress(x, y)
    pred_x = float(len(ALL_GAPS))  # index 49
    raw = slope * pred_x + intercept
    return make(
        "Linear regression",
        raw,
        f"slope={slope:.4f}, intercept={intercept:.4f}, R²={r**2:.4f}, p={p:.3f}",
    )


def predict_fft(top_k: int = 3) -> Baseline:
    """Keep top_k frequency components, evaluate one step past the known sequence."""
    y = np.array(ALL_GAPS, dtype=float)
    n = len(y)
    spectrum = np.fft.fft(y)
    magnitudes = np.abs(spectrum)
    # Zero out all but top_k frequencies (symmetric: keep conjugate pairs)
    top_indices = np.argsort(magnitudes)[::-1][:top_k]
    filtered = np.zeros_like(spectrum)
    for idx in top_indices:
        filtered[idx] = spectrum[idx]
        conj_idx = n - idx
        if 0 < conj_idx < n:   # skip DC (idx=0) and Nyquist boundary
            filtered[conj_idx] = spectrum[conj_idx]
    # Evaluate continuous DFT at t = n (one step past end)
    t = float(n)
    raw_complex = sum(
        filtered[k] * np.exp(2j * math.pi * k * t / n) for k in range(n)
    )
    raw = float(np.real(raw_complex)) / n
    return make(
        f"FFT (top-{top_k})",
        raw,
        f"DFT extrapolation at t={n}, keeping {top_k} dominant components",
    )

# ── Plotting ───────────────────────────────────────────────────────────────────

BG   = "#0d1117"
GRID = "#1e293b"
MUT  = "#94a3b8"
FG   = "#f8fafc"
GRN  = "#22c55e"
RED  = "#ef4444"
ORG  = "#fb923c"
BLU  = "#7dd3fc"
PRP  = "#a78bfa"


def _dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT)
    ax.spines[:].set_color(GRID)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    ax.title.set_color(FG)


def plot_comparison(baselines: list[Baseline], ts: str, out_dir: Path) -> Path:
    methods = ["Quantum HW\n(ibm_kingston)"] + [b.name.replace(" (", "\n(") for b in baselines]
    raw_gaps = [QC_HW_EGAP] + [b.raw for b in baselines]
    errors   = [abs(QC_HW_EGAP - TRUE_GAP)] + [b.error for b in baselines]

    n = len(methods)
    x = np.arange(n)

    bar_colors = []
    for i, (gap, _err) in enumerate(zip(raw_gaps, errors, strict=True)):
        if i == 0:
            bar_colors.append(PRP)   # quantum — purple
        elif round(gap) == TRUE_GAP:
            bar_colors.append(GRN)   # correct integer prediction — green
        else:
            bar_colors.append("#334155")  # wrong — muted

    fig, (ax_gap, ax_err) = plt.subplots(2, 1, figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Classical Baselines vs Quantum Hardware — predicting gap after prime 229  [{ts}]",
        color=FG, fontsize=11, y=0.98,
    )

    # Top panel: predicted gap values
    _dark_ax(ax_gap)
    bars = ax_gap.bar(x, raw_gaps, color=bar_colors, edgecolor=GRID,
                      linewidth=0.5, width=0.6, zorder=3)
    ax_gap.axhline(TRUE_GAP, color=GRN, linewidth=1.5, linestyle="--",
                   label=f"Ground truth gap = {TRUE_GAP}", zorder=4)
    ax_gap.set_ylabel("Predicted gap (raw float)")
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(methods, fontsize=8)
    ax_gap.legend(framealpha=0, labelcolor=GRN, fontsize=9)
    ax_gap.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_gap.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    for bar, gap in zip(bars, raw_gaps, strict=True):
        ax_gap.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{gap:.2f}",
            ha="center", va="bottom", fontsize=8, color=FG,
        )

    # Bottom panel: absolute error
    _dark_ax(ax_err)
    err_colors = [PRP if i == 0 else (GRN if e < 0.5 else RED) for i, e in enumerate(errors)]
    ebars = ax_err.bar(x, errors, color=err_colors, edgecolor=GRID,
                       linewidth=0.5, width=0.6, zorder=3)
    ax_err.axhline(0, color=GRID, linewidth=0.8, zorder=0)
    ax_err.set_ylabel("|predicted gap − true gap|")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(methods, fontsize=8)
    ax_err.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_err.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    for bar, err in zip(ebars, errors, strict=True):
        ax_err.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{err:.3f}",
            ha="center", va="bottom", fontsize=8, color=FG,
        )

    legend_els = [
        plt.Rectangle((0, 0), 1, 1, fc=PRP, label="Quantum HW"),
        plt.Rectangle((0, 0), 1, 1, fc=GRN, label="Rounds to correct (gap=4)"),
        plt.Rectangle((0, 0), 1, 1, fc=RED, label="Error > 0.5"),
        plt.Rectangle((0, 0), 1, 1, fc="#334155", label="Rounds to wrong gap"),
    ]
    ax_err.legend(handles=legend_els, framealpha=0, labelcolor="white",
                  fontsize=8, ncol=2, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "classical_baseline_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p

# ── Markdown report ────────────────────────────────────────────────────────────

def write_md(baselines: list[Baseline], ts: str, out_dir: Path) -> Path:
    qc_err = abs(QC_HW_EGAP - TRUE_GAP)
    qc_correct = QC_HW_RGAP == TRUE_GAP

    lines = [
        "# Classical Baseline Comparison",
        "",
        "**Target:** gap after prime 229  |  **Ground truth:** gap=4, prime=233",
        f"**Date:** {ts[:4]}-{ts[4:6]}-{ts[6:8]}  |  **Quantum HW run:** 20260815_204703 (ibm_kingston)",
        "",
        "---",
        "",
        "## Comparison Table",
        "",
        "| Method | Raw E[gap] | Rounded gap | Predicted prime | Error | Correct? |",
        "|--------|:----------:|:-----------:|:---------------:|------:|:--------:|",
        f"| **Quantum circuit (hardware)** | {QC_HW_EGAP:.4f} | **{QC_HW_RGAP}** | **{QC_HW_PRIME}** | {qc_err:.4f} | {'✓' if qc_correct else '✗'} |",
    ]

    for b in baselines:
        correct = b.rounded == TRUE_GAP
        lines.append(
            f"| {b.name} | {b.raw:.4f} | {b.rounded} | {b.predicted_prime} "
            f"| {b.error:.4f} | {'✓' if correct else '✗'} |"
        )

    # Rank by error
    all_results = [
        ("Quantum circuit (hardware)", QC_HW_EGAP, qc_err, qc_correct),
    ] + [(b.name, b.raw, b.error, b.rounded == TRUE_GAP) for b in baselines]
    ranked = sorted(all_results, key=lambda r: r[2])

    lines += [
        "",
        "---",
        "",
        "## Ranked by Error (lowest to highest)",
        "",
        "| Rank | Method | Error | Rounds correct? |",
        "|------|--------|------:|:---------------:|",
    ]
    for i, (name, _raw, err, correct) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {name} | {err:.4f} | {'✓' if correct else '✗'} |"
        )

    # Narrative
    qc_rank = next(i for i, (n, _, _, _) in enumerate(ranked, 1) if "Quantum" in n)
    methods_beating_qc = [n for n, _, e, _ in ranked if e < qc_err]
    methods_correct = [n for n, _, _, c in all_results if c]

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        f"The quantum circuit hardware result (E[gap]={QC_HW_EGAP:.4f}, error={qc_err:.4f}) ranks "
        f"**{qc_rank} of {len(all_results)}** by absolute error.",
        "",
    ]

    if not methods_beating_qc:
        lines.append(
            "No classical baseline achieves lower error than the quantum hardware result on this instance."
        )
    else:
        lines.append(
            f"Classical methods with lower error: {', '.join(methods_beating_qc)}."
        )

    correct_str = ", ".join(methods_correct) if methods_correct else "none"
    lines += [
        "",
        f"Methods that round to the correct gap=4: **{correct_str}**.",
        "",
        "**Important caveats:**",
        "",
        "- This is a single-instance comparison on one gap value. A robust evaluation "
          "would use backward verification across many held-out gaps.",
        "- The last gap (gap[48]=2) and moving average of the final four gaps (12, 12, 4, 2) "
          "reflect recent sequence history but not its spectral structure.",
        "- The median of prime gaps in this range is 4 — the same as the true answer — "
          "so a median predictor is a strong baseline for this particular instance.",
        "- The FFT predictor's performance depends heavily on `top_k`; a fuller comparison "
          "would sweep it.",
        "- The quantum circuit's recurrent feedback loop introduces structure that pure "
          "statistical baselines cannot replicate, but the single-instance comparison cannot "
          "confirm whether that structure is causally responsible for the correct prediction.",
        "",
        "---",
        "",
        "## Baseline Notes",
        "",
    ]
    for b in baselines:
        lines.append(f"**{b.name}:** {b.note}")
    lines += ["", f"*Generated: {ts}*"]

    p = out_dir / "classical_baseline_comparison.md"
    p.write_text("\n".join(lines))
    return p

# ── JSON output ────────────────────────────────────────────────────────────────

def save_json(baselines: list[Baseline], ts: str, out_dir: Path) -> Path:
    data = {
        "timestamp": ts,
        "target": {"last_prime": LAST_PRIME, "true_gap": TRUE_GAP, "true_prime": TRUE_PRIME},
        "quantum_hw": {
            "method": "Quantum circuit (ibm_kingston, v3-hw, 46 windows)",
            "raw": QC_HW_EGAP,
            "rounded": QC_HW_RGAP,
            "predicted_prime": QC_HW_PRIME,
            "error": round(abs(QC_HW_EGAP - TRUE_GAP), 6),
            "correct": QC_HW_RGAP == TRUE_GAP,
        },
        "classical": [
            {
                "method": b.name,
                "raw": round(b.raw, 6),
                "rounded": b.rounded,
                "predicted_prime": b.predicted_prime,
                "error": round(b.error, 6),
                "correct": b.rounded == TRUE_GAP,
                "note": b.note,
            }
            for b in baselines
        ],
    }
    p = out_dir / "classical_baseline_results.json"
    p.write_text(json.dumps(data, indent=2))
    return p

# ── Auto-commit ────────────────────────────────────────────────────────────────

def auto_commit_push(ts: str) -> None:
    subprocess.run(
        ["git", "add", f"output/prime/{ts}/"],
        check=True, cwd=REPO_ROOT,
    )
    result = subprocess.run(
        ["git", "commit", "-m", f"Classical baseline comparison {ts} — quantum HW vs 6 classical predictors"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  Committed output/prime/{ts}/")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed.")
    else:
        print(f"  Commit skipped: {result.stdout.strip()}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "output" / "prime" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Classical Baseline Comparison")
    print(f"Target: gap after prime {LAST_PRIME}  |  Ground truth: gap={TRUE_GAP}, prime={TRUE_PRIME}")
    print()

    baselines = [
        predict_mean(),
        predict_last(),
        predict_moving_average(4),
        predict_median(),
        predict_linear_regression(),
        predict_fft(top_k=3),
    ]

    # Print table
    print(f"{'Method':<28} {'Raw':>8}  {'Rounded':>7}  {'Prime':>6}  {'Error':>7}  Correct?")
    print("-" * 72)
    print(f"{'Quantum HW (ibm_kingston)':<28} {QC_HW_EGAP:>8.4f}  {QC_HW_RGAP:>7}  {QC_HW_PRIME:>6}  {abs(QC_HW_EGAP-TRUE_GAP):>7.4f}  {'✓' if QC_HW_RGAP == TRUE_GAP else '✗'}")
    for b in baselines:
        correct = b.rounded == TRUE_GAP
        print(f"{b.name:<28} {b.raw:>8.4f}  {b.rounded:>7}  {b.predicted_prime:>6}  {b.error:>7.4f}  {'✓' if correct else '✗'}")
    print()

    print("Writing output files...")
    plot_comparison(baselines, ts, out_dir)
    print("  → classical_baseline_comparison.png")
    write_md(baselines, ts, out_dir)
    print("  → classical_baseline_comparison.md")
    save_json(baselines, ts, out_dir)
    print("  → classical_baseline_results.json")

    auto_commit_push(ts)
    print()


if __name__ == "__main__":
    main()
