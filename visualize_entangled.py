"""visualize_entangled.py

Five visualizations for the quantum prime gaps explicit-entanglement experiment,
followed by a 4-qubit scale-up (simulator run + hardware retention prediction).

Charts saved to quantum_prime_gaps/screenshots/:
  1. distribution_bars.png      — sim vs hw outcome counts, |10⟩ annotated
  2. mi_retention.png           — StatePrep 7q vs Bell+RY 2q, 7%→91% retention
  3. correlation_heatmap.png    — q0–q1 Pearson r, sim vs hw
  4. probability_landscape.png  — topographic terrain map (red peaks, blue valleys)
  5. gap_sequence.png           — all 49 gaps, twin primes + gap-6 runs annotated

Then: 4-qubit circuit — depth, gate counts, MI, retention prediction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_interp_spline

matplotlib.use("Agg")

# ── Palette ───────────────────────────────────────────────────────────────────

BG       = "#06090f"
SURFACE  = "#0d1a28"
BORDER   = "#1e3048"
SIM_COL  = "#10c9a0"
HW_COL   = "#e8784a"
GOLD     = "#f0c030"
TEXT1    = "#c0dcf0"
TEXT2    = "#4a6e90"
TEXT3    = "#243450"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.xaxis.label.set_color(TEXT2)
    ax.yaxis.label.set_color(TEXT2)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, linestyle="--", alpha=0.5)
    if title:
        ax.set_title(title, color=TEXT1, fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)

def fig_setup(w=9, h=5):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig

OUT = Path("quantum_prime_gaps/screenshots")
OUT.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

SIM = json.loads((OUT / "results_aersimulator.json").read_text())
HW  = json.loads((OUT / "results_ibm_kingston.json").read_text())

SHOTS = SIM["shots"]
STATES = ["00", "01", "10", "11"]
LABELS = [f"|{s}⟩" for s in STATES]

sim_counts = [SIM["counts"].get(s, 0) for s in STATES]
hw_counts  = [HW["counts"].get(s, 0)  for s in STATES]
sim_probs  = [c / SHOTS for c in sim_counts]
hw_probs   = [c / SHOTS for c in hw_counts]

FIRST_50_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
]
ALL_GAPS = [FIRST_50_PRIMES[i + 1] - FIRST_50_PRIMES[i] for i in range(len(FIRST_50_PRIMES) - 1)]
MAX_GAP  = max(ALL_GAPS)  # 14

# ── Chart 1: Distribution bars ────────────────────────────────────────────────

def chart_distribution():
    fig = fig_setup(9, 5)
    ax = fig.add_subplot(111)
    style_ax(ax, "Outcome Distribution — Simulator vs Hardware",
             "Basis state", "Probability")

    x = np.arange(4)
    w = 0.35
    bars_s = ax.bar(x - w/2, sim_probs, w, label="Simulator", color=SIM_COL, alpha=0.85, zorder=3)
    bars_h = ax.bar(x + w/2, hw_probs,  w, label="Hardware (ibm_kingston)", color=HW_COL, alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11, color=TEXT1)
    ax.set_ylim(0, 0.60)

    # Independent-qubit reference line at 0.25
    ax.axhline(0.25, color=GOLD, linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)
    ax.text(3.5, 0.252, "independent\nqubits (0.25)", color=GOLD, fontsize=7.5,
            va="bottom", ha="right", alpha=0.8)

    # Annotate |10⟩ suppression
    i10 = STATES.index("10")
    ax.annotate(
        f"  |10⟩ suppressed\n  {hw_probs[i10]*100:.1f}% hw  (25% expected)",
        xy=(i10 + w/2, hw_probs[i10]),
        xytext=(i10 + 0.9, hw_probs[i10] + 0.18),
        color=GOLD, fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
        zorder=5,
    )

    # Value labels on bars
    for bar in bars_s:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color=SIM_COL)
    for bar in bars_h:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color=HW_COL)

    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT1, fontsize=9)
    fig.tight_layout(pad=1.4)
    fig.savefig(OUT / "distribution_bars.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  ✓ distribution_bars.png")


# ── Chart 2: MI retention comparison ─────────────────────────────────────────

def chart_mi_retention():
    # Old StatePrep 7q values (from archive/2026-08-15/qubit_hierarchy_report.md)
    old_sim_mi = 0.8599   # root MI sim
    old_hw_mi  = 0.0598   # root MI hw (7% retention)
    new_sim_mi = SIM["mi_bits"]   # 0.2681
    new_hw_mi  = HW["mi_bits"]    # 0.2446

    old_ret = old_hw_mi / old_sim_mi   # 0.0696 → 7%
    new_ret = new_hw_mi / new_sim_mi   # 0.9133 → 91%

    fig = fig_setup(10, 5)
    ax = fig.add_subplot(111)
    style_ax(ax, "MI Retention: StatePrep 7q  vs  Bell+RY 2q",
             "Circuit", "Mutual Information (bits)")

    x = np.array([0, 1, 3, 4])
    vals = [old_sim_mi, old_hw_mi, new_sim_mi, new_hw_mi]
    cols = [SIM_COL, HW_COL, SIM_COL, HW_COL]
    bars = ax.bar(x, vals, 0.6, color=cols, alpha=0.85, zorder=3)

    # Retention ratio brackets
    # Old: bracket from bar[1] to bar[0]
    y_brk = max(old_sim_mi, old_hw_mi) + 0.06
    ax.plot([0-0.3, 0-0.3, 1+0.3, 1+0.3],
            [old_sim_mi+0.02, y_brk, y_brk, old_hw_mi+0.02],
            color=HW_COL, linewidth=1, alpha=0.7, zorder=4)
    ax.text(0.5, y_brk + 0.03, f"7% retained\n(×{old_ret:.2f})",
            ha="center", color=HW_COL, fontsize=9, fontweight="bold")

    # New: bracket
    y_brk2 = max(new_sim_mi, new_hw_mi) + 0.06
    ax.plot([3-0.3, 3-0.3, 4+0.3, 4+0.3],
            [new_sim_mi+0.02, y_brk2, y_brk2, new_hw_mi+0.02],
            color=GOLD, linewidth=1, alpha=0.9, zorder=4)
    ax.text(3.5, y_brk2 + 0.03, f"91% retained\n(×{new_ret:.2f})",
            ha="center", color=GOLD, fontsize=9, fontweight="bold")

    # X-axis labels
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["StatePrep + iQFT\n7 qubits", "Bell + RY + iQFT\n2 qubits"],
                       fontsize=10, color=TEXT1)
    ax.set_xlim(-0.7, 5.2)
    ax.set_ylim(0, 1.15)

    # Value labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT1)

    # Legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=SIM_COL, alpha=0.85, label="Simulator"),
                  Patch(facecolor=HW_COL,  alpha=0.85, label="Hardware")]
    ax.legend(handles=legend_els, facecolor=SURFACE, edgecolor=BORDER,
              labelcolor=TEXT1, fontsize=9)

    fig.tight_layout(pad=1.4)
    fig.savefig(OUT / "mi_retention.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  ✓ mi_retention.png")


# ── Chart 3: Correlation heatmap ──────────────────────────────────────────────

def chart_correlation_heatmap():
    sim_r = SIM["pearson_r"]
    hw_r  = HW["pearson_r"]

    sim_mat = np.array([[1.0, sim_r], [sim_r, 1.0]])
    hw_mat  = np.array([[1.0, hw_r],  [hw_r,  1.0]])

    fig = fig_setup(9, 4)
    fig.suptitle("Pearson Correlation — q0 vs q1", color=TEXT1, fontsize=11,
                 fontweight="bold", y=0.97)

    cmap = LinearSegmentedColormap.from_list(
        "diverg",
        [(0.0, "#1e3048"), (0.35, SURFACE), (0.5, "#c0dcf0"), (0.75, SIM_COL), (1.0, "#00ffcc")],
    )

    for idx, (mat, label, r_val) in enumerate(
        [(sim_mat, "Simulator", sim_r), (hw_mat, "Hardware (ibm_kingston)", hw_r)]
    ):
        ax = fig.add_subplot(1, 2, idx + 1)
        ax.set_facecolor(SURFACE)
        ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["q0", "q1"], color=TEXT1, fontsize=10)
        ax.set_yticklabels(["q0", "q1"], color=TEXT1, fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT2)

        # Cell annotations
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                txt_col = "#0a1828" if val > 0.5 else TEXT1
                ax.text(j, i, f"{val:+.4f}" if val != 1.0 else "1.000",
                        ha="center", va="center", fontsize=12,
                        fontweight="bold", color=txt_col)

        col = SIM_COL if idx == 0 else HW_COL
        ax.set_title(f"{label}\nr = {r_val:+.4f}", color=col, fontsize=10, pad=8)

    fig.tight_layout(pad=1.6)
    fig.savefig(OUT / "correlation_heatmap.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  ✓ correlation_heatmap.png")


# ── Chart 4: Probability landscape (topographic) ─────────────────────────────

TOPO_CMAP = LinearSegmentedColormap.from_list(
    "topo",
    [
        (0.00, "#03050a"),   # abyssal black
        (0.18, "#0a1f3a"),   # deep navy
        (0.35, "#0e4060"),   # ocean blue
        (0.50, "#1a6a5a"),   # shallow teal
        (0.62, "#4a9040"),   # upland green
        (0.74, "#c8a020"),   # highland gold
        (0.86, "#d04010"),   # peak orange
        (1.00, "#ff1010"),   # summit red
    ],
)

def chart_probability_landscape():
    fig = fig_setup(10, 5)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    x_int = np.array([0.0, 1.0, 2.0, 3.0])

    # Smooth both curves via B-spline interpolation
    x_fine = np.linspace(0, 3, 600)
    for probs, label, alpha_scale in [
        (sim_probs, "Simulator", 1.0),
        (hw_probs,  "Hardware",  0.72),
    ]:
        spl = make_interp_spline(x_int, probs, k=3)
        p_smooth = np.clip(spl(x_fine), 0, None)

        # Topographic fill: stack colored horizontal bands
        p_max = max(probs) * 1.05
        n_bands = 240
        band_h = p_max / n_bands
        for b in range(n_bands):
            y_lo = b * band_h
            y_hi = y_lo + band_h
            t = b / n_bands
            color = TOPO_CMAP(t)
            mask = p_smooth >= y_lo
            # Fill band only where curve is above this level
            ax.fill_between(x_fine, y_lo, np.where(mask, np.minimum(p_smooth, y_hi), y_lo),
                            color=color, alpha=alpha_scale * 0.9, linewidth=0)

        # Contour lines
        for level in np.linspace(0.02, max(probs)*0.98, 8):
            ax.plot(x_fine, np.where(p_smooth >= level, level, np.nan),
                    color="white", linewidth=0.3, alpha=0.25 * alpha_scale)

        # Ridge line
        lc = SIM_COL if label == "Simulator" else HW_COL
        ax.plot(x_fine, p_smooth, color=lc, linewidth=1.6 if label=="Simulator" else 1.1,
                alpha=0.9, label=label, zorder=5)

    # Data point markers
    ax.scatter(x_int, sim_probs, color=SIM_COL, s=60, zorder=6, edgecolors="white", linewidth=0.8)
    ax.scatter(x_int, hw_probs,  color=HW_COL,  s=40, zorder=6, edgecolors="white", linewidth=0.8,
               marker="D")

    # |10⟩ annotation
    ax.annotate(
        f"|10⟩ valley\n{hw_probs[2]*100:.2f}% hw",
        xy=(2, hw_probs[2]), xytext=(2.4, 0.12),
        color=GOLD, fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.1), zorder=7,
    )

    ax.set_xticks(x_int)
    ax.set_xticklabels(LABELS, fontsize=11, color=TEXT1)
    ax.set_ylabel("Probability", color=TEXT2, fontsize=9)
    ax.tick_params(colors=TEXT2)
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(0, max(sim_probs) * 1.18)
    ax.set_title("Probability Landscape — Topographic", color=TEXT1,
                 fontsize=11, fontweight="bold", pad=10)

    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT1, fontsize=9, loc="upper right")

    fig.tight_layout(pad=1.4)
    fig.savefig(OUT / "probability_landscape.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  ✓ probability_landscape.png")


# ── Chart 5: Gap sequence ─────────────────────────────────────────────────────

def chart_gap_sequence():
    gaps = ALL_GAPS
    n = len(gaps)
    xs = np.arange(n)

    twin_prime_idx = [i for i, g in enumerate(gaps) if g == 2]
    gap6_idx       = [i for i, g in enumerate(gaps) if g == 6]
    large_idx      = [i for i, g in enumerate(gaps) if g >= 10]

    fig = fig_setup(12, 5)
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    # Base bars
    ax.bar(xs, gaps, color=TEXT3, alpha=0.5, zorder=2, width=0.8)

    # Colour-coded highlights
    ax.bar(twin_prime_idx, [gaps[i] for i in twin_prime_idx],
           color=SIM_COL, alpha=0.85, zorder=3, width=0.8, label="Gap = 2 (twin primes)")
    ax.bar(gap6_idx, [gaps[i] for i in gap6_idx],
           color=HW_COL, alpha=0.75, zorder=3, width=0.8, label="Gap = 6")
    ax.bar(large_idx, [gaps[i] for i in large_idx],
           color=GOLD, alpha=0.9, zorder=4, width=0.8, label="Gap ≥ 10 (desert)")

    # Step line
    ax.step(xs, gaps, where="mid", color=TEXT2, linewidth=0.8, alpha=0.6, zorder=5)

    # Annotate the MAX_GAP=14
    max_i = gaps.index(MAX_GAP)
    ax.annotate(
        f"Gap = {MAX_GAP}\n({FIRST_50_PRIMES[max_i]}→{FIRST_50_PRIMES[max_i+1]})",
        xy=(max_i, MAX_GAP), xytext=(max_i + 2.5, MAX_GAP - 1),
        color=GOLD, fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2), zorder=6,
    )

    # Annotate first used gaps [1, 2] (circuit encoding)
    ax.axvspan(-0.5, 1.5, alpha=0.08, color=SIM_COL, zorder=1)
    ax.text(0.5, MAX_GAP * 0.92, "circuit\nencodes\nthese 2",
            ha="center", color=SIM_COL, fontsize=7.5, alpha=0.9)

    # X-axis: prime labels at every 5th gap
    tick_pos  = list(range(0, n, 5))
    tick_lbls = [f"p{i+1}→p{i+2}\n({FIRST_50_PRIMES[i]})" for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbls, fontsize=7, color=TEXT2)
    ax.set_ylabel("Gap size", color=TEXT2, fontsize=9)
    ax.set_title("Prime Gap Sequence — first 49 gaps (primes 2–229)",
                 color=TEXT1, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, MAX_GAP * 1.18)
    ax.tick_params(colors=TEXT2)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, linestyle="--", alpha=0.4)
    ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT1, fontsize=9)

    fig.tight_layout(pad=1.4)
    fig.savefig(OUT / "gap_sequence.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("  ✓ gap_sequence.png")


# ── 4-qubit scale-up ──────────────────────────────────────────────────────────

def run_4qubit():
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFTGate
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    N = 4
    gaps4 = ALL_GAPS[:N]  # [1, 2, 2, 4]
    angles4 = [g * math.pi / MAX_GAP for g in gaps4]

    # Build circuit: Bell pair (q0-q1) → RY on all 4 → 4-qubit iQFT
    qc = QuantumCircuit(N, N)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier(label="Bell pair")
    for i, theta in enumerate(angles4):
        qc.ry(theta, i)
    qc.barrier(label=f"gaps {gaps4}")
    iqft = QFTGate(N).inverse()
    qc.append(iqft, range(N))
    qc.barrier(label="iQFT")
    qc.measure(range(N), range(N))

    print("\n── 4-Qubit Circuit ─────────────────────────────────────────")
    print(qc.draw(output="text", fold=90))

    # Transpile to Kingston-class backend
    backend = FakeSherbrooke()
    tqc = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
    ops = tqc.count_ops()
    two_q  = sum(v for k, v in ops.items() if k in {"ecr", "cx", "cz"})
    one_q  = sum(v for k, v in ops.items() if k in {"sx", "rz", "x"})
    total  = sum(v for k, v in ops.items() if k not in {"barrier", "measure"})
    depth4 = tqc.depth()

    # Simulate
    sim = AerSimulator()
    tqc_sim = transpile(qc, sim)
    result = sim.run(tqc_sim, shots=8192).result()
    counts4 = result.get_counts()

    # Pairwise MI (q0-q1 as root split)
    shots4 = sum(counts4.values())
    bits4 = np.zeros((shots4, N), dtype=np.int8)
    row = 0
    for bs, count in counts4.items():
        bs = bs.zfill(N)
        for _ in range(count):
            for q in range(N):
                bits4[row, q] = int(bs[N - 1 - q])
            row += 1

    def mi_pair(b, i, j):
        n = len(b)
        p_i = np.bincount(b[:, i].astype(int), minlength=2) / n
        p_j = np.bincount(b[:, j].astype(int), minlength=2) / n
        joint = np.bincount(b[:, i].astype(int) * 2 + b[:, j].astype(int), minlength=4) / n
        def h(p):
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        def mm(p: np.ndarray) -> float:
            return (np.sum(p > 0) - 1) / (2 * n * math.log(2))
        return max(0.0, h(p_i) + mm(p_i) + h(p_j) + mm(p_j) - h(joint) - mm(joint))

    # Root MI: first half [q0,q1] vs second half [q2,q3]
    def mi_halves(bits, left_qs, right_qs):
        n = len(bits)
        # Encode each half as a single integer
        l_enc = sum(bits[:, q] * (2 ** i) for i, q in enumerate(left_qs))
        r_enc = sum(bits[:, q] * (2 ** i) for i, q in enumerate(right_qs))
        kl = 2 ** len(left_qs)
        kr = 2 ** len(right_qs)
        p_l = np.bincount(l_enc.astype(int), minlength=kl) / n
        p_r = np.bincount(r_enc.astype(int), minlength=kr) / n
        joint = np.bincount(l_enc.astype(int) * kr + r_enc.astype(int),
                            minlength=kl * kr) / n
        def h(p: np.ndarray) -> float:
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        def mm(p: np.ndarray) -> float:
            return (np.sum(p > 0) - 1) / (2 * n * math.log(2))
        return max(0.0, h(p_l) + mm(p_l) + h(p_r) + mm(p_r) - h(joint) - mm(joint))

    root_mi4 = mi_halves(bits4, [0, 1], [2, 3])
    mi_01 = mi_pair(bits4, 0, 1)

    # Hardware retention prediction
    # Calibrate: 2q circuit retained 91% at depth 18
    # Model: retention = exp(-depth / T2_depth)
    ret_2q  = HW["mi_bits"] / SIM["mi_bits"]  # 0.913
    depth2  = 18
    T2_depth = -depth2 / math.log(ret_2q)      # characteristic depth scale
    pred_ret = math.exp(-depth4 / T2_depth)

    print("\n── 4-Qubit Results ─────────────────────────────────────────")
    print(f"  Gaps encoded:    {gaps4}")
    print(f"  RY angles (rad): {[f'{a:.4f}' for a in angles4]}")
    print()
    print("  Transpiled to Kingston-class (opt=3, seed=42):")
    print(f"    Circuit depth:  {depth4}")
    print(f"    ECR gates:      {two_q}")
    print(f"    1q gates:       {one_q}")
    print(f"    Total gates:    {total}")
    print()
    print("  AerSimulator (8192 shots):")
    print(f"    Root MI [q0,q1|q2,q3]: {root_mi4:.4f} bits")
    print(f"    Pairwise MI q0-q1:     {mi_01:.4f} bits")
    print()
    print("  Hardware retention prediction:")
    print(f"    T2 depth-scale (calibrated from 2q run): {T2_depth:.1f} layers")
    print(f"    Predicted retention at depth={depth4}: {pred_ret*100:.1f}%")
    print(f"    → Predicted HW root MI: {root_mi4 * pred_ret:.4f} bits")

    # Save
    import json as _json
    result4 = {
        "n_qubits": N,
        "gaps": gaps4,
        "angles_rad": [round(a, 6) for a in angles4],
        "shots": 8192,
        "transpiled_depth": depth4,
        "gates_2q": two_q,
        "gates_1q": one_q,
        "gates_total": total,
        "sim_root_mi_bits": round(root_mi4, 6),
        "sim_mi_q0q1_bits": round(mi_01, 6),
        "hw_retention_model": {
            "t2_depth_scale": round(T2_depth, 2),
            "predicted_retention_pct": round(pred_ret * 100, 1),
            "predicted_hw_root_mi": round(root_mi4 * pred_ret, 6),
        },
    }
    (OUT / "results_4qubit_sim.json").write_text(_json.dumps(result4, indent=2))
    print(f"\n  Saved → {OUT}/results_4qubit_sim.json")

    return result4


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Quantum Prime Gaps — Visualization + 4-Qubit Scale-Up")
    print("=" * 60)
    print("\nGenerating charts...")
    chart_distribution()
    chart_mi_retention()
    chart_correlation_heatmap()
    chart_probability_landscape()
    chart_gap_sequence()
    print("\nRunning 4-qubit scale-up...")
    run_4qubit()
    print("\nDone.")


if __name__ == "__main__":
    main()
