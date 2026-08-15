"""prime239_generalization.py

Generalization test: predict the gap after prime 233 (ground truth = 6, prime 239).

This extends the training set by one prime (233) and runs the same recurrent
feedback loop, but submits only the final window to ibm_kingston hardware:

  - Windows 0–45: AerSimulator + Kingston noise model (fast, free, chain accumulates)
  - Window 46   : ibm_kingston real hardware (1 job, 8192 shots)

If the quantum circuit correctly predicts gap=6 and the median predictor (gap=4)
doesn't, the "median coincidence" argument from the prime 233 baseline collapses.

Ground truth: prime[51]=233, prime[52]=239, gap=6
Final window: gaps[46:50] = [12, 4, 2, 4]  local_max=12

Output → output/prime/{YYYYMMDD_HHMMSS}/:
  prime239_generalization.md    comparison table + interpretation
  prime239_generalization.png   bar chart
  prime239_results.json         machine-readable results

Auto-commits and pushes.

Run: python prime239_generalization.py
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis.qft import synth_qft_full
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

# ── Constants ──────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent
BACKEND_NAME = "ibm_kingston"
SCALE        = 0.05
SHOTS        = 8_192
WINDOW_SIZE  = 4

# ── Extended prime data (51 primes) ───────────────────────────────────────────

FIRST_51_PRIMES: list[int] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
]

ALL_GAPS: list[int] = [
    FIRST_51_PRIMES[i + 1] - FIRST_51_PRIMES[i]
    for i in range(len(FIRST_51_PRIMES) - 1)
]  # 50 gaps

LAST_KNOWN_PRIME = FIRST_51_PRIMES[-1]  # 233
TRUE_NEXT_PRIME  = 239
TRUE_GAP         = TRUE_NEXT_PRIME - LAST_KNOWN_PRIME  # 6

# 47 windows (one more than the prime-233 run)
WINDOWS: list[list[int]] = [
    ALL_GAPS[i : i + WINDOW_SIZE]
    for i in range(len(ALL_GAPS) - WINDOW_SIZE + 1)
]

# ── Circuit (same architecture as v3) ─────────────────────────────────────────

def build_circuit(gaps: list[int], offsets: list[float]) -> tuple[QuantumCircuit, int]:
    local_max = max(gaps)
    qc = QuantumCircuit(WINDOW_SIZE, WINDOW_SIZE)
    qc.h(0)
    qc.cx(0, 1)
    for i, gap in enumerate(gaps):
        angle = gap * math.pi / local_max + offsets[i]
        qc.ry(angle, i)
    iqft = synth_qft_full(WINDOW_SIZE, inverse=True, do_swaps=True, approximation_degree=1)
    qc.compose(iqft, inplace=True)
    qc.measure(range(WINDOW_SIZE), range(WINDOW_SIZE))
    return qc, local_max


def bitstring_to_offsets(bs: str, scale: float) -> list[float]:
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

# ── Decode ─────────────────────────────────────────────────────────────────────

def decode_prediction(counts: dict, local_max: int) -> tuple[float, int, str, int]:
    total = sum(counts.values())
    weighted_gap = 0.0
    for bs, cnt in counts.items():
        val = int(bs, 2)
        gap = val * local_max / 15
        weighted_gap += (cnt / total) * gap
    mode_bs  = max(counts, key=counts.get)
    mode_gap = int(mode_bs, 2) * local_max / 15
    return weighted_gap, max(1, round(weighted_gap)), mode_bs, max(1, round(mode_gap))

# ── Sim feedback chain (windows 0 – n_sim-1) ──────────────────────────────────

def run_sim_chain(n_sim: int, noise_model: NoiseModel) -> tuple[list[float], list[dict]]:
    """Run n_sim windows on noisy AerSimulator, return (final_offsets, per_window_info)."""
    sim = AerSimulator(noise_model=noise_model)
    offsets = [0.0] * WINDOW_SIZE
    window_info = []

    for w_idx in range(n_sim):
        window = WINDOWS[w_idx]
        qc, local_max = build_circuit(window, offsets)
        tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=42)
        counts = sim.run(tqc, shots=SHOTS).result().get_counts()
        bits = counts_to_bits(counts, WINDOW_SIZE)
        mi = mi_halves(bits, [0, 1], [2, 3])
        mode_bs = max(counts, key=counts.get)
        next_offsets = bitstring_to_offsets(mode_bs, SCALE)
        window_info.append({
            "w": w_idx, "gaps": window, "local_max": local_max,
            "mi": round(mi, 6), "mode": mode_bs,
            "offsets_in": [round(o, 4) for o in offsets],
        })
        print(
            f"  sim w{w_idx:02d}/{n_sim-1}  gaps={window}  "
            f"lmax={local_max}  MI={mi:.4f}  mode={mode_bs}",
            flush=True,
        )
        offsets = next_offsets

    return offsets, window_info

# ── Hardware final window ──────────────────────────────────────────────────────

def run_hw_window(
    backend, pm, final_offsets: list[float]
) -> tuple[dict, str, int]:
    """Submit the final window (window 46) to hardware. Returns (counts, job_id, local_max)."""
    sampler = Sampler(mode=backend)
    window = WINDOWS[-1]
    qc, local_max = build_circuit(window, final_offsets)
    isa_qc = pm.run(qc)
    job = sampler.run([isa_qc], shots=SHOTS)
    job_id = job.job_id()
    print(f"\n  HW job {job_id} submitted — window={window}  lmax={local_max}  waiting...", flush=True)
    t0 = time.time()
    result = job.result()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s", flush=True)

    pub_result = result[0]
    bitarray = pub_result.data.c
    counts: dict[str, int] = {}
    for bs in bitarray.get_bitstrings():
        counts[bs] = counts.get(bs, 0) + 1
    return counts, job_id, local_max

# ── Classical baselines (on 50 gaps) ──────────────────────────────────────────

@dataclass
class Baseline:
    name: str
    raw: float
    rounded: int
    predicted_prime: int
    error: float
    note: str = ""


def _make(name: str, raw: float, note: str = "") -> Baseline:
    rounded = max(1, round(raw))
    return Baseline(
        name=name, raw=raw, rounded=rounded,
        predicted_prime=LAST_KNOWN_PRIME + rounded,
        error=abs(raw - TRUE_GAP), note=note,
    )


def classical_baselines() -> list[Baseline]:
    gaps = np.array(ALL_GAPS, dtype=float)
    n = len(gaps)

    mean_val = float(np.mean(gaps))
    median_val = float(np.median(gaps))
    last4 = ALL_GAPS[-4:]
    moving_val = float(np.mean(last4))

    # FFT top-3 extrapolation
    spectrum = np.fft.fft(gaps)
    magnitudes = np.abs(spectrum)
    top_idx = np.argsort(magnitudes)[::-1][:3]
    filtered = np.zeros_like(spectrum)
    for idx in top_idx:
        filtered[idx] = spectrum[idx]
        conj = n - idx
        if 0 < conj < n:
            filtered[conj] = spectrum[conj]
    t = float(n)
    raw_complex = sum(filtered[k] * np.exp(2j * math.pi * k * t / n) for k in range(n))
    fft_val = float(np.real(raw_complex)) / n

    return [
        _make("Median gap (50)", median_val, f"median of all {n} gaps"),
        _make("FFT (top-3)", fft_val, f"DFT extrapolation at t={n}, top-3 components"),
        _make("Moving avg (w=4)", moving_val, f"mean of last 4 gaps: {last4}"),
        _make("Mean gap (50)", mean_val, f"mean of all {n} gaps = {mean_val:.4f}"),
    ]

# ── Plotting ───────────────────────────────────────────────────────────────────

BG   = "#0d1117"
GRID = "#1e293b"
MUT  = "#94a3b8"
FG   = "#f8fafc"
GRN  = "#22c55e"
RED  = "#ef4444"
ORG  = "#fb923c"
PRP  = "#a78bfa"
BLU  = "#7dd3fc"


def _dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT)
    ax.spines[:].set_color(GRID)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    ax.title.set_color(FG)


def plot_results(
    hw_egap: float, baselines: list[Baseline],
    sim_window_info: list[dict], ts: str, out_dir: Path,
) -> Path:
    methods = ["Quantum HW\n(ibm_kingston)"] + [b.name.replace(" (", "\n(") for b in baselines]
    raw_gaps = [hw_egap] + [b.raw for b in baselines]
    errors   = [abs(hw_egap - TRUE_GAP)] + [b.error for b in baselines]

    n = len(methods)
    x = np.arange(n)

    bar_colors = []
    for i, gap in enumerate(raw_gaps):
        if i == 0:
            bar_colors.append(PRP)
        elif round(gap) == TRUE_GAP:
            bar_colors.append(GRN)
        else:
            bar_colors.append("#334155")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1.5], hspace=0.45)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Prime 239 Generalization Test — predicting gap after prime 233  [{ts}]\n"
        f"Ground truth: gap = {TRUE_GAP}  →  prime {TRUE_NEXT_PRIME}",
        color=FG, fontsize=11, y=0.98,
    )

    # Top: predicted gap bars
    ax_gap = fig.add_subplot(gs[0])
    _dark_ax(ax_gap)
    bars = ax_gap.bar(x, raw_gaps, color=bar_colors, edgecolor=GRID,
                      linewidth=0.5, width=0.6, zorder=3)
    ax_gap.axhline(TRUE_GAP, color=GRN, linewidth=1.5, linestyle="--",
                   label=f"Ground truth gap = {TRUE_GAP}", zorder=4)
    ax_gap.set_ylabel("Predicted gap (raw float)")
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(methods, fontsize=9)
    ax_gap.legend(framealpha=0, labelcolor=GRN, fontsize=9)
    ax_gap.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax_gap.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    for bar, gap in zip(bars, raw_gaps, strict=True):
        ax_gap.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{gap:.2f}", ha="center", va="bottom", fontsize=9, color=FG,
        )

    # Middle: error bars
    ax_err = fig.add_subplot(gs[1])
    _dark_ax(ax_err)
    err_colors = [PRP if i == 0 else (GRN if e < 0.5 else RED) for i, e in enumerate(errors)]
    ebars = ax_err.bar(x, errors, color=err_colors, edgecolor=GRID,
                       linewidth=0.5, width=0.6, zorder=3)
    ax_err.axhline(0, color=GRID, linewidth=0.8)
    ax_err.set_ylabel("|predicted − true gap|")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(methods, fontsize=9)
    ax_err.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax_err.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    for bar, err in zip(ebars, errors, strict=True):
        ax_err.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f"{err:.3f}", ha="center", va="bottom", fontsize=9, color=FG,
        )
    legend_els = [
        plt.Rectangle((0, 0), 1, 1, fc=PRP, label="Quantum HW"),
        plt.Rectangle((0, 0), 1, 1, fc=GRN, label="Rounds to correct (gap=6)"),
        plt.Rectangle((0, 0), 1, 1, fc=RED, label="Error > 0.5"),
        plt.Rectangle((0, 0), 1, 1, fc="#334155", label="Rounds wrong"),
    ]
    ax_err.legend(handles=legend_els, framealpha=0, labelcolor="white",
                  fontsize=8, ncol=2)

    # Bottom: sim MI trace across windows 0-45
    ax_mi = fig.add_subplot(gs[2])
    _dark_ax(ax_mi)
    mi_vals = [w["mi"] for w in sim_window_info]
    ax_mi.plot(range(len(mi_vals)), mi_vals, color=BLU, linewidth=1.5, zorder=3)
    ax_mi.fill_between(range(len(mi_vals)), mi_vals, alpha=0.12, color=BLU)
    ax_mi.set_xlabel("Sim window index (0–45, noise model)")
    ax_mi.set_ylabel("Root MI (bits)")
    ax_mi.set_title("Noisy-sim MI chain — feedback offsets fed to HW final window", fontsize=9)
    ax_mi.grid(axis="y", color=GRID, linewidth=0.5)

    p = out_dir / "prime239_generalization.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p

# ── Markdown report ────────────────────────────────────────────────────────────

def write_md(
    hw_egap: float, hw_rounded: int, hw_mode: str, hw_mode_gap: int,
    hw_mi: float, hw_job_id: str, hw_local_max: int,
    baselines: list[Baseline], sim_window_info: list[dict],
    ts: str, out_dir: Path,
) -> Path:
    hw_error   = abs(hw_egap - TRUE_GAP)
    hw_correct = hw_rounded == TRUE_GAP

    all_results = [
        ("Quantum HW (ibm_kingston)", hw_egap, hw_error, hw_correct),
    ] + [(b.name, b.raw, b.error, b.rounded == TRUE_GAP) for b in baselines]
    ranked = sorted(all_results, key=lambda r: r[2])
    qc_rank = next(i for i, (n, *_) in enumerate(ranked, 1) if "Quantum" in n)
    correct_methods = [n for n, _, _, c in all_results if c]
    mean_sim_mi = sum(w["mi"] for w in sim_window_info) / len(sim_window_info)

    lines = [
        "# Prime 239 Generalization Test",
        "",
        f"**Date:** {ts[:4]}-{ts[4:6]}-{ts[6:8]}",
        f"**Target:** gap after prime {LAST_KNOWN_PRIME}  |  **Ground truth:** gap={TRUE_GAP}, prime={TRUE_NEXT_PRIME}",
        "**Architecture:** 47 windows (46 noisy-sim + 1 ibm_kingston hardware)",
        f"**Hardware job:** `{hw_job_id}`  |  **Backend:** {BACKEND_NAME}  |  **Shots:** {SHOTS:,}",
        "",
        "---",
        "",
        "## Comparison Table",
        "",
        "| Method | Raw E[gap] | Rounded | Prime | Error | Correct? |",
        "|--------|:----------:|:-------:|:-----:|------:|:--------:|",
        f"| **Quantum HW (ibm_kingston)** | {hw_egap:.4f} | **{hw_rounded}** | **{LAST_KNOWN_PRIME + hw_rounded}** | {hw_error:.4f} | {'✓' if hw_correct else '✗'} |",
    ]
    for b in baselines:
        correct = b.rounded == TRUE_GAP
        lines.append(
            f"| {b.name} | {b.raw:.4f} | {b.rounded} | {b.predicted_prime} | {b.error:.4f} | {'✓' if correct else '✗'} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Ranked by Error",
        "",
        "| Rank | Method | Error | Correct? |",
        "|------|--------|------:|:--------:|",
    ]
    for i, (name, _raw, err, correct) in enumerate(ranked, 1):
        lines.append(f"| {i} | {name} | {err:.4f} | {'✓' if correct else '✗'} |")

    # Determine if the coincidence argument collapsed
    median_correct = any("Median" in n and c for n, _, _, c in all_results)
    coincidence_status = (
        "The median predictor was correct on prime 233 because the median of gaps 1–229 happens "
        "to equal 4. On this instance (target gap=6), the median predicts gap="
        + str(next(b.rounded for b in baselines if "Median" in b.name))
        + (" — the coincidence argument collapses." if not median_correct else
           " — still correct, coincidence persists.")
    )

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        f"Quantum circuit ranked **{qc_rank} of {len(all_results)}** by absolute error.",
        "",
        f"Methods rounding to the correct gap={TRUE_GAP}: **{', '.join(correct_methods) if correct_methods else 'none'}**.",
        "",
        coincidence_status,
        "",
        "**Hardware details:**",
        f"- Final window: gaps={WINDOWS[-1]}  local_max={hw_local_max}",
        f"- Feedback offsets derived from 46-window noisy-sim chain (mean MI={mean_sim_mi:.4f} bits)",
        f"- Hardware MI (final window): {hw_mi:.4f} bits",
        f"- Mode bitstring: `{hw_mode}` → mode gap≈{hw_mode_gap}",
        f"- Weighted E[gap]: {hw_egap:.4f} → rounded gap={hw_rounded} → prime {LAST_KNOWN_PRIME + hw_rounded}",
        "",
        "**Context:** This is the first test of the circuit on a prime outside its original training window.",
        "Windows 0–45 ran on AerSimulator with the Kingston noise model to propagate the feedback chain.",
        "Window 46 (the prediction) ran on real ibm_kingston hardware with those accumulated offsets.",
        "",
        "---",
        "",
        "## Noisy-Sim Chain (windows 0–45)",
        "",
        "| Window | Gaps | lmax | MI | Mode | Offsets in |",
        "|--------|------|------|----|------|------------|",
    ]
    for w in sim_window_info:
        off_str = "[" + ",".join(f"{o:+.3f}" for o in w["offsets_in"]) + "]"
        lines.append(
            f"| {w['w']} | {w['gaps']} | {w['local_max']} | {w['mi']:.4f} | `{w['mode']}` | {off_str} |"
        )

    lines += ["", f"*Generated: {ts}*"]
    p = out_dir / "prime239_generalization.md"
    p.write_text("\n".join(lines))
    return p

# ── JSON output ────────────────────────────────────────────────────────────────

def save_json(
    hw_egap: float, hw_rounded: int, hw_mi: float, hw_job_id: str,
    baselines: list[Baseline], sim_window_info: list[dict],
    ts: str, out_dir: Path,
) -> Path:
    data = {
        "timestamp": ts,
        "target": {
            "last_prime": LAST_KNOWN_PRIME,
            "true_gap": TRUE_GAP,
            "true_prime": TRUE_NEXT_PRIME,
        },
        "architecture": {
            "sim_windows": len(sim_window_info),
            "hw_windows": 1,
            "total_windows": len(WINDOWS),
            "scale": SCALE,
            "backend": BACKEND_NAME,
            "shots": SHOTS,
        },
        "quantum_hw": {
            "job_id": hw_job_id,
            "raw_egap": round(hw_egap, 6),
            "rounded": hw_rounded,
            "predicted_prime": LAST_KNOWN_PRIME + hw_rounded,
            "error": round(abs(hw_egap - TRUE_GAP), 6),
            "correct": hw_rounded == TRUE_GAP,
            "hw_mi": round(hw_mi, 6),
        },
        "classical": [
            {
                "method": b.name, "raw": round(b.raw, 6),
                "rounded": b.rounded,
                "predicted_prime": b.predicted_prime,
                "error": round(b.error, 6),
                "correct": b.rounded == TRUE_GAP,
            }
            for b in baselines
        ],
        "sim_chain": sim_window_info,
    }
    p = out_dir / "prime239_results.json"
    p.write_text(json.dumps(data, indent=2))
    return p

# ── Auto-commit ────────────────────────────────────────────────────────────────

def auto_commit_push(ts: str, hw_egap: float, hw_rounded: int) -> None:
    correct = hw_rounded == TRUE_GAP
    verdict = "CORRECT" if correct else "MISS"
    msg = (
        f"Prime 239 generalization {ts} — "
        f"E[gap]={hw_egap:.4f}, predicted={LAST_KNOWN_PRIME + hw_rounded}, "
        f"truth={TRUE_NEXT_PRIME}, {verdict}"
    )
    subprocess.run(["git", "add", f"output/prime/{ts}/"], check=True, cwd=REPO_ROOT)
    result = subprocess.run(
        ["git", "commit", "-m", msg],
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

    print("=" * 65)
    print("  Prime 239 Generalization Test")
    print(f"  Primes: 51  |  Gaps: 50  |  Windows: {len(WINDOWS)}")
    print(f"  Final window: {WINDOWS[-1]}  local_max={max(WINDOWS[-1])}")
    print(f"  Target: gap={TRUE_GAP}  prime={TRUE_NEXT_PRIME}")
    print(f"  Run: {ts}")
    print("=" * 65)

    # Connect to Kingston once — need it for noise model AND hardware submission
    print("\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    print(f"  Backend: {backend.name}  status: {backend.status().status_msg}")

    print("\nBuilding Kingston noise model for sim chain...")
    noise_model = NoiseModel.from_backend(backend)

    print(f"\nRunning {len(WINDOWS) - 1} windows on noisy AerSimulator...")
    final_offsets, sim_window_info = run_sim_chain(len(WINDOWS) - 1, noise_model)
    mean_sim_mi = sum(w["mi"] for w in sim_window_info) / len(sim_window_info)
    print(f"\n  Sim chain complete — mean MI={mean_sim_mi:.4f}  final offsets={[round(o,4) for o in final_offsets]}")

    print(f"\nSubmitting final window to {BACKEND_NAME}...")
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=42,
    )
    hw_counts, hw_job_id, hw_local_max = run_hw_window(backend, pm, final_offsets)

    bits = counts_to_bits(hw_counts, WINDOW_SIZE)
    hw_mi = mi_halves(bits, [0, 1], [2, 3])
    hw_egap, hw_rounded, hw_mode, hw_mode_gap = decode_prediction(hw_counts, hw_local_max)
    hw_correct = hw_rounded == TRUE_GAP

    print()
    print("── Results ──────────────────────────────────────────────────")
    print(f"  Final window:   {WINDOWS[-1]}  local_max={hw_local_max}")
    print(f"  Weighted E[gap]: {hw_egap:.4f}  →  {hw_rounded}")
    print(f"  Mode:           |{hw_mode}⟩  →  gap≈{hw_mode_gap}")
    print(f"  Predicted prime: {LAST_KNOWN_PRIME + hw_rounded}")
    print(f"  Ground truth:   gap={TRUE_GAP}  prime={TRUE_NEXT_PRIME}")
    print(f"  Error:          {abs(hw_egap - TRUE_GAP):.4f}")
    print(f"  Verdict:        {'✓ CORRECT' if hw_correct else '✗ MISS'}")
    print(f"  Hardware MI:    {hw_mi:.4f} bits")
    print("─" * 65)

    print("\nClassical baselines...")
    baselines = classical_baselines()
    for b in baselines:
        correct = b.rounded == TRUE_GAP
        print(f"  {b.name:<26}  raw={b.raw:.4f}  rounded={b.rounded}  error={b.error:.4f}  {'✓' if correct else '✗'}")

    print("\nGenerating output files...")
    plot_results(hw_egap, baselines, sim_window_info, ts, out_dir)
    print("  → prime239_generalization.png")
    write_md(
        hw_egap, hw_rounded, hw_mode, hw_mode_gap,
        hw_mi, hw_job_id, hw_local_max,
        baselines, sim_window_info, ts, out_dir,
    )
    print("  → prime239_generalization.md")
    save_json(hw_egap, hw_rounded, hw_mi, hw_job_id, baselines, sim_window_info, ts, out_dir)
    print("  → prime239_results.json")

    auto_commit_push(ts, hw_egap, hw_rounded)
    print()


if __name__ == "__main__":
    main()
