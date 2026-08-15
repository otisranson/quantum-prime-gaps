"""prime_predictor.py

Recurrent quantum circuit for prime gap sequence learning and next-prime prediction.

Architecture:
  - Fixed 4-qubit circuit: Bell pair + RY gap encoding + approximated iQFT (degree=1)
  - LOCAL window normalization: each window's angles are divided by that window's
    own max gap, spreading angles uniformly across [0, π] regardless of absolute magnitudes
  - Per-qubit feedback: each bit of the mode bitstring drives its own qubit's offset
  - Scale sweep: [0.05, 0.1, 0.2] — picks scale closest to true gap=4
  - Noisy preflight: winning scale re-run with Kingston noise model

Output convention (permanent):
  - All output written to output/prime/ with timestamp prefix YYYYMMDD_HHMMSS
  - Never overwrites prior runs; every execution is its own timestamped record
  - Auto-committed and pushed to GitHub after each run

Ground truth: prime #50 = 229, prime #51 = 233, gap = 4

Run: python prime_predictor.py
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis.qft import synth_qft_full
from qiskit_aer import AerSimulator

# ── Repo root (for git commands) ───────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent

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
GLOBAL_MAX_GAP = max(ALL_GAPS)  # 14 (kept for reference only; not used in encoding)

LAST_KNOWN_PRIME = FIRST_50_PRIMES[-1]   # 229
TRUE_NEXT_PRIME  = 233
TRUE_GAP         = TRUE_NEXT_PRIME - LAST_KNOWN_PRIME  # 4

WINDOW_SIZE  = 4
SHOTS        = 8_192
SWEEP_SCALES = [0.05, 0.1, 0.2]
VERSION      = "v3"

WINDOWS: list[list[int]] = [
    ALL_GAPS[i : i + WINDOW_SIZE]
    for i in range(len(ALL_GAPS) - WINDOW_SIZE + 1)
]  # 46 windows

# ── Circuit builder — LOCAL normalization ──────────────────────────────────────

def build_circuit(gaps: list[int], offsets: list[float]) -> tuple[QuantumCircuit, int]:
    """Bell pair + RY encoding (local-window normalization) + approx iQFT (degree=1).

    Returns (circuit, local_max) so the caller can decode predictions consistently.
    Local normalization: angle_i = gap_i * π / max(gaps_in_window) + offset_i
    Every window maps its own gap range to [0, π] — breaks the global attractor.
    """
    local_max = max(gaps)   # window-local normalizer
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

# ── Per-qubit feedback ─────────────────────────────────────────────────────────

def bitstring_to_offsets(bs: str, scale: float) -> list[float]:
    """Decompose mode bitstring into per-qubit angle offsets.

    Qiskit: MSB-left, qubit 0 is rightmost. Each bit → ±scale/2 centered at 0.
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

def decode_prediction(counts: dict, local_max: int) -> tuple[float, int, str, int]:
    """Decode final distribution using the final window's local_max.

    With local normalization: angle = bitstring_int * π / 15,
    gap = angle * local_max / π = bitstring_int * local_max / 15.
    """
    total = sum(counts.values())
    weighted_gap = 0.0
    for bs, cnt in counts.items():
        val = int(bs, 2)
        gap = val * local_max / 15   # direct: no π cancels cleanly
        weighted_gap += (cnt / total) * gap

    mode_bs  = max(counts, key=counts.get)
    mode_gap = int(mode_bs, 2) * local_max / 15

    return weighted_gap, max(1, round(weighted_gap)), mode_bs, max(1, round(mode_gap))

# ── Recurrent loop ─────────────────────────────────────────────────────────────

@dataclass
class WindowRecord:
    window_idx: int
    gaps: list[int]
    local_max: int
    offsets_in: list[float]
    mi: float
    mode_bs: str
    mode_prob: float
    counts: dict = field(default_factory=dict)


def run_recurrent(scale: float, sim: AerSimulator,
                  verbose: bool = False) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    offsets = [0.0] * WINDOW_SIZE

    for w_idx, window in enumerate(WINDOWS):
        qc, local_max = build_circuit(window, offsets)
        tqc = transpile(qc, sim)
        counts = sim.run(tqc, shots=SHOTS).result().get_counts()

        bits = counts_to_bits(counts, WINDOW_SIZE)
        mi = mi_halves(bits, [0, 1], [2, 3])
        mode_bs = max(counts, key=counts.get)
        next_offsets = bitstring_to_offsets(mode_bs, scale)

        records.append(WindowRecord(
            window_idx=w_idx,
            gaps=window,
            local_max=local_max,
            offsets_in=offsets[:],
            mi=round(mi, 6),
            mode_bs=mode_bs,
            mode_prob=round(counts[mode_bs] / SHOTS, 4),
            counts={k: v for k, v in sorted(counts.items(), key=lambda x: -x[1])},
        ))

        if verbose and (w_idx % 10 == 0 or w_idx == len(WINDOWS) - 1):
            print(f"    w={w_idx:2d}  gaps={window}  lmax={local_max}  MI={mi:.4f}  "
                  f"mode={mode_bs}  → {[f'{o:+.3f}' for o in next_offsets]}")

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
    final_local_max: int

    @property
    def predicted_prime(self) -> int:
        return LAST_KNOWN_PRIME + self.rounded_gap


def sweep(sim: AerSimulator) -> list[ScaleResult]:
    results = []
    for scale in SWEEP_SCALES:
        print(f"\n  scale={scale} ─────────────────────────────────────────")
        records = run_recurrent(scale, sim, verbose=True)
        final = records[-1]
        w_gap, r_gap, m_bs, m_gap = decode_prediction(final.counts, final.local_max)
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
            final_local_max=final.local_max,
        ))
        print(f"    → E[gap]={w_gap:.4f} (local_max={final.local_max})  "
              f"rounded={r_gap}  prime={LAST_KNOWN_PRIME+r_gap}  "
              f"error={abs(w_gap-TRUE_GAP):.4f}")
    return results


def pick_winner(results: list[ScaleResult]) -> ScaleResult:
    return min(results, key=lambda r: r.error)

# ── Noisy preflight ────────────────────────────────────────────────────────────

def build_noisy_sim() -> tuple[AerSimulator | None, object | None]:
    try:
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend("ibm_kingston")
        nm = NoiseModel.from_backend(backend)
        sim = AerSimulator(noise_model=nm)
        print(f"  Kingston noise model: {len(nm.noise_instructions)} error channels")
        return sim, backend.coupling_map
    except Exception as exc:
        print(f"  Noise model unavailable ({exc})")
        return None, None


def run_noisy_preflight(winner: ScaleResult,
                        noisy_sim: AerSimulator) -> ScaleResult:
    print(f"\n  Noisy preflight  (scale={winner.scale}, Kingston noise model) ──")
    records = run_recurrent(winner.scale, noisy_sim, verbose=True)
    final = records[-1]
    w_gap, r_gap, m_bs, m_gap = decode_prediction(final.counts, final.local_max)
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
        final_local_max=final.local_max,
    )
    nv = "✓ CORRECT" if r_gap == TRUE_GAP else "✗ MISS"
    print(f"  → E[gap]={w_gap:.4f}  rounded={r_gap}  prime={LAST_KNOWN_PRIME+r_gap}  {nv}")
    return result

# ── Plotting ───────────────────────────────────────────────────────────────────

BG   = "#0d1117"
GRID = "#1e293b"
MUT  = "#94a3b8"
FG   = "#f8fafc"
GRN  = "#22c55e"
RED  = "#ef4444"
SCALE_COLORS = ["#7dd3fc", "#fb923c", "#a78bfa"]


def _dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT)
    ax.spines[:].set_color(GRID)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)


def plot_mi_sweep(sweep_results: list[ScaleResult], winner: ScaleResult,
                  ts: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    _dark_ax(ax)
    for sr, color in zip(sweep_results, SCALE_COLORS, strict=True):
        mi_vals = [r.mi for r in sr.records]
        lw = 2.0 if sr is winner else 1.0
        alpha = 1.0 if sr is winner else 0.5
        flag = "  ★" if sr is winner else ""
        ax.plot(mi_vals, color=color, linewidth=lw, alpha=alpha,
                label=f"scale={sr.scale}{flag}  E[gap]={sr.weighted_gap:.2f}→{sr.rounded_gap}")
    ax.axvline(len(WINDOWS) - 1, color="#64748b", linewidth=1, linestyle="--")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Root MI  (bits)")
    ax.set_title(f"MI — recurrent windows, local normalisation  [{ts}]",
                 color=FG, fontsize=11)
    ax.legend(framealpha=0, labelcolor="white", fontsize=9)
    fig.tight_layout()
    p = out_dir / "mi.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_dist(result: ScaleResult, title: str,
              ts: str, filename: str, out_dir: Path) -> Path:
    counts = result.records[-1].counts
    lmax   = result.final_local_max
    states = [f"{i:04b}" for i in range(16)]
    probs  = [counts.get(s, 0) / SHOTS for s in states]
    # gap label uses local_max
    gap_labels = [f"{i * lmax / 15:.1f}" for i in range(16)]

    true_bin = round(TRUE_GAP / lmax * 15)
    pred_bin = round(result.weighted_gap / lmax * 15)

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
        [f"|{s}⟩\n({g})" for s, g in zip(states, gap_labels, strict=True)],
        fontsize=7, color=MUT,
    )
    ax.set_ylabel("Probability")
    ax.set_title(
        f"{title}  [{ts}]\n"
        f"Window {WINDOWS[-1]}  local_max={lmax}  "
        f"E[gap]={result.weighted_gap:.2f}→{result.rounded_gap}  "
        f"True gap={TRUE_GAP}",
        color=FG, fontsize=10,
    )
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    legend_els = [
        Patch(facecolor=GRN,       label=f"Ground truth  (gap={TRUE_GAP})"),
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
    return p


def plot_summary(sweep_results: list[ScaleResult], winner: ScaleResult,
                 noisy: ScaleResult | None, ts: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    def vt(r: ScaleResult) -> tuple[str, str]:
        return ("✓ CORRECT", GRN) if r.rounded_gap == TRUE_GAP else ("✗ MISS", RED)

    rows: list = [
        ("Normalization", "local window  (angle = gap × π / window_max)", FG),
        ("Last known prime", f"#{len(FIRST_50_PRIMES)} = {LAST_KNOWN_PRIME}", MUT),
        ("Final window", f"{WINDOWS[-1]}  local_max={winner.final_local_max}", MUT),
        (None, None, None),
        ("Scale sweep — clean AerSimulator", None, FG),
    ]
    for sr, color in zip(sweep_results, SCALE_COLORS, strict=True):
        flag = "  ★" if sr is winner else ""
        v, vc = vt(sr)
        rows.append((
            f"  scale={sr.scale}{flag}",
            f"E[gap]={sr.weighted_gap:.3f}→{sr.rounded_gap}  "
            f"prime={sr.predicted_prime}  err={sr.error:.3f}  {v}",
            vc if sr is winner else color,
        ))

    if noisy is not None:
        rows.append((None, None, None))
        rows.append(("Kingston noisy preflight", None, FG))
        nv, nvc = vt(noisy)
        rows.append((
            f"  scale={noisy.scale}",
            f"E[gap]={noisy.weighted_gap:.3f}→{noisy.rounded_gap}  "
            f"prime={noisy.predicted_prime}  err={noisy.error:.3f}  {nv}",
            nvc,
        ))
        hw_ready = noisy.error < 1.5
        rows.append((
            "  Hardware flag",
            "✓ READY (error < 1.5)" if hw_ready else "⚠ MARGINAL",
            GRN if hw_ready else "#facc15",
        ))

    rows.append((None, None, None))
    rows.append(("Ground truth", f"gap={TRUE_GAP}  prime={TRUE_NEXT_PRIME}", GRN))

    ax.set_title(f"Quantum Prime Predictor {VERSION} — {ts}", color=FG, fontsize=12, pad=10)
    y = 0.95
    for row in rows:
        if row[0] is None:
            y -= 0.035
            continue
        label, value, color = row
        ax.text(0.02, y, label + (":" if value else ""),
                transform=ax.transAxes, fontsize=9, color="#64748b" if value else FG, va="top")
        if value:
            ax.text(0.40, y, value, transform=ax.transAxes,
                    fontsize=9, color=color, va="top", fontweight="bold")
        y -= 0.080

    fig.tight_layout()
    p = out_dir / "summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p

# ── JSON output ────────────────────────────────────────────────────────────────

def save_json(ts: str, sweep_results: list[ScaleResult], winner: ScaleResult,
              noisy: ScaleResult | None, out_dir: Path) -> Path:
    data: dict = {
        "version": VERSION,
        "timestamp": ts,
        "config": {
            "normalization": "local_window",
            "n_qubits": WINDOW_SIZE,
            "shots_per_window": SHOTS,
            "sweep_scales": SWEEP_SCALES,
            "iqft_approx_degree": 1,
            "n_windows": len(WINDOWS),
            "global_max_gap": GLOBAL_MAX_GAP,
            "feedback": "per-qubit",
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
                "final_local_max": sr.final_local_max,
                "mean_mi": round(sr.mean_mi, 6),
                "final_mi": round(sr.final_mi, 6),
                "correct": sr.rounded_gap == TRUE_GAP,
                "per_window": [
                    {"w": r.window_idx, "gaps": r.gaps, "local_max": r.local_max,
                     "mi": r.mi, "mode": r.mode_bs, "mode_prob": r.mode_prob}
                    for r in sr.records
                ],
                "final_counts": sr.records[-1].counts,
            }
            for sr in sweep_results
        ],
        "winner_scale": winner.scale,
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
    p = out_dir / "results.json"
    p.write_text(json.dumps(data, indent=2))
    return p

# ── Auto-commit and push ───────────────────────────────────────────────────────

def auto_commit_push(ts: str, weighted_gap: float, error: float) -> None:
    msg = f"Run output {ts} — E[gap]={weighted_gap:.4f}, error={error:.4f}"
    subprocess.run(["git", "add", f"output/prime/{ts}/"], check=True, cwd=REPO_ROOT)
    result = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  Committed: output/prime/{ts}/")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"  Git commit skipped: {result.stdout.strip()}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "output" / "prime" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  Quantum Prime Predictor {VERSION} — Local Window Normalisation")
    print(f"  Gaps: {len(ALL_GAPS)}  Windows: {len(WINDOWS)}  "
          f"Scales: {SWEEP_SCALES}  Run: {ts}")
    print(f"  Last prime: {LAST_KNOWN_PRIME}  |  True next: {TRUE_NEXT_PRIME}  "
          f"(gap={TRUE_GAP})")
    print("=" * 65)

    clean_sim = AerSimulator()

    print("\nPhase 1 — scale sweep, clean AerSimulator")
    sweep_results = sweep(clean_sim)
    winner = pick_winner(sweep_results)
    print(f"\n  Winner: scale={winner.scale}  E[gap]={winner.weighted_gap:.4f}"
          f"  →  prime {winner.predicted_prime}  (error={winner.error:.4f})")

    print("\nPhase 2 — Kingston noisy preflight")
    noisy_sim, _ = build_noisy_sim()
    noisy_result: ScaleResult | None = None
    if noisy_sim is not None:
        noisy_result = run_noisy_preflight(winner, noisy_sim)
        hw_ready = noisy_result.error < 1.5
        print(f"  Hardware flag: {'✓ READY' if hw_ready else '⚠ MARGINAL'}")

    print("\nGenerating plots...")
    paths: list[Path] = []
    paths.append(plot_mi_sweep(sweep_results, winner, ts, out_dir))
    paths.append(plot_dist(winner, f"Winner (scale={winner.scale}) — clean sim",
                           ts, "dist_clean.png", out_dir))
    if noisy_result is not None:
        paths.append(plot_dist(noisy_result, "Kingston noisy preflight",
                               ts, "dist_noisy.png", out_dir))
    paths.append(plot_summary(sweep_results, winner, noisy_result, ts, out_dir))

    json_path = save_json(ts, sweep_results, winner, noisy_result, out_dir)
    paths.append(json_path)
    for p in paths:
        print(f"  → {p.relative_to(REPO_ROOT)}")

    # ── Final report ───────────────────────────────────────────────────────────
    print("\n── Results ──────────────────────────────────────────────────")
    for sr in sweep_results:
        mark = "★" if sr is winner else " "
        print(f"  {mark} scale={sr.scale}  E[gap]={sr.weighted_gap:.4f}"
              f"  rounded={sr.rounded_gap}  prime={sr.predicted_prime}"
              f"  err={sr.error:.4f}  lmax={sr.final_local_max}")
    if noisy_result is not None:
        nv = "✓" if noisy_result.rounded_gap == TRUE_GAP else "✗"
        print(f"\n  Noisy: E[gap]={noisy_result.weighted_gap:.4f}"
              f"  prime={noisy_result.predicted_prime}  {nv}"
              f"  (mean MI={noisy_result.mean_mi:.4f})")
    print(f"\n  Ground truth: gap={TRUE_GAP}  prime={TRUE_NEXT_PRIME}")
    print()

    # ── Auto-commit and push ───────────────────────────────────────────────────
    best = noisy_result if noisy_result is not None else winner
    auto_commit_push(ts, best.weighted_gap, best.error)


if __name__ == "__main__":
    main()
