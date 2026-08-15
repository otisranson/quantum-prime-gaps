"""prime_predictor_hw.py

Hardware validation of the v3 recurrent prime predictor on ibm_kingston.

Runs all 46 windows sequentially with real hardware feedback — each window's
measurement result feeds per-qubit angle offsets into the next window's RY
encoding, exactly as in the AerSimulator v3 run.

Architecture: Bell pair + RY (local window normalization) + approx iQFT (degree=1)
Backend:      ibm_kingston, optimization_level=3, seed_transpiler=42, 8192 shots/window
Expected:     E[gap]=4.006 → prime 233 (v3 sim result, all scales correct)

Output → output/prime/{YYYYMMDD_HHMMSS}/:
  results_hw.md       full markdown report
  mi_hw.png           MI over 46 windows
  dist_hw.png         final window distribution
  summary_hw.png      prediction scorecard
  results_hw.json     machine-readable results
  checkpoint.json     per-window checkpoint (deleted on success)

Timing: 46 sequential jobs — expect 2–8 hours depending on Kingston queue.
Checkpoints after every window so no data is lost on interruption.

Run: python prime_predictor_hw.py
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch
from qiskit import QuantumCircuit
from qiskit.synthesis.qft import synth_qft_full
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

# ── Constants ──────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent
BACKEND_NAME = "ibm_kingston"
SCALE        = 0.05     # best from v3 clean sim (error=0.006)
SHOTS        = 8_192
VERSION      = "v3-hw"

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
]
GLOBAL_MAX_GAP   = max(ALL_GAPS)   # 14
LAST_KNOWN_PRIME = FIRST_50_PRIMES[-1]   # 229
TRUE_NEXT_PRIME  = 233
TRUE_GAP         = 4

WINDOW_SIZE = 4
WINDOWS: list[list[int]] = [
    ALL_GAPS[i : i + WINDOW_SIZE]
    for i in range(len(ALL_GAPS) - WINDOW_SIZE + 1)
]  # 46 windows

# ── Circuit builder — local window normalization ───────────────────────────────

def build_circuit(gaps: list[int], offsets: list[float]) -> tuple[QuantumCircuit, int]:
    """Bell pair + RY (local-window normalization) + approx iQFT degree=1."""
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

# ── Per-qubit feedback ─────────────────────────────────────────────────────────

def bitstring_to_offsets(bs: str, scale: float) -> list[float]:
    n = len(bs)
    return [(int(bs[n - 1 - q]) - 0.5) * scale for q in range(n)]

# ── Hardware job submission ────────────────────────────────────────────────────

def submit_and_wait(qc: QuantumCircuit, pm, sampler: Sampler) -> tuple[dict, str]:
    """Transpile, submit to hardware, wait for result. Returns (counts, job_id)."""
    isa_qc = pm.run(qc)
    job = sampler.run([isa_qc], shots=SHOTS)
    job_id = job.job_id()
    print(f"    Job {job_id} submitted — waiting...", flush=True)
    t0 = time.time()
    result = job.result()
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s", flush=True)

    pub_result = result[0]
    bitarray = pub_result.data.c
    counts: dict[str, int] = {}
    for bs in bitarray.get_bitstrings():
        counts[bs] = counts.get(bs, 0) + 1
    return counts, job_id

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
    total = sum(counts.values())
    weighted_gap = 0.0
    for bs, cnt in counts.items():
        val = int(bs, 2)
        gap = val * local_max / 15
        weighted_gap += (cnt / total) * gap
    mode_bs  = max(counts, key=counts.get)
    mode_gap = int(mode_bs, 2) * local_max / 15
    return weighted_gap, max(1, round(weighted_gap)), mode_bs, max(1, round(mode_gap))

# ── Window record ──────────────────────────────────────────────────────────────

@dataclass
class WindowRecord:
    window_idx: int
    gaps: list[int]
    local_max: int
    offsets_in: list[float]
    mi: float
    mode_bs: str
    mode_prob: float
    job_id: str
    counts: dict = field(default_factory=dict)

# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(out_dir: Path, ts: str, records: list[WindowRecord],
                    next_offsets: list[float]) -> None:
    data = {
        "ts": ts,
        "scale": SCALE,
        "completed": len(records),
        "next_offsets": next_offsets,
        "records": [
            {k: v for k, v in asdict(r).items() if k != "counts"}
            | {"job_id": r.job_id, "top_counts": dict(list(r.counts.items())[:8])}
            for r in records
        ],
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(data, indent=2))

# ── Hardware recurrent loop ────────────────────────────────────────────────────

def run_hw_recurrent(backend, pm, out_dir: Path, ts: str) -> list[WindowRecord]:
    sampler = Sampler(mode=backend)
    records: list[WindowRecord] = []
    offsets = [0.0] * WINDOW_SIZE
    t_start = time.time()

    for w_idx, window in enumerate(WINDOWS):
        elapsed_total = time.time() - t_start
        avg_per_window = elapsed_total / max(w_idx, 1)
        remaining = avg_per_window * (len(WINDOWS) - w_idx)
        eta = f"~{remaining/60:.0f}min remaining" if w_idx > 0 else "estimating..."

        print(f"\n  Window {w_idx+1}/{len(WINDOWS)}  gaps={window}  "
              f"lmax={max(window)}  {eta}", flush=True)

        qc, local_max = build_circuit(window, offsets)
        counts, job_id = submit_and_wait(qc, pm, sampler)

        bits = counts_to_bits(counts, WINDOW_SIZE)
        mi = mi_halves(bits, [0, 1], [2, 3])
        mode_bs = max(counts, key=counts.get)
        next_offsets = bitstring_to_offsets(mode_bs, SCALE)

        rec = WindowRecord(
            window_idx=w_idx,
            gaps=window,
            local_max=local_max,
            offsets_in=offsets[:],
            mi=round(mi, 6),
            mode_bs=mode_bs,
            mode_prob=round(counts[mode_bs] / SHOTS, 4),
            job_id=job_id,
            counts={k: v for k, v in sorted(counts.items(), key=lambda x: -x[1])},
        )
        records.append(rec)
        save_checkpoint(out_dir, ts, records, next_offsets)

        top = list(rec.counts.items())[:3]
        top_str = "  ".join(f"|{bs}⟩={cnt}" for bs, cnt in top)
        print(f"    MI={mi:.4f}  mode={mode_bs}  {top_str}", flush=True)

        offsets = next_offsets

    return records

# ── Plotting ───────────────────────────────────────────────────────────────────

BG   = "#0d1117"
GRID = "#1e293b"
MUT  = "#94a3b8"
FG   = "#f8fafc"
GRN  = "#22c55e"
RED  = "#ef4444"
ORG  = "#fb923c"


def _dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT)
    ax.spines[:].set_color(GRID)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)


def plot_mi(records: list[WindowRecord], ts: str, out_dir: Path) -> Path:
    mi_vals = [r.mi for r in records]
    x = range(len(records))
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    _dark_ax(ax)
    BLUE = "#7dd3fc"
    ax.plot(x, mi_vals, color=BLUE, linewidth=1.8, zorder=3)
    ax.fill_between(x, mi_vals, alpha=0.15, color=BLUE, zorder=2)
    ax.axvline(len(records) - 1, color=ORG, linewidth=1.2,
               linestyle="--", label="Prediction window")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Root MI  (bits)")
    ax.set_title(f"MI — {BACKEND_NAME} hardware, 46 recurrent windows  [{ts}]",
                 color=FG, fontsize=11)
    ax.legend(framealpha=0, labelcolor=ORG, fontsize=9)
    fig.tight_layout()
    p = out_dir / "mi_hw.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_dist(records: list[WindowRecord], weighted_gap: float,
              rounded_gap: int, ts: str, out_dir: Path) -> Path:
    final = records[-1]
    lmax  = final.local_max
    states = [f"{i:04b}" for i in range(16)]
    probs  = [final.counts.get(s, 0) / SHOTS for s in states]
    gap_labels = [f"{i * lmax / 15:.1f}" for i in range(16)]

    true_bin = round(TRUE_GAP / lmax * 15)
    pred_bin = round(weighted_gap / lmax * 15)
    bar_colors = [
        GRN if i == true_bin else ORG if i == pred_bin else "#334155"
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
        f"Final window — {BACKEND_NAME} hardware  [{ts}]\n"
        f"Window {WINDOWS[-1]}  local_max={lmax}  "
        f"E[gap]={weighted_gap:.3f}→{rounded_gap}  True gap={TRUE_GAP}",
        color=FG, fontsize=10,
    )
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    legend_els = [
        Patch(facecolor=GRN, label=f"Ground truth  (gap={TRUE_GAP})"),
        Patch(facecolor=ORG, label=f"Weighted prediction ({weighted_gap:.3f}→{rounded_gap})"),
        Patch(facecolor="#334155", label="Other states"),
    ]
    ax.legend(handles=legend_els, framealpha=0, labelcolor="white", fontsize=9)
    for bar, prob in zip(bars, probs, strict=True):
        if prob > 0.04:
            ax.text(bar.get_x() + bar.get_width() / 2, prob + 0.003,
                    f"{prob:.1%}", ha="center", va="bottom", fontsize=7, color=FG)
    fig.tight_layout()
    p = out_dir / "dist_hw.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_summary(records: list[WindowRecord], weighted_gap: float,
                 rounded_gap: int, mode_bs: str, mode_gap: int,
                 ts: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    correct = (rounded_gap == TRUE_GAP)
    mean_mi = sum(r.mi for r in records) / len(records)
    final_mi = records[-1].mi

    rows = [
        ("Backend", BACKEND_NAME, MUT),
        ("Shots per window", f"{SHOTS:,}", MUT),
        ("Windows run", f"{len(records)} / 46", MUT),
        ("Feedback scale", str(SCALE), MUT),
        ("Normalization", "local window", MUT),
        (None, None, None),
        ("Mean hardware MI", f"{mean_mi:.4f} bits", "#7dd3fc"),
        ("Final window MI", f"{final_mi:.4f} bits", "#7dd3fc"),
        (None, None, None),
        ("Weighted E[gap]", f"{weighted_gap:.4f}  →  gap={rounded_gap}", ORG),
        ("Mode prediction", f"|{mode_bs}⟩  →  gap≈{mode_gap}", "#a78bfa"),
        ("Predicted prime",
         f"{LAST_KNOWN_PRIME} + {rounded_gap} = {LAST_KNOWN_PRIME + rounded_gap}", ORG),
        (None, None, None),
        ("Ground truth", f"gap={TRUE_GAP}  prime={TRUE_NEXT_PRIME}", GRN),
        ("Verdict",
         "✓ CORRECT" if correct else "✗ MISS",
         GRN if correct else RED),
    ]

    ax.set_title(f"Quantum Prime Predictor — {BACKEND_NAME} Hardware  [{ts}]",
                 color=FG, fontsize=12, pad=10)
    y = 0.96
    for row in rows:
        if row[0] is None:
            y -= 0.035
            continue
        label, value, color = row
        ax.text(0.02, y, f"{label}:", transform=ax.transAxes,
                fontsize=10, color="#64748b", va="top")
        ax.text(0.48, y, value, transform=ax.transAxes,
                fontsize=10, color=color, va="top", fontweight="bold")
        y -= 0.082

    fig.tight_layout()
    p = out_dir / "summary_hw.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p

# ── Markdown report ────────────────────────────────────────────────────────────

def write_md(records: list[WindowRecord], weighted_gap: float,
             rounded_gap: int, ts: str, out_dir: Path) -> Path:
    correct = (rounded_gap == TRUE_GAP)
    mean_mi = sum(r.mi for r in records) / len(records)
    lines = [
        f"# Quantum Prime Predictor — {BACKEND_NAME} Hardware Run",
        "",
        f"**Date:** {ts[:4]}-{ts[4:6]}-{ts[6:8]}  ",
        f"**Backend:** {BACKEND_NAME}  ",
        f"**Shots/window:** {SHOTS:,}  ",
        f"**Windows:** {len(records)}/46  ",
        f"**Scale:** {SCALE}  ",
        "**Normalization:** local window  ",
        "",
        "---",
        "",
        "## Prediction",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Weighted E[gap] | {weighted_gap:.4f} |",
        f"| Predicted gap | {rounded_gap} |",
        f"| Predicted prime | {LAST_KNOWN_PRIME + rounded_gap} |",
        f"| Ground truth gap | {TRUE_GAP} |",
        f"| Ground truth prime | {TRUE_NEXT_PRIME} |",
        f"| Error | {abs(weighted_gap - TRUE_GAP):.4f} |",
        f"| Verdict | {'✓ CORRECT' if correct else '✗ MISS'} |",
        f"| Mean hardware MI | {mean_mi:.4f} bits |",
        f"| Final window MI | {records[-1].mi:.4f} bits |",
        "",
        "---",
        "",
        "## Per-Window Results",
        "",
        "| # | Gaps | lmax | Offsets | MI | Mode | Mode% | Job ID |",
        "|---|------|------|---------|----|------|-------|--------|",
    ]
    for r in records:
        off_str = "[" + ",".join(f"{o:+.3f}" for o in r.offsets_in) + "]"
        lines.append(
            f"| {r.window_idx} | {r.gaps} | {r.local_max} | {off_str} "
            f"| {r.mi:.4f} | `{r.mode_bs}` | {r.mode_prob:.1%} | `{r.job_id}` |"
        )
    lines += [
        "",
        "---",
        "",
        "## Final Window Distribution",
        "",
        f"Window: {WINDOWS[-1]}  local_max={records[-1].local_max}",
        "",
        "| State | Count | % |",
        "|-------|------:|--:|",
    ]
    total = sum(records[-1].counts.values())
    for bs, cnt in list(records[-1].counts.items())[:8]:
        lines.append(f"| `{bs}` | {cnt} | {cnt/total:.1%} |")
    lines += [
        "",
        "---",
        "",
        f"*Run timestamp: {ts}*",
    ]
    p = out_dir / "results_hw.md"
    p.write_text("\n".join(lines))
    return p

# ── JSON output ────────────────────────────────────────────────────────────────

def save_json(records: list[WindowRecord], weighted_gap: float,
              rounded_gap: int, ts: str, out_dir: Path) -> Path:
    data = {
        "version": VERSION,
        "timestamp": ts,
        "backend": BACKEND_NAME,
        "config": {
            "scale": SCALE,
            "shots": SHOTS,
            "normalization": "local_window",
            "iqft_approx_degree": 1,
            "n_windows": len(records),
        },
        "ground_truth": {
            "last_known_prime": LAST_KNOWN_PRIME,
            "true_next_prime": TRUE_NEXT_PRIME,
            "true_gap": TRUE_GAP,
        },
        "prediction": {
            "weighted_gap": round(weighted_gap, 6),
            "rounded_gap": rounded_gap,
            "predicted_prime": LAST_KNOWN_PRIME + rounded_gap,
            "error": round(abs(weighted_gap - TRUE_GAP), 4),
            "correct": rounded_gap == TRUE_GAP,
        },
        "mean_mi": round(sum(r.mi for r in records) / len(records), 6),
        "per_window": [
            {
                "w": r.window_idx, "gaps": r.gaps, "local_max": r.local_max,
                "offsets_in": [round(o, 4) for o in r.offsets_in],
                "mi": r.mi, "mode": r.mode_bs, "mode_prob": r.mode_prob,
                "job_id": r.job_id,
                "top_counts": dict(list(r.counts.items())[:8]),
            }
            for r in records
        ],
        "final_counts": records[-1].counts,
    }
    p = out_dir / "results_hw.json"
    p.write_text(json.dumps(data, indent=2))
    return p

# ── Auto-commit and push ───────────────────────────────────────────────────────

def auto_commit_push(ts: str, weighted_gap: float, error: float) -> None:
    predicted = LAST_KNOWN_PRIME + max(1, round(weighted_gap))
    msg = (f"Kingston hardware run {ts} — "
           f"E[gap]={weighted_gap:.4f}, predicted={predicted}, error={error:.4f}")
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
        print(f"  Commit skipped: {result.stdout.strip()}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "output" / "prime" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  Quantum Prime Predictor {VERSION} — {BACKEND_NAME} hardware")
    print(f"  Windows: {len(WINDOWS)}  Scale: {SCALE}  Shots/window: {SHOTS:,}")
    print(f"  Run: {ts}")
    print("  Expect 2–8 hours (46 sequential jobs + queue)")
    print(f"  Checkpointing to output/prime/{ts}/checkpoint.json")
    print("=" * 65)

    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    print(f"\n  Backend: {backend.name}  status: {backend.status().status_msg}")

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=42,
    )

    print(f"\nRunning {len(WINDOWS)} hardware windows...")
    records = run_hw_recurrent(backend, pm, out_dir, ts)

    final = records[-1]
    weighted_gap, rounded_gap, mode_bs, mode_gap = decode_prediction(
        final.counts, final.local_max
    )
    correct = (rounded_gap == TRUE_GAP)

    print("\n── Results ──────────────────────────────────────────────────")
    print(f"  Final window gaps:   {WINDOWS[-1]}  local_max={final.local_max}")
    print(f"  Weighted E[gap]:     {weighted_gap:.4f}  →  {rounded_gap}")
    print(f"  Mode prediction:     |{mode_bs}⟩  →  gap≈{mode_gap}")
    print(f"  Predicted prime:     {LAST_KNOWN_PRIME + rounded_gap}")
    print(f"  Ground truth:        gap={TRUE_GAP}  prime={TRUE_NEXT_PRIME}")
    print(f"  Error:               {abs(weighted_gap - TRUE_GAP):.4f}")
    print(f"  Verdict:             {'✓ CORRECT' if correct else '✗ MISS'}")
    print(f"  Mean hardware MI:    {sum(r.mi for r in records)/len(records):.4f} bits")
    print("─" * 65)

    print("\nGenerating output files...")
    plot_mi(records, ts, out_dir)
    print("  → mi_hw.png")
    plot_dist(records, weighted_gap, rounded_gap, ts, out_dir)
    print("  → dist_hw.png")
    plot_summary(records, weighted_gap, rounded_gap, mode_bs, mode_gap, ts, out_dir)
    print("  → summary_hw.png")
    write_md(records, weighted_gap, rounded_gap, ts, out_dir)
    print("  → results_hw.md")
    save_json(records, weighted_gap, rounded_gap, ts, out_dir)
    print("  → results_hw.json")

    # Remove checkpoint now that the full run is saved
    cp = out_dir / "checkpoint.json"
    if cp.exists():
        cp.unlink()

    auto_commit_push(ts, weighted_gap, abs(weighted_gap - TRUE_GAP))
    print()


if __name__ == "__main__":
    main()
