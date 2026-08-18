"""experiments/terrain_20000primes_sim.py

20k-scale extension of terrain_5000primes.py's quantum MI generation --
same recurrent v3 circuit, same windowing, same MI estimator, same period
detection, run against the full 20,000-prime cache. AerSimulator only, no
IBM hardware, no Qiskit Runtime session (this script doesn't import
qiskit_ibm_runtime at all, same as the original).

**Core MI computation logic is unchanged, copied verbatim from
terrain_5000primes.py**: build_circuit, bitstring_to_offsets, counts_to_bits,
_entropy, _mm, mi_halves, run_recurrent's per-window loop, and detect_period
are the same functions, not redesigned. Per this repo's standalone-script
convention (every layer3_*.py, terrain_*.py duplicates rather than imports
from siblings), they're copied here rather than imported from
terrain_5000primes.py.

**Two deliberate differences from the original, both scope choices, not
changes to the MI computation itself:**
  1. **Data source.** terrain_5000primes.py sieves its own primes inline.
     This reads `data/primes_20000.json` instead, per explicit instruction
     to run "against the full 20k-prime cache" -- the same cache
     experiments/gap_entropy_windows.py and layer3_20k_scaleup.py already use.
  2. **No terrain heatmap PNG.** terrain_5000primes.py's plot_terrain()
     renders a spline-upsampled topographic map of the full per-window
     probability distribution -- expensive (order=3 zoom on a
     ~20,000x16 array) and not needed by the downstream changepoint/overlap
     pipeline this run exists for. The 3-panel MI waveform plot (trace +
     ACF + FFT) *is* kept, reused verbatim, since it's cheap and useful for
     sanity-checking the MI series before changepoint detection runs on it.

**Runtime, measured before committing to this run:** a 300-window benchmark
of the exact per-window circuit+transpile+run cost on this machine measured
~91ms/window, extrapolating to ~30 minutes for the full 19,996-window run
(vs. ~7.6 min extrapolated for the original 4,996-window 5k run at the same
rate -- consistent with the ~4x window-count scaling, not a per-window
slowdown). This is well within AerSimulator's practical range for a 4-qubit
circuit at 8192 shots; no memory constraints were hit in the benchmark. If a
real run diverges substantially from this estimate, that itself is worth
flagging (see the periodic elapsed/ETA progress print, kept from the
original, which makes a stall or slowdown visible rather than silent).

Outputs -> output/prime/{YYYYMMDD_HHMMSS}/terrain_20000primes/:
  mi_waveform_20k.png       MI signal + ACF + FFT period analysis
  results_20000primes.json  per-window MI, mode, probability distributions --
                             same schema as results_5000primes.json

Run: python experiments/terrain_20000primes_sim.py
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis.qft import synth_qft_full
from qiskit_aer import AerSimulator
from scipy import signal

REPO_ROOT = Path(__file__).parent.parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"
N_PRIMES = 20_000
WINDOW_SIZE = 4
SHOTS = 8_192
SCALE = 0.05

BG, FG, MUT, GRN, GOLD, PRP, BLU = (
    "#0a0e18", "#e8eaf6", "#7986cb", "#66bb6a", "#ffd60a", "#ce93d8", "#7dd3fc",
)

# ── Circuit (v3, identical to terrain_5000primes.py) ────────────────────────


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


# ── MI (identical to terrain_5000primes.py) ──────────────────────────────────


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


# ── Simulation (identical loop body to terrain_5000primes.py's run_recurrent) ─


def run_recurrent(all_gaps: list[int]) -> list[dict]:
    windows = [all_gaps[i : i + WINDOW_SIZE] for i in range(len(all_gaps) - WINDOW_SIZE + 1)]
    n_windows = len(windows)
    records = []

    sim = AerSimulator()  # AerSimulator only -- no qiskit_ibm_runtime import anywhere in this file
    offsets = [0.0] * WINDOW_SIZE
    t0 = time.time()

    for w_idx, window in enumerate(windows):
        qc, local_max = build_circuit(window, offsets)
        tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=42)
        counts = sim.run(tqc, shots=SHOTS).result().get_counts()

        bits = counts_to_bits(counts, WINDOW_SIZE)
        mi = mi_halves(bits, [0, 1], [2, 3])
        mode_bs = max(counts, key=counts.get)
        offsets = bitstring_to_offsets(mode_bs, SCALE)

        records.append({
            "w": w_idx, "gaps": window, "local_max": local_max,
            "mi": round(mi, 6), "mode": mode_bs,
        })

        if w_idx % 1000 == 0 or w_idx == n_windows - 1:
            elapsed = time.time() - t0
            rate = (w_idx + 1) / elapsed
            eta = (n_windows - w_idx - 1) / rate if rate > 0 else 0
            print(f"  w{w_idx:05d}/{n_windows-1}  gaps={window}  lmax={local_max}  "
                  f"MI={mi:.4f}  mode={mode_bs}  [{elapsed:.0f}s elapsed, ~{eta:.0f}s left]", flush=True)

    return records


# ── Period detection (identical to terrain_5000primes.py) ──────────────────


def detect_period(mi_vals: list[float]) -> dict:
    arr = np.array(mi_vals)
    centered = arr - arr.mean()
    n = len(arr)

    acf_full = np.correlate(centered, centered, mode="full")
    acf = acf_full[n - 1:]
    acf /= acf[0]

    acf_peaks, _ = signal.find_peaks(acf[1:], height=0.05, prominence=0.03, distance=5)
    acf_period = int(acf_peaks[0] + 1) if len(acf_peaks) > 0 else None
    acf_strength = float(acf[acf_period]) if acf_period else None

    fft_coeffs = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n)
    magnitudes = np.abs(fft_coeffs)
    magnitudes[0] = 0
    dominant_idx = int(np.argmax(magnitudes))
    fft_period = round(1 / freqs[dominant_idx]) if freqs[dominant_idx] > 0 else None
    fft_power_fraction = float(magnitudes[dominant_idx] ** 2 / np.sum(magnitudes ** 2))

    top5_idx = np.argsort(magnitudes)[::-1][:5]
    top5 = [
        {"period": round(1 / freqs[i]) if freqs[i] > 0 else None,
         "power_frac": float(magnitudes[i] ** 2 / np.sum(magnitudes ** 2))}
        for i in top5_idx
    ]

    return {
        "acf_dominant_period": acf_period,
        "acf_strength": round(acf_strength, 4) if acf_strength else None,
        "fft_dominant_period": fft_period,
        "fft_dominant_power_fraction": round(fft_power_fraction, 4),
        "fft_top5": top5,
        "acf": acf[:500].tolist(),
    }


# ── MI waveform plot (identical to terrain_5000primes.py) ──────────────────


def plot_mi_waveform(records: list[dict], period_info: dict, ts: str, out_dir: Path) -> Path:
    mi_vals = [r["mi"] for r in records]
    n = len(mi_vals)
    x = np.arange(n)

    period_acf = period_info["acf_dominant_period"]
    period_fft = period_info["fft_dominant_period"]
    acf = np.array(period_info["acf"])

    fig, axes = plt.subplots(3, 1, figsize=(22, 14), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"MI Waveform -- First {N_PRIMES:,} Primes, {n:,} Windows (AerSimulator)  [{ts}]\n"
        "Mutual information between qubit halves [q0,q1] vs [q2,q3]",
        color=FG, fontsize=11, y=0.99,
    )

    ax1 = axes[0]
    ax1.set_facecolor(BG)
    ax1.tick_params(colors=MUT)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e293b")
    ax1.plot(x, mi_vals, color=BLU, linewidth=0.4, alpha=0.7, zorder=3)
    kernel30 = np.ones(30) / 30
    kernel300 = np.ones(300) / 300
    rolling30 = np.convolve(mi_vals, kernel30, mode="same")
    rolling300 = np.convolve(mi_vals, kernel300, mode="same")
    ax1.plot(x, rolling30, color=GOLD, linewidth=0.9, alpha=0.85, label="30-window rolling mean")
    ax1.plot(x, rolling300, color=GRN, linewidth=1.6, alpha=0.9, label="300-window rolling mean (slow trend)")
    if period_fft and period_fft < n // 2:
        for tick in range(0, n, period_fft):
            ax1.axvline(tick, color=PRP, linewidth=0.3, alpha=0.35, linestyle="--")
    ax1.set_xlim(0, n)
    ax1.set_ylabel("MI (bits)", color=MUT, fontsize=9)
    ax1.set_xlabel("Window index", color=MUT, fontsize=9)
    ax1.legend(framealpha=0, labelcolor=FG, fontsize=8)
    ax1.grid(axis="y", color="#1e293b", linewidth=0.5)

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.tick_params(colors=MUT)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e293b")
    lags = np.arange(len(acf))
    ax2.plot(lags, acf, color=BLU, linewidth=1.0)
    ax2.axhline(0, color="#1e293b", linewidth=0.8)
    if period_acf:
        ax2.axvline(period_acf, color=GOLD, linewidth=1.2, alpha=0.8)
        ax2.text(period_acf + 2, 0.5, f"ACF period ~ {period_acf} windows\n(r={period_info['acf_strength']:.3f})",
                  color=GOLD, fontsize=8, va="center")
    ax2.set_xlim(0, len(acf) - 1)
    ax2.set_ylabel("Autocorrelation", color=MUT, fontsize=9)
    ax2.set_xlabel("Lag (windows)", color=MUT, fontsize=9)
    ax2.grid(axis="y", color="#1e293b", linewidth=0.5)

    ax3 = axes[2]
    ax3.set_facecolor(BG)
    ax3.tick_params(colors=MUT)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#1e293b")
    mi_arr = np.array(mi_vals) - np.mean(mi_vals)
    fft_coeffs = np.fft.rfft(mi_arr)
    freqs = np.fft.rfftfreq(n)
    power = np.abs(fft_coeffs) ** 2
    power[0] = 0
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1 / freqs, 0)
    mask = (periods >= 2) & (periods <= n // 3)
    ax3.fill_between(periods[mask], power[mask], color=PRP, alpha=0.45, zorder=2)
    ax3.plot(periods[mask], power[mask], color=PRP, linewidth=0.7, zorder=3)
    if period_fft:
        ax3.axvline(period_fft, color=GOLD, linewidth=1.2, alpha=0.8)
    ax3.set_xlim(2, n // 3)
    ax3.set_ylabel("Spectral power", color=MUT, fontsize=9)
    ax3.set_xlabel("Period (windows)", color=MUT, fontsize=9)
    ax3.grid(axis="y", color="#1e293b", linewidth=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "mi_waveform_20k.png"
    fig.savefig(p, dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p


# ── JSON (same schema as results_5000primes.json) ───────────────────────────


def save_json(records: list[dict], period_info: dict, primes: list[int], ts: str,
              out_dir: Path, wall_clock_seconds: float) -> Path:
    mi_vals = [r["mi"] for r in records]
    data = {
        "timestamp": ts,
        "config": {
            "n_primes": len(primes),
            "n_gaps": len(primes) - 1,
            "n_windows": len(records),
            "window_size": WINDOW_SIZE,
            "scale": SCALE,
            "shots": SHOTS,
            "backend": "AerSimulator (clean)",
            "data_source": str(GAPS_CACHE_PATH.relative_to(REPO_ROOT)),
        },
        "prime_bounds": {
            "first": primes[0],
            "last": primes[-1],
            "largest_gap": max(primes[i + 1] - primes[i] for i in range(len(primes) - 1)),
        },
        "mi_stats": {
            "mean": round(float(np.mean(mi_vals)), 6),
            "std": round(float(np.std(mi_vals)), 6),
            "max": round(float(np.max(mi_vals)), 6),
            "min": round(float(np.min(mi_vals)), 6),
            "peak_window": int(np.argmax(mi_vals)),
            "trough_window": int(np.argmin(mi_vals)),
        },
        "period_analysis": {k: v for k, v in period_info.items() if k != "acf"},
        "runtime": {
            "wall_clock_seconds": round(wall_clock_seconds, 1),
            "wall_clock_minutes": round(wall_clock_seconds / 60, 2),
            "ms_per_window": round(1000 * wall_clock_seconds / len(records), 3),
        },
        "per_window": records,
    }
    p = out_dir / "results_20000primes.json"
    p.write_text(json.dumps(data, indent=2))
    return p


def auto_commit_push(ts: str) -> None:
    subprocess.run(["git", "add", f"output/prime/{ts}/"], check=True, cwd=REPO_ROOT)
    result = subprocess.run(
        ["git", "commit", "-m", f"experiment: 20000-prime terrain (AerSimulator) {ts} -- "
                                 f"19996-window MI generation, sim-only, no hardware"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  Committed output/prime/{ts}/")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed.")
    else:
        print(f"  Commit skipped: {result.stdout.strip()}")


def main() -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "output" / "prime" / ts / "terrain_20000primes"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  Quantum Terrain (sim) -- First {N_PRIMES:,} Primes, AerSimulator only")
    print(f"  Run: {ts}")
    print("=" * 65)

    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    primes = cache["primes"]
    assert len(primes) == N_PRIMES and primes[0] == 2 and primes[-1] == cache["last_prime"]
    all_gaps = cache["gaps"]
    assert len(all_gaps) == len(primes) - 1
    n_windows = len(all_gaps) - WINDOW_SIZE + 1
    print(f"\nLoaded {len(primes)} primes from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")
    print(f"  Gaps: {len(all_gaps)}  Windows: {n_windows}")
    print(f"  Gap range: {min(all_gaps)}-{max(all_gaps)}  Mean: {sum(all_gaps)/len(all_gaps):.2f}")

    print(f"\nRunning {n_windows:,} windows on AerSimulator (clean, no noise, no hardware)...")
    t0 = time.time()
    records = run_recurrent(all_gaps)
    wall_clock_seconds = time.time() - t0
    mi_vals = [r["mi"] for r in records]
    print(f"\n  Done in {wall_clock_seconds:.1f}s ({wall_clock_seconds/60:.2f} min). "
          f"MI mean={sum(mi_vals)/len(mi_vals):.4f}  max={max(mi_vals):.4f}  min={min(mi_vals):.4f}")

    print("\nAnalysing MI signal for periodicity...")
    period_info = detect_period(mi_vals)
    print(f"  ACF dominant period: {period_info['acf_dominant_period']} windows "
          f"(strength={period_info['acf_strength']})")
    print(f"  FFT dominant period: {period_info['fft_dominant_period']} windows "
          f"({period_info['fft_dominant_power_fraction']:.1%} of spectral power)")

    print("\nRendering MI waveform plot (terrain heatmap skipped -- not needed downstream, see docstring)...")
    plot_mi_waveform(records, period_info, ts, out_dir)
    print("  -> mi_waveform_20k.png")

    print("Saving JSON...")
    save_json(records, period_info, primes, ts, out_dir, wall_clock_seconds)
    print("  -> results_20000primes.json")

    auto_commit_push(ts)
    print()


if __name__ == "__main__":
    main()
