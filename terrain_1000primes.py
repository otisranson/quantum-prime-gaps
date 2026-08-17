"""terrain_1000primes.py

Extend the prime gap dataset to the first 1000 primes and run the recurrent
quantum circuit (v3 architecture) on all 996 windows via AerSimulator.
No hardware call — clean simulation only.

Outputs → output/prime/{YYYYMMDD_HHMMSS}/terrain_1000primes/:
  terrain_1000primes.png      full 996-window topographic map
  mi_waveform.png             MI across all 996 windows + period analysis
  results_1000primes.json     per-window MI, mode, and probability distributions

The MI waveform is the key signal: if prime gaps carry any quasi-periodic
structure, it should show up as a repeating pattern of peaks and troughs in
the mutual information between the circuit's two qubit halves.

Run: python terrain_1000primes.py
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
from matplotlib.colors import LightSource, LinearSegmentedColormap
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis.qft import synth_qft_full
from qiskit_aer import AerSimulator
from scipy import signal
from scipy.ndimage import zoom

# ── Constants ──────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).parent
WINDOW_SIZE = 4
SHOTS       = 8_192
SCALE       = 0.05
N_STATES    = 16
STATE_LABELS = [f"|{i:04b}⟩" for i in range(N_STATES)]

# ── Colormap (same as terrain_visualizer.py) ───────────────────────────────────

TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "quantum_terrain",
    [
        "#000814",
        "#03045e",
        "#5a189a",
        "#9b2226",
        "#d62828",
        "#f48c06",
        "#ffd60a",
    ],
)

BG   = "#0a0e18"
FG   = "#e8eaf6"
MUT  = "#7986cb"
GRN  = "#66bb6a"
GOLD = "#ffd60a"
PRP  = "#ce93d8"
BLU  = "#7dd3fc"

# ── Sieve of Eratosthenes ──────────────────────────────────────────────────────

def sieve(n_primes: int) -> list[int]:
    """Return the first n_primes prime numbers."""
    limit = max(n_primes * 15, 200)   # generous upper bound; 1000th prime = 7919
    sieve_arr = bytearray([1]) * (limit + 1)
    sieve_arr[0] = sieve_arr[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve_arr[i]:
            sieve_arr[i * i :: i] = bytearray(len(sieve_arr[i * i :: i]))
    primes = [i for i, v in enumerate(sieve_arr) if v]
    return primes[:n_primes]


def verify_primes(primes: list[int]) -> None:
    assert primes[0] == 2,    f"First prime wrong: {primes[0]}"
    assert primes[999] == 7919, f"1000th prime wrong: {primes[999]}"
    assert len(primes) == 1000, f"Count wrong: {len(primes)}"
    print(f"  Primes verified: first={primes[0]}, last={primes[-1]}, count={len(primes)}")

# ── Circuit (identical to v3) ──────────────────────────────────────────────────

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

# ── Recurrent simulation ───────────────────────────────────────────────────────

def run_recurrent(all_gaps: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Run all windows on clean AerSimulator. Returns (terrain, per_window_info)."""
    windows = [
        all_gaps[i : i + WINDOW_SIZE]
        for i in range(len(all_gaps) - WINDOW_SIZE + 1)
    ]
    n_windows = len(windows)
    terrain = np.zeros((n_windows, N_STATES))
    records = []

    sim = AerSimulator()
    offsets = [0.0] * WINDOW_SIZE
    t0 = time.time()

    for w_idx, window in enumerate(windows):
        qc, local_max = build_circuit(window, offsets)
        tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=42)
        counts = sim.run(tqc, shots=SHOTS).result().get_counts()

        # Full 16-state probability row
        probs = np.zeros(N_STATES)
        for bs, cnt in counts.items():
            probs[int(bs, 2)] = cnt / SHOTS
        terrain[w_idx] = probs

        bits    = counts_to_bits(counts, WINDOW_SIZE)
        mi      = mi_halves(bits, [0, 1], [2, 3])
        mode_bs = max(counts, key=counts.get)
        offsets = bitstring_to_offsets(mode_bs, SCALE)

        records.append({
            "w": w_idx,
            "gaps": window,
            "local_max": local_max,
            "mi": round(mi, 6),
            "mode": mode_bs,
        })

        if w_idx % 100 == 0 or w_idx == n_windows - 1:
            elapsed = time.time() - t0
            rate = (w_idx + 1) / elapsed
            eta = (n_windows - w_idx - 1) / rate if rate > 0 else 0
            print(
                f"  w{w_idx:03d}/{n_windows-1}  "
                f"gaps={window}  lmax={local_max}  "
                f"MI={mi:.4f}  mode={mode_bs}  "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s left]",
                flush=True,
            )

    return terrain, records

# ── Period detection ───────────────────────────────────────────────────────────

def detect_period(mi_vals: list[float]) -> dict:
    """Autocorrelation + FFT period detection on the MI signal."""
    arr = np.array(mi_vals)
    centered = arr - arr.mean()
    n = len(arr)

    # Autocorrelation (full, normalized)
    acf_full = np.correlate(centered, centered, mode="full")
    acf = acf_full[n - 1 :]   # positive lags
    acf /= acf[0]              # normalize so lag-0 = 1.0

    # Find peaks in ACF (exclude lag 0)
    acf_peaks, acf_props = signal.find_peaks(
        acf[1:], height=0.05, prominence=0.03, distance=5
    )
    acf_period = int(acf_peaks[0] + 1) if len(acf_peaks) > 0 else None
    acf_strength = float(acf[acf_period]) if acf_period else None

    # FFT — dominant frequency (excluding DC)
    fft_coeffs = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n)
    magnitudes = np.abs(fft_coeffs)
    magnitudes[0] = 0   # kill DC
    dominant_idx = int(np.argmax(magnitudes))
    fft_period = round(1 / freqs[dominant_idx]) if freqs[dominant_idx] > 0 else None
    fft_power_fraction = float(
        magnitudes[dominant_idx] ** 2 / np.sum(magnitudes ** 2)
    )

    # Top 5 FFT periods
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
        "acf": acf[:200].tolist(),   # store first 200 lags
    }

# ── Static terrain plot ────────────────────────────────────────────────────────

def plot_terrain(terrain: np.ndarray, records: list[dict], period_info: dict,
                 ts: str, out_dir: Path) -> Path:
    n_windows = terrain.shape[0]
    mi_vals = [r["mi"] for r in records]
    peak_win   = int(np.argmax(mi_vals))
    trough_win = int(np.argmin(mi_vals))

    # Smooth to 2× resolution for hillshading
    terrain_hires = zoom(terrain, (2, 4), order=3)
    terrain_hires = np.clip(terrain_hires, 0, None)

    ls = LightSource(azdeg=270, altdeg=40)
    rgb = ls.shade(
        terrain_hires,
        cmap=TERRAIN_CMAP,
        vert_exag=8.0,
        blend_mode="soft",
        vmin=0,
        vmax=terrain.max(),
    )

    fig_h = max(20, n_windows // 30)   # ~33 rows per inch
    fig, ax = plt.subplots(figsize=(18, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.imshow(rgb, aspect="auto", origin="upper",
              extent=[-0.5, N_STATES - 0.5, n_windows - 0.5, -0.5])

    # Contour lines at 6 levels
    X = np.arange(N_STATES)
    Y = np.arange(n_windows)
    levels = np.linspace(terrain.min() + 0.003, terrain.max(), 6)
    ax.contour(X, Y, terrain, levels=levels,
               colors="white", alpha=0.15, linewidths=0.4)

    # Annotate peak MI
    ax.annotate(
        f"Peak MI\nw{peak_win}: {mi_vals[peak_win]:.3f}",
        xy=(N_STATES - 0.5, peak_win), xytext=(N_STATES + 0.3, peak_win),
        fontsize=7, color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8),
        ha="left", va="center",
    )
    ax.annotate(
        f"Trough MI\nw{trough_win}: {mi_vals[trough_win]:.3f}",
        xy=(N_STATES - 0.5, trough_win), xytext=(N_STATES + 0.3, trough_win),
        fontsize=7, color=PRP,
        arrowprops=dict(arrowstyle="->", color=PRP, lw=0.8),
        ha="left", va="center",
    )

    # Detected period: draw dashed horizontal lines every period windows
    period = period_info.get("fft_dominant_period")
    if period and period < n_windows // 2:
        for tick in range(0, n_windows, period):
            ax.axhline(tick, color=BLU, linewidth=0.3, alpha=0.35, linestyle="--")

    ax.set_xlim(-0.5, N_STATES - 0.5)
    ax.set_ylim(n_windows - 0.5, -0.5)
    ax.set_xticks(range(N_STATES))
    ax.set_xticklabels(STATE_LABELS, rotation=45, ha="right", fontsize=7, color=MUT)
    yticks = list(range(0, n_windows, 100)) + [n_windows - 1]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"w{j}" for j in yticks], fontsize=7, color=MUT)
    ax.tick_params(colors=MUT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    period_str = (
        f"Detected FFT period ≈ {period} windows (dashed lines)"
        if period else "No dominant period detected"
    )
    ax.set_title(
        f"Quantum Terrain — First 1000 Primes, {n_windows} Recurrent Windows (AerSimulator)\n"
        f"Red peaks = high probability  |  {period_str}",
        color=FG, fontsize=10, pad=10,
    )
    ax.set_xlabel("Basis state", color=MUT, fontsize=9, labelpad=8)
    ax.set_ylabel("Window (position in prime gap sequence)", color=MUT, fontsize=9)

    # MI strip
    mi_arr = np.array(mi_vals).reshape(-1, 1)
    mi_ax = ax.inset_axes([-0.06, 0, 0.04, 1])
    mi_ax.imshow(mi_arr, aspect="auto", origin="upper",
                 cmap="plasma", vmin=0, vmax=max(mi_vals))
    mi_ax.set_xticks([])
    mi_ax.set_yticks([])
    mi_ax.set_title("MI", color=MUT, fontsize=6, pad=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=TERRAIN_CMAP,
                                norm=plt.Normalize(0, terrain.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.12)
    cbar.set_label("Probability", color=MUT, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=MUT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUT, fontsize=7)

    fig.tight_layout()
    p = out_dir / "terrain_1000primes.png"
    fig.savefig(p, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p

# ── MI waveform + period analysis plot ────────────────────────────────────────

def plot_mi_waveform(records: list[dict], period_info: dict,
                     ts: str, out_dir: Path) -> Path:
    mi_vals = [r["mi"] for r in records]
    n = len(mi_vals)
    x = np.arange(n)

    period_acf = period_info["acf_dominant_period"]
    period_fft = period_info["fft_dominant_period"]
    acf        = np.array(period_info["acf"])

    fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"MI Waveform — First 1000 Primes, {n} Windows (AerSimulator)  [{ts}]\n"
        "Mutual information between qubit halves [q0,q1] vs [q2,q3]",
        color=FG, fontsize=11, y=0.99,
    )

    # ── Panel 1: full MI trace ────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    ax1.tick_params(colors=MUT)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e293b")

    ax1.plot(x, mi_vals, color=BLU, linewidth=0.7, alpha=0.85, zorder=3)

    # Rolling mean (window=30) overlay
    kernel = np.ones(30) / 30
    rolling = np.convolve(mi_vals, kernel, mode="same")
    ax1.plot(x, rolling, color=GOLD, linewidth=1.5, alpha=0.9, label="30-window rolling mean")

    # Mark global peak and trough
    peak_w   = int(np.argmax(mi_vals))
    trough_w = int(np.argmin(mi_vals))
    ax1.scatter([peak_w], [mi_vals[peak_w]], color=GRN, s=40, zorder=5)
    ax1.scatter([trough_w], [mi_vals[trough_w]], color="#ef5350", s=40, zorder=5)
    ax1.annotate(
        f"peak w{peak_w}\nMI={mi_vals[peak_w]:.3f}",
        xy=(peak_w, mi_vals[peak_w]), xytext=(peak_w + 20, mi_vals[peak_w] + 0.01),
        fontsize=7, color=GRN,
        arrowprops=dict(arrowstyle="->", color=GRN, lw=0.8),
    )

    # Detected period tick marks
    if period_fft and period_fft < n // 2:
        for tick in range(0, n, period_fft):
            ax1.axvline(tick, color=PRP, linewidth=0.4, alpha=0.4, linestyle="--")

    ax1.set_xlim(0, n)
    ax1.set_ylabel("Root MI (bits)", color=MUT, fontsize=9)
    ax1.set_xlabel("Window index", color=MUT, fontsize=9)
    ax1.legend(framealpha=0, labelcolor=GOLD, fontsize=9)
    ax1.grid(axis="y", color="#1e293b", linewidth=0.5)

    # ── Panel 2: autocorrelation ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.tick_params(colors=MUT)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e293b")

    lags = np.arange(len(acf))
    ax2.plot(lags, acf, color=BLU, linewidth=1.0)
    ax2.axhline(0, color="#1e293b", linewidth=0.8)
    ax2.axhline(0.1, color=MUT, linewidth=0.5, linestyle="--", alpha=0.5)

    if period_acf:
        ax2.axvline(period_acf, color=GOLD, linewidth=1.2, alpha=0.8)
        ax2.text(period_acf + 1, 0.5, f"ACF period\n≈ {period_acf} windows",
                 color=GOLD, fontsize=8, va="center")

    ax2.set_xlim(0, len(acf) - 1)
    ax2.set_ylabel("Autocorrelation", color=MUT, fontsize=9)
    ax2.set_xlabel("Lag (windows)", color=MUT, fontsize=9)
    ax2.set_title("Autocorrelation of MI signal", color=FG, fontsize=9)
    ax2.grid(axis="y", color="#1e293b", linewidth=0.5)

    # ── Panel 3: FFT power spectrum ───────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(BG)
    ax3.tick_params(colors=MUT)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#1e293b")

    mi_arr     = np.array(mi_vals) - np.mean(mi_vals)
    fft_coeffs = np.fft.rfft(mi_arr)
    freqs      = np.fft.rfftfreq(n)
    power      = np.abs(fft_coeffs) ** 2
    power[0]   = 0   # suppress DC

    # Convert frequency to period for x-axis
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1 / freqs, 0)

    # Plot against period (skip DC and very short periods)
    mask = (periods >= 2) & (periods <= n // 2)
    ax3.fill_between(periods[mask], power[mask],
                     color=PRP, alpha=0.5, zorder=2)
    ax3.plot(periods[mask], power[mask], color=PRP, linewidth=0.8, zorder=3)

    if period_fft:
        ax3.axvline(period_fft, color=GOLD, linewidth=1.2, alpha=0.8)
        ax3.text(
            period_fft + 2,
            power[mask].max() * 0.7,
            f"FFT period\n≈ {period_fft} windows\n"
            f"({period_info['fft_dominant_power_fraction']:.1%} power)",
            color=GOLD, fontsize=8, va="center",
        )

    ax3.set_xlim(2, n // 2)
    ax3.set_ylabel("Spectral power", color=MUT, fontsize=9)
    ax3.set_xlabel("Period (windows)", color=MUT, fontsize=9)
    ax3.set_title("FFT power spectrum of MI signal", color=FG, fontsize=9)
    ax3.grid(axis="y", color="#1e293b", linewidth=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "mi_waveform.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p

# ── JSON output ────────────────────────────────────────────────────────────────

def save_json(records: list[dict], period_info: dict,
              primes: list[int], ts: str, out_dir: Path) -> Path:
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
        },
        "prime_bounds": {
            "first": primes[0],
            "last": primes[-1],
            "largest_gap": max(primes[i+1] - primes[i] for i in range(len(primes)-1)),
        },
        "mi_stats": {
            "mean": round(float(np.mean(mi_vals)), 6),
            "std":  round(float(np.std(mi_vals)), 6),
            "max":  round(float(np.max(mi_vals)), 6),
            "min":  round(float(np.min(mi_vals)), 6),
            "peak_window": int(np.argmax(mi_vals)),
            "trough_window": int(np.argmin(mi_vals)),
        },
        "period_analysis": {
            k: v for k, v in period_info.items() if k != "acf"
        },
        "per_window": records,
    }
    p = out_dir / "results_1000primes.json"
    p.write_text(json.dumps(data, indent=2))
    return p

# ── Auto-commit ────────────────────────────────────────────────────────────────

def auto_commit_push(ts: str) -> None:
    subprocess.run(
        ["git", "add", f"output/prime/{ts}/"],
        check=True, cwd=REPO_ROOT,
    )
    result = subprocess.run(
        ["git", "commit", "-m",
         f"1000-prime terrain {ts} — 996-window AerSimulator run + MI period analysis"],
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
    out_dir = REPO_ROOT / "output" / "prime" / ts / "terrain_1000primes"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Quantum Terrain — First 1000 Primes")
    print(f"  Run: {ts}")
    print("=" * 65)

    # Step 1: primes
    print("\nGenerating first 1000 primes...")
    primes = sieve(1000)
    verify_primes(primes)
    all_gaps = [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]
    n_windows = len(all_gaps) - WINDOW_SIZE + 1
    print(f"  Gaps: {len(all_gaps)}  Windows: {n_windows}")
    print(f"  Gap range: {min(all_gaps)}–{max(all_gaps)}  Mean: {sum(all_gaps)/len(all_gaps):.2f}")

    # Step 2: recurrent simulation
    print(f"\nRunning {n_windows} windows on AerSimulator...")
    terrain, records = run_recurrent(all_gaps)
    mi_vals = [r["mi"] for r in records]
    print(f"\n  Done. MI mean={sum(mi_vals)/len(mi_vals):.4f}  "
          f"max={max(mi_vals):.4f}  min={min(mi_vals):.4f}")

    # Period analysis
    print("\nAnalysing MI signal for periodicity...")
    period_info = detect_period(mi_vals)
    print(f"  ACF dominant period: {period_info['acf_dominant_period']} windows "
          f"(strength={period_info['acf_strength']})")
    print(f"  FFT dominant period: {period_info['fft_dominant_period']} windows "
          f"({period_info['fft_dominant_power_fraction']:.1%} of spectral power)")
    print(f"  FFT top-5 periods: {[t['period'] for t in period_info['fft_top5']]}")

    # Step 3: render
    print("\nRendering terrain map...")
    plot_terrain(terrain, records, period_info, ts, out_dir)
    print("  → terrain_1000primes.png")

    print("Rendering MI waveform...")
    plot_mi_waveform(records, period_info, ts, out_dir)
    print("  → mi_waveform.png")

    print("Saving JSON...")
    save_json(records, period_info, primes, ts, out_dir)
    print("  → results_1000primes.json")

    auto_commit_push(ts)
    print()


if __name__ == "__main__":
    main()
