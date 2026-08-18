"""layer3_regime_characterization.py

Follow-up to the Layer 3 refutation (hypotheses/regime_internal_wave_structure.md,
"Empirical Check -- Regime Overlay"): that test found the three changepoint-bounded
regimes are significantly *less* alike than a random-slice null, both raw and
detrended (bottom ~1st percentile, 5000-trial null). Cross-regime shape matching
is refuted -- this script does not retest that claim.

Instead of comparing the regimes to each other, this treats each of the three
regimes as its own independent object and characterizes it on its own terms: no
pairwise comparison, no forced matching, no similarity scoring, no null testing
(purely descriptive, so there is nothing to compute a base rate against).

Same regime definitions as the refutation test: three changepoints (windows
1529, 2501, 4211) delimit [0, 1529), [1529, 2501), [2501, 4211) in raw gap-index
space, sequence start as regime 0's implicit open end, tail after 4211 excluded
(no closing changepoint -- see the refutation writeup for the full scope note).

Per regime, independently:
  1. Spike density   -- fraction of points with |z-score| > 2 (z relative to
                         that regime's own mean/std, not the other regimes')
  2. Volatility       -- rolling std (K=100) over the regime's own raw sequence,
                         plus its linear trend (increasing/decreasing across
                         the regime)
  3. Dominant frequency -- FFT power spectrum of the (de-meaned) raw sequence,
                         top 3 peaks by power, reported as period (gap-steps
                         per cycle) and power
  4. Distribution shape -- skewness and excess kurtosis of the raw gap values
  5. Mean and variance -- of the raw gap values themselves

Output: the usual timestamped output/prime/{ts}/ folder (PNG + JSON), a
summary table (console + table image + JSON), auto-committed and pushed.

Run: python layer3_regime_characterization.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_5000.json"
KNOWN_CHANGEPOINTS = [1529, 2501, 4211]
REGIME_LABELS = ["regime 0", "regime 1", "regime 2"]
REGIME_COLORS = ["#4c72b0", "#e08214", "#2a9d5c"]

ROLLING_STD_WINDOW = 100
SPIKE_Z_THRESHOLD = 2.0
N_FFT_PEAKS = 3

OUT_ROOT = REPO_ROOT / "output" / "prime"


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        return np.array(json.load(f)["gaps"])


def spike_density(x: np.ndarray, z_thresh: float) -> tuple[float, int]:
    std = x.std()
    if std == 0:
        return 0.0, 0
    z = (x - x.mean()) / std
    spikes = np.abs(z) > z_thresh
    return float(spikes.sum() / len(x)), int(spikes.sum())


def rolling_std(x: np.ndarray, k: int) -> np.ndarray:
    """Leading-window rolling std: value at position i is std(x[i:i+k])."""
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x**2, 0, 0.0))
    mean = (c1[k:] - c1[:-k]) / k
    mean_sq = (c2[k:] - c2[:-k]) / k
    return np.sqrt(np.clip(mean_sq - mean**2, 0.0, None))


def volatility_summary(x: np.ndarray, k: int) -> dict:
    vol = rolling_std(x, k)
    pos = np.linspace(0.0, 1.0, len(vol))
    slope, intercept = np.polyfit(pos, vol, 1)
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    return {
        "window": k,
        "mean": float(vol.mean()),
        "start": float(vol[0]),
        "end": float(vol[-1]),
        "trend_slope": float(slope),
        "trend_direction": direction,
        "series": vol,
    }


def fft_top_peaks(x: np.ndarray, n_peaks: int) -> list[dict]:
    n = len(x)
    detrended = x - x.mean()
    spectrum = np.fft.rfft(detrended)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Drop the DC bin (freq=0, undefined period) before ranking peaks.
    power_ac, freqs_ac = power[1:], freqs[1:]
    top_idx = np.argsort(power_ac)[::-1][:n_peaks]
    peaks = [
        {"period_gaps": float(1.0 / freqs_ac[i]), "frequency": float(freqs_ac[i]), "power": float(power_ac[i])}
        for i in top_idx
    ]
    return peaks, freqs_ac, power_ac


def skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    mean, std = x.mean(), x.std()
    if std == 0:
        return 0.0, 0.0
    skew = float(np.mean((x - mean) ** 3) / std**3)
    kurt = float(np.mean((x - mean) ** 4) / std**4 - 3.0)  # excess kurtosis
    return skew, kurt


def characterize_regime(x: np.ndarray, k: int) -> dict:
    density, count = spike_density(x, SPIKE_Z_THRESHOLD)
    vol = volatility_summary(x, k)
    peaks, freqs_ac, power_ac = fft_top_peaks(x, N_FFT_PEAKS)
    skew, kurt = skew_kurtosis(x)
    return {
        "length": len(x),
        "spike_density": density,
        "spike_count": count,
        "volatility": vol,
        "fft_peaks": peaks,
        "fft_freqs": freqs_ac,
        "fft_power": power_ac,
        "skew": skew,
        "kurtosis_excess": kurt,
        "mean": float(x.mean()),
        "variance": float(x.var()),
    }


def print_summary_table(records: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'metric':<28}" + "".join(f"{lbl:>24}" for lbl in REGIME_LABELS))
    print("-" * 100)
    rows = [
        ("length", lambda r: f"{r['length']}"),
        ("spike density (>2 sigma)", lambda r: f"{r['spike_density']:.4f} (n={r['spike_count']})"),
        ("volatility mean (K=100)", lambda r: f"{r['volatility']['mean']:.4f}"),
        ("volatility trend", lambda r: f"{r['volatility']['trend_slope']:+.4f} ({r['volatility']['trend_direction']})"),
        ("FFT top period (gaps)", lambda r: f"{r['fft_peaks'][0]['period_gaps']:.1f}"),
        ("skew", lambda r: f"{r['skew']:.4f}"),
        ("excess kurtosis", lambda r: f"{r['kurtosis_excess']:.4f}"),
        ("mean gap", lambda r: f"{r['mean']:.4f}"),
        ("variance", lambda r: f"{r['variance']:.4f}"),
    ]
    for label, fn in rows:
        print(f"{label:<28}" + "".join(f"{fn(r):>24}" for r in records))
    print("=" * 100)


def plot_table_image(records: list[dict], ts: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    row_labels = [
        "length", "spike density (>2σ)", "volatility mean (K=100)", "volatility trend",
        "FFT top period (gaps)", "FFT top power", "skew", "excess kurtosis", "mean gap", "variance",
    ]
    cell_text = []
    for r in records:
        col = [
            f"{r['length']}",
            f"{r['spike_density']:.4f} (n={r['spike_count']})",
            f"{r['volatility']['mean']:.4f}",
            f"{r['volatility']['trend_slope']:+.4f} ({r['volatility']['trend_direction']})",
            f"{r['fft_peaks'][0]['period_gaps']:.1f}",
            f"{r['fft_peaks'][0]['power']:.2f}",
            f"{r['skew']:.4f}",
            f"{r['kurtosis_excess']:.4f}",
            f"{r['mean']:.4f}",
            f"{r['variance']:.4f}",
        ]
        cell_text.append(col)
    # Transpose so metrics are rows, regimes are columns.
    table_data = list(map(list, zip(*cell_text, strict=True)))
    table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=REGIME_LABELS,
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title(f"Layer 3 regime characterization -- summary table [{ts}]", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regime_panels(regimes_raw: list[np.ndarray], records: list[dict], ts: str, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for i, (x, rec, color) in enumerate(zip(regimes_raw, records, REGIME_COLORS, strict=True)):
        ax_seq = axes[i, 0]
        ax_seq.plot(np.arange(len(x)), x, color=color, lw=0.6, alpha=0.6, label="raw gap")
        ax_vol = ax_seq.twinx()
        vol = rec["volatility"]["series"]
        vol_x = np.arange(len(vol)) + ROLLING_STD_WINDOW / 2
        ax_vol.plot(vol_x, vol, color="black", lw=1.4, label=f"rolling std (K={ROLLING_STD_WINDOW})")
        ax_seq.set_title(f"{REGIME_LABELS[i]} (n={rec['length']}) -- raw sequence + volatility overlay")
        ax_seq.set_xlabel("index within regime")
        ax_seq.set_ylabel("gap size", color=color)
        ax_vol.set_ylabel("rolling std", color="black")
        lines1, labels1 = ax_seq.get_legend_handles_labels()
        lines2, labels2 = ax_vol.get_legend_handles_labels()
        ax_seq.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

        ax_fft = axes[i, 1]
        freqs_ac, power_ac = rec["fft_freqs"], rec["fft_power"]
        ax_fft.plot(freqs_ac, power_ac, color=color, lw=0.8)
        ax_fft.set_xscale("log")
        ax_fft.set_yscale("log")
        for peak in rec["fft_peaks"]:
            ax_fft.axvline(peak["frequency"], color="black", lw=1, ls="--", alpha=0.7)
            ax_fft.annotate(f"period={peak['period_gaps']:.0f}", xy=(peak["frequency"], peak["power"]),
                             fontsize=7, rotation=90, va="bottom", ha="right")
        ax_fft.set_title(f"{REGIME_LABELS[i]} -- FFT power spectrum (top {N_FFT_PEAKS} peaks marked)")
        ax_fft.set_xlabel("frequency (cycles / gap-step)")
        ax_fft.set_ylabel("power")

    fig.suptitle(f"Layer 3 regime characterization -- descriptive only, no cross-regime comparison [{ts}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def auto_commit_push(out_dir: Path, records: list[dict], ts: str) -> None:
    means = [f"{r['mean']:.2f}" for r in records]
    densities = [f"{r['spike_density']:.3f}" for r in records]
    top_periods = [f"{r['fft_peaks'][0]['period_gaps']:.0f}" for r in records]
    msg = (f"analysis: layer3 regime characterization {ts} -- "
           f"mean gaps={means}, spike density={densities}, top FFT periods={top_periods}")
    subprocess.run(["git", "add", str(out_dir.relative_to(REPO_ROOT))], check=True, cwd=REPO_ROOT)
    commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        print(f"\n  Committed: {out_dir.relative_to(REPO_ROOT)}")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"\n  Git commit skipped: {commit.stdout.strip()}")


def main() -> None:
    full_gaps = load_full_gaps()
    assert len(full_gaps) == 4999, f"expected 4999 raw gaps, got {len(full_gaps)}"
    print(f"Loaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")

    bounds = [(0, KNOWN_CHANGEPOINTS[0]), (KNOWN_CHANGEPOINTS[0], KNOWN_CHANGEPOINTS[1]),
              (KNOWN_CHANGEPOINTS[1], KNOWN_CHANGEPOINTS[2])]
    regimes_raw = [full_gaps[a:b] for a, b in bounds]
    print(f"Regime bounds (same as the refutation test): {bounds}")
    print(f"Excluded tail: [{KNOWN_CHANGEPOINTS[2]}, {len(full_gaps)})")

    records = [characterize_regime(x, ROLLING_STD_WINDOW) for x in regimes_raw]
    print_summary_table(records)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    panels_path = out_dir / "layer3_characterization_panels.png"
    table_path = out_dir / "layer3_characterization_table.png"
    plot_regime_panels(regimes_raw, records, ts, panels_path)
    plot_table_image(records, ts, table_path)
    print(f"\nSaved figure to {panels_path.relative_to(REPO_ROOT)}")
    print(f"Saved figure to {table_path.relative_to(REPO_ROOT)}")

    json_out = {
        "timestamp": ts,
        "results_source": str(GAPS_CACHE_PATH.relative_to(REPO_ROOT)),
        "regime_bounds": bounds,
        "excluded_tail_bounds": [KNOWN_CHANGEPOINTS[2], len(full_gaps)],
        "config": {
            "rolling_std_window": ROLLING_STD_WINDOW,
            "spike_z_threshold": SPIKE_Z_THRESHOLD,
            "n_fft_peaks": N_FFT_PEAKS,
        },
        "scope_note": "Purely descriptive per-regime characterization; no cross-regime similarity "
                       "scoring or null testing performed here -- see the refutation writeup for that.",
        "regimes": [
            {
                "label": REGIME_LABELS[i],
                "bounds": bounds[i],
                "length": records[i]["length"],
                "spike_density": round(records[i]["spike_density"], 6),
                "spike_count": records[i]["spike_count"],
                "volatility": {
                    "window": records[i]["volatility"]["window"],
                    "mean": round(records[i]["volatility"]["mean"], 6),
                    "start": round(records[i]["volatility"]["start"], 6),
                    "end": round(records[i]["volatility"]["end"], 6),
                    "trend_slope": round(records[i]["volatility"]["trend_slope"], 6),
                    "trend_direction": records[i]["volatility"]["trend_direction"],
                },
                "fft_top_peaks": [
                    {"period_gaps": round(p["period_gaps"], 3), "frequency": round(p["frequency"], 6),
                     "power": round(p["power"], 3)}
                    for p in records[i]["fft_peaks"]
                ],
                "skew": round(records[i]["skew"], 6),
                "kurtosis_excess": round(records[i]["kurtosis_excess"], 6),
                "mean": round(records[i]["mean"], 6),
                "variance": round(records[i]["variance"], 6),
            }
            for i in range(3)
        ],
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(json_out, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    auto_commit_push(out_dir, records, ts)


if __name__ == "__main__":
    main()
