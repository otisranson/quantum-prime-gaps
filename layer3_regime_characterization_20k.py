"""layer3_regime_characterization_20k.py

Scale-up of layer3_regime_characterization.py (which characterized the
original 3 regimes on their own terms -- mean, variance, spike density,
volatility trend, FFT top peaks, skew, kurtosis -- with no cross-regime
comparison) to the 40 regimes carved by the 39 changepoints found in the
20k Scale-Up run (hypotheses/regime_internal_wave_structure.md, "## 20k
Scale-Up: Intensity vs. Position"). Reuses those exact changepoint
positions from output/prime/20260818_015045/results.json rather than
re-detecting them.

Same per-regime methodology as the original 3-regime script, with one
deliberate change: the original found regime 0's top FFT peak sitting at
period == its own length, later diagnosed as an artifact of only
mean-subtracting (not detrending) before the FFT when a regime has a
strong internal trend. This script detrends (removes each regime's own
best-fit line, not just its mean) before every FFT here, prospectively,
rather than repeating that artifact 40 times.

Regime definition: the 39 changepoints split the full 19999-gap sequence
into 40 regimes, unlike the original 3-regime work which excluded the
tail after the last changepoint (no closing changepoint for it there).
Here all 40 segments are used, including the leading segment before the
first changepoint and the trailing segment after the last one -- both are
"bounded" in the sense of being one contiguous run between two adjacent
detected changepoints (or the sequence boundary), which is the natural
reading with 39 changepoints for "carve into 40 regimes."

The real test this time: rather than eyeballing whether mean/variance/
skew/kurtosis look like they trend with position across 40 points, each
is tested with the same permutation-correlation method used for the
position-vs-intensity result in the 20k scale-up (correlation + a
permutation-null p-value, not an assumed-normal significance test).

Run: python layer3_regime_characterization_20k.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"
CHANGEPOINTS_SOURCE = REPO_ROOT / "output/prime/20260818_015045/results.json"

ROLLING_STD_WINDOW = 100
SPIKE_THRESHOLDS = [2.0, 2.5, 3.0]
N_FFT_PEAKS = 3
MIN_CONFIDENT_SIZE = 100  # skew/kurtosis (4th moment) need a reasonable n; below this, flag not drop
N_PERM_CORR = 5000
SEED = 42

OUT_ROOT = REPO_ROOT / "output" / "prime"


# ── Data loading ─────────────────────────────────────────────────────────


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    gaps = np.array(cache["gaps"])
    assert len(gaps) == cache["n_gaps"]
    return gaps


def load_changepoints() -> list[dict]:
    with open(CHANGEPOINTS_SOURCE) as f:
        data = json.load(f)
    cps = data["changepoints"]
    assert cps == sorted(cps, key=lambda c: c["position"]), "changepoints in source file are not position-sorted"
    return cps


# ── Per-regime statistics (same formulas as layer3_regime_characterization.py) ─


def spike_density(x: np.ndarray, z_thresh: float) -> tuple[float, int]:
    std = x.std()
    if std == 0:
        return 0.0, 0
    z = (x - x.mean()) / std
    spikes = np.abs(z) > z_thresh
    return float(spikes.sum() / len(x)), int(spikes.sum())


def rolling_std(x: np.ndarray, k: int) -> np.ndarray:
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x**2, 0, 0.0))
    mean = (c1[k:] - c1[:-k]) / k
    mean_sq = (c2[k:] - c2[:-k]) / k
    return np.sqrt(np.clip(mean_sq - mean**2, 0.0, None))


def volatility_summary(x: np.ndarray, k: int) -> dict:
    if len(x) <= k:
        return {"window": k, "mean": float("nan"), "start": float("nan"), "end": float("nan"),
                "trend_slope": float("nan"), "trend_direction": "n/a (regime shorter than window)",
                "series": np.array([])}
    vol = rolling_std(x, k)
    pos = np.linspace(0.0, 1.0, len(vol))
    slope, intercept = np.polyfit(pos, vol, 1)
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    return {"window": k, "mean": float(vol.mean()), "start": float(vol[0]), "end": float(vol[-1]),
            "trend_slope": float(slope), "trend_direction": direction, "series": vol}


def fft_top_peaks_detrended(x: np.ndarray, n_peaks: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Same top-N-by-power FFT peak extraction as the original script, but
    linear-detrended (own best-fit line removed) before the transform
    instead of only mean-subtracted -- avoids the regime-0 trend-artifact
    (top period landing at the regime's own length) found last time."""
    n = len(x)
    idx = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(idx, x, 1)
    detrended = x - (slope * idx + intercept)
    spectrum = np.fft.rfft(detrended)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
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
    kurt = float(np.mean((x - mean) ** 4) / std**4 - 3.0)
    return skew, kurt


def characterize_regime(x: np.ndarray, k: int) -> dict:
    spikes = {thresh: spike_density(x, thresh) for thresh in SPIKE_THRESHOLDS}
    vol = volatility_summary(x, k)
    peaks, freqs_ac, power_ac = fft_top_peaks_detrended(x, N_FFT_PEAKS)
    skew, kurt = skew_kurtosis(x)
    return {
        "length": len(x),
        "spike_density": {str(t): d for t, (d, _c) in spikes.items()},
        "spike_count": {str(t): c for t, (_d, c) in spikes.items()},
        "volatility": vol,
        "fft_peaks": peaks,
        "fft_freqs": freqs_ac,
        "fft_power": power_ac,
        "skew": skew,
        "kurtosis_excess": kurt,
        "mean": float(x.mean()),
        "variance": float(x.var()),
        "low_confidence": len(x) < MIN_CONFIDENT_SIZE,
    }


# ── Correlation test (same method as the 20k scale-up's position-vs-intensity test) ─


def permutation_correlation_test(positions: np.ndarray, values: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    observed_r = float(np.corrcoef(positions, values)[0, 1])
    null_r = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(values)
        null_r[i] = np.corrcoef(positions, shuffled)[0, 1]
    p_value = float(np.mean(np.abs(null_r) >= abs(observed_r)))
    return {"observed_r": observed_r, "null_mean": float(null_r.mean()), "null_std": float(null_r.std()),
            "p_value_two_tailed": p_value}


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_stat_trends(positions: np.ndarray, records: list[dict], corr_tests: dict, low_conf: np.ndarray, ts: str, out_path: Path) -> None:
    stats = [("mean", "mean gap"), ("variance", "variance"), ("skew", "skew"), ("kurtosis_excess", "excess kurtosis")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (key, label) in zip(axes.flat, stats, strict=True):
        values = np.array([r[key] for r in records])
        colors = np.where(low_conf, "#d1495b", "#4c72b0")
        ax.scatter(positions, values, c=colors, s=30, zorder=3)
        fit = np.polyfit(positions, values, 1)
        xs = np.linspace(positions.min(), positions.max(), 100)
        ax.plot(xs, np.polyval(fit, xs), color="#888888", ls="--", lw=1.2)
        corr = corr_tests[key]
        ax.set_title(f"{label} vs. position (r={corr['observed_r']:.3f}, p={corr['p_value_two_tailed']:.4f})")
        ax.set_xlabel("regime start position (raw gap index)")
        ax.set_ylabel(label)
    fig.suptitle(f"40-regime stats vs. position (n=40, red = below MIN_CONFIDENT_SIZE={MIN_CONFIDENT_SIZE}) [{ts}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_table_image(rows: list[dict], out_path: Path, ts: str) -> None:
    fig, ax = plt.subplots(figsize=(14, max(8, 0.28 * len(rows))))
    ax.axis("off")
    col_labels = ["regime", "position", "length", "mean", "variance", "skew",
                  "kurtosis\n(internal)", "kurtosis\n(boundary)", "spike\ndensity 2σ", "low\nconf."]
    cell_text = []
    for r in rows:
        cell_text.append([
            f"{r['index']}", f"{r['position']}", f"{r['length']}",
            f"{r['mean']:.2f}", f"{r['variance']:.2f}", f"{r['skew']:.3f}",
            f"{r['kurtosis_internal']:.3f}",
            f"{r['kurtosis_boundary']:.3f}" if r["kurtosis_boundary"] is not None else "—",
            f"{r['spike_density_2s']:.4f}", "YES" if r["low_confidence"] else "",
        ])
    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.15)
    ax.set_title(f"40-regime characterization -- full stats table [{ts}]", pad=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Auto-commit / push ──────────────────────────────────────────────────


def auto_commit_push(out_dir: Path, corr_tests: dict, n_low_conf: int, ts: str) -> None:
    sig = [k for k, v in corr_tests.items() if v["p_value_two_tailed"] < 0.05]
    msg = (f"analysis: layer3 40-regime characterization {ts} -- "
           f"40 regimes, {n_low_conf} low-confidence, significant trends (p<0.05): {sig or 'none'}")
    subprocess.run(["git", "add", str(out_dir.relative_to(REPO_ROOT))], check=True, cwd=REPO_ROOT)
    commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        print(f"\n  Committed: {out_dir.relative_to(REPO_ROOT)}")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"\n  Git commit skipped: {commit.stdout.strip()}")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    rng = np.random.default_rng(SEED)

    full_gaps = load_full_gaps()
    cps = load_changepoints()
    positions_cp = [c["position"] for c in cps]
    print(f"Loaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")
    print(f"Loaded {len(cps)} changepoints from {CHANGEPOINTS_SOURCE.relative_to(REPO_ROOT)}: {positions_cp[:3]}...{positions_cp[-3:]}")

    bounds = [(0, positions_cp[0])]
    bounds += [(positions_cp[i], positions_cp[i + 1]) for i in range(len(positions_cp) - 1)]
    bounds += [(positions_cp[-1], len(full_gaps))]
    assert len(bounds) == len(cps) + 1 == 40
    assert sum(b - a for a, b in bounds) == len(full_gaps)

    regimes_raw = [full_gaps[a:b] for a, b in bounds]
    lengths = [len(r) for r in regimes_raw]
    print(f"40 regimes, lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    records = [characterize_regime(x, ROLLING_STD_WINDOW) for x in regimes_raw]
    n_low_conf = sum(r["low_confidence"] for r in records)
    print(f"Regimes below MIN_CONFIDENT_SIZE={MIN_CONFIDENT_SIZE}: {n_low_conf}")

    # Boundary kurtosis: regime i+1 (the regime that *starts* at changepoint i)
    # gets that changepoint's fine-grained local peak kurtosis; regime 0 has no
    # preceding changepoint, so no boundary value.
    boundary_kurtosis: list[float | None] = [None] + [c["peak_kurtosis"] for c in cps]

    positions = np.array([b[0] for b in bounds], dtype=float)
    low_conf = np.array([r["low_confidence"] for r in records])

    print("\nRunning permutation correlation tests (position vs. mean/variance/skew/kurtosis, n=40, "
          f"n_perm={N_PERM_CORR})...")
    corr_tests = {}
    for key in ["mean", "variance", "skew", "kurtosis_excess"]:
        values = np.array([r[key] for r in records])
        corr_tests[key] = permutation_correlation_test(positions, values, N_PERM_CORR, rng)
        c = corr_tests[key]
        sig = "SIGNIFICANT (p<0.05)" if c["p_value_two_tailed"] < 0.05 else "not significant"
        print(f"  {key:16s} r={c['observed_r']:+.4f}  null(mean={c['null_mean']:+.4f}, std={c['null_std']:.4f})  "
              f"p={c['p_value_two_tailed']:.4f}  -> {sig}")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    trend_path = out_dir / "layer3_40regime_trends.png"
    plot_stat_trends(positions, records, corr_tests, low_conf, ts, trend_path)
    print(f"\nSaved figure to {trend_path.relative_to(REPO_ROOT)}")

    table_rows = [
        {
            "index": i, "position": bounds[i][0], "length": records[i]["length"],
            "mean": records[i]["mean"], "variance": records[i]["variance"], "skew": records[i]["skew"],
            "kurtosis_internal": records[i]["kurtosis_excess"], "kurtosis_boundary": boundary_kurtosis[i],
            "spike_density_2s": records[i]["spike_density"]["2.0"], "low_confidence": records[i]["low_confidence"],
        }
        for i in range(40)
    ]
    table_path = out_dir / "layer3_40regime_table.png"
    plot_table_image(table_rows, table_path, ts)
    print(f"Saved figure to {table_path.relative_to(REPO_ROOT)}")

    json_out = {
        "timestamp": ts,
        "gaps_source": str(GAPS_CACHE_PATH.relative_to(REPO_ROOT)),
        "changepoints_source": str(CHANGEPOINTS_SOURCE.relative_to(REPO_ROOT)),
        "config": {
            "rolling_std_window": ROLLING_STD_WINDOW, "spike_thresholds": SPIKE_THRESHOLDS,
            "n_fft_peaks": N_FFT_PEAKS, "min_confident_size": MIN_CONFIDENT_SIZE,
            "n_perm_corr": N_PERM_CORR, "seed": SEED,
        },
        "n_regimes": 40,
        "n_low_confidence": int(n_low_conf),
        "correlation_tests": corr_tests,
        "regimes": [
            {
                "index": i, "bounds": bounds[i], "length": records[i]["length"],
                "mean": round(records[i]["mean"], 6), "variance": round(records[i]["variance"], 6),
                "skew": round(records[i]["skew"], 6), "kurtosis_excess_internal": round(records[i]["kurtosis_excess"], 6),
                "kurtosis_peak_boundary": boundary_kurtosis[i],
                "spike_density": records[i]["spike_density"], "spike_count": records[i]["spike_count"],
                "volatility_mean": round(records[i]["volatility"]["mean"], 6) if not np.isnan(records[i]["volatility"]["mean"]) else None,
                "volatility_trend_slope": round(records[i]["volatility"]["trend_slope"], 6) if not np.isnan(records[i]["volatility"]["trend_slope"]) else None,
                "volatility_trend_direction": records[i]["volatility"]["trend_direction"],
                "fft_top_peaks": [
                    {"period_gaps": round(p["period_gaps"], 3), "power": round(p["power"], 3)}
                    for p in records[i]["fft_peaks"]
                ],
                "low_confidence": bool(records[i]["low_confidence"]),
            }
            for i in range(40)
        ],
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(json_out, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    auto_commit_push(out_dir, corr_tests, int(n_low_conf), ts)


if __name__ == "__main__":
    main()
