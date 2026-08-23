"""layer3_log_detrend_residual.py

Implements the "Next Session: Log-Detrended Residual Analysis" objective
recorded in CLAUDE.md (added 2026-08-18, left unstarted through the following
session's exploratory detour into 1/p and log-polar reframings).

**Motivation, restated from that note:** the 40-Regime Characterization
(hypotheses/regime_internal_wave_structure.md) confirmed mean gap and rolling
variance both climb with position via a real, permutation-significant
log-scale trend (PNT-driven, r=0.85 / r=0.88, p<0.0001). Every wave/shape
analysis run so far (regime overlay, kurtosis robustness, cross-regime
self-similarity) operated on raw or per-regime-normalized data -- if real
periodic/self-similar structure exists, it could be masked by this trend
still being present in the signal. This script removes the trend first, then
reruns the standard structure-detection toolkit (FFT, autocorrelation,
kurtosis scan) on what's left.

**Method:**
1. Rolling mean and rolling std (K=100, leading-window convention, identical
   to layer3_full_sequence_overview.py's rolling_std) over the full
   20,000-prime gap sequence.
2. Fit each to a*ln(center+2)+b (same convention as layer3_pi_coefficient_test.py
   and gap_entropy_windows.py) and subtract to get two residual series:
   mean_residual(N) and std_residual(N).
3. On each residual: FFT magnitude spectrum (mean-subtracted before the
   transform, since a residual should already be ~zero-mean but this is
   applied explicitly rather than assumed), autocorrelation out to lag 200,
   and sliding-window excess kurtosis (width=500, step=100, same parameters
   as layer3_kurtosis_robustness.py's sliding_window_kurtosis).
4. Every "is this a peak/departure" claim is checked against a permutation
   null rather than eyeballed -- this repo's standing discipline after
   several prior checks in this file turned out to have no discriminating
   power without one (e.g. the two Layer 2 zero-crossing proximity tests in
   hypotheses/second_order_gap_structure.md).

**Methodological catch, caught and fixed before this script's first commit
(same failure class as the Layer 2 checks above):** a first pass used a full
i.i.d. shuffle as the null for both the FFT-peak and autocorrelation tests.
That null returned a "moderate confidence, significant" verdict on both
residuals at effectively p=0 -- but the underlying cause was the null itself
having no discriminating power, not real structure. The rolling mean/std
series is built with a step=1, K=100 *overlapping* window, so value[i] and
value[i+1] share 99 of their 100 underlying raw points by construction; this
alone produces a triangular, boxcar-induced autocorrelation approaching 1 at
lag 1 and decaying to ~0 only past lag K, with zero dependence on any real
signal in the data. A full shuffle destroys that trivial construction
artifact along with everything else, so *any* smoothed series -- structured
or not -- looks "significant" against it. The null used below is a
**block-shuffle** instead: the residual is cut into non-overlapping blocks of
size BLOCK_SIZE=2*ROLLING_WINDOW (200, comfortably larger than the K=100
boxcar decay length) and block *order* is randomly permuted, block *contents*
left intact. This preserves the trivial short-range correlation inside each
block (present identically in every null draw, so it no longer drives
significance) while still destroying any genuine longer-range periodic or
autocorrelated arrangement across blocks -- an honest test of whether
structure exists *beyond* the windowing artifact.

Run: python layer3_log_detrend_residual.py
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
OUT_ROOT = REPO_ROOT / "output" / "prime"

ROLLING_WINDOW = 100
AUTOCORR_MAX_LAG = 200
KURT_WIDTH = 500
KURT_STEP = 100
N_PERM = 2000
SEED = 42
BLOCK_SIZE = 2 * ROLLING_WINDOW  # null-preserving block size, see module docstring


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    gaps = np.array(cache["gaps"])
    assert len(gaps) == cache["n_gaps"]
    return gaps


def rolling_mean_std(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x ** 2, 0, 0.0))
    mean = (c1[k:] - c1[:-k]) / k
    mean_sq = (c2[k:] - c2[:-k]) / k
    std = np.sqrt(np.clip(mean_sq - mean ** 2, 0.0, None))
    return mean, std


def fit_log_linear(centers: np.ndarray, values: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    x = np.log(centers + 2)
    a, b = np.polyfit(x, values, 1)
    predicted = a * x + b
    residuals = values - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), predicted, r_squared


def fft_magnitude(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean()
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    mag = np.abs(spectrum)
    return freqs[1:], mag[1:]  # drop DC bin


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    centered = x - x.mean()
    n = len(centered)
    var = np.dot(centered, centered) / n
    ac = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        ac[lag] = np.dot(centered[:n - lag], centered[lag:]) / n / var
    return ac


def sliding_window_kurtosis(x: np.ndarray, width: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(x) - width + 1, step)
    kurt = np.empty(len(starts))
    for i, s in enumerate(starts):
        w = x[s:s + width]
        mean, std = w.mean(), w.std()
        kurt[i] = 0.0 if std == 0 else np.mean((w - mean) ** 4) / std ** 4 - 3.0
    centers = starts + width / 2
    return centers, kurt


def block_shuffle(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Cut x into non-overlapping blocks of block_size (dropping any
    remainder so every block is full-length) and randomly permute block
    *order*, leaving each block's internal contents untouched. Preserves
    within-block short-range correlation (here: the K=100 boxcar windowing
    artifact) while destroying longer-range structure -- see module
    docstring's Methodological catch section for why this replaces a naive
    i.i.d. shuffle."""
    n_blocks = len(x) // block_size
    blocks = x[:n_blocks * block_size].reshape(n_blocks, block_size)
    order = rng.permutation(n_blocks)
    return blocks[order].reshape(-1)


def permutation_peak_test(x: np.ndarray, observed_peak_mag: float, n_perm: int, rng: np.random.Generator) -> dict:
    """Block-shuffle null (see block_shuffle) -- recompute the max FFT
    magnitude (excluding DC) each trial and report where the observed top
    peak sits relative to that null."""
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = block_shuffle(x, BLOCK_SIZE, rng)
        spectrum = np.fft.rfft(shuffled - shuffled.mean())
        mag = np.abs(spectrum)[1:]
        null_max[i] = mag.max()
    p_value = float(np.mean(null_max >= observed_peak_mag))
    return {"null_mean": float(null_max.mean()), "null_std": float(null_max.std()), "p_value": p_value}


def permutation_autocorr_test(x: np.ndarray, observed_ac: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """Block-shuffle null (see block_shuffle) -- recompute autocorrelation at
    lags 1..max_lag, take the max |autocorrelation| across those lags each
    trial, tests whether the single largest non-zero-lag autocorrelation in
    the real series exceeds what block-shuffling (which preserves the
    trivial windowing-induced short-range correlation) alone would produce."""
    observed_max_abs = float(np.max(np.abs(observed_ac[1:])))
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = block_shuffle(x, BLOCK_SIZE, rng)
        n = len(shuffled)
        centered = shuffled - shuffled.mean()
        var = np.dot(centered, centered) / n
        max_abs = 0.0
        for lag in range(1, len(observed_ac)):
            ac = np.dot(centered[:n - lag], centered[lag:]) / n / var
            max_abs = max(max_abs, abs(ac))
        null_max[i] = max_abs
    p_value = float(np.mean(null_max >= observed_max_abs))
    null_mean = float(null_max.mean())
    null_std = float(null_max.std())
    z_score = float((observed_max_abs - null_mean) / null_std) if null_std > 0 else float("inf")
    rel_effect_pct = float((observed_max_abs - null_mean) / null_mean * 100.0) if null_mean != 0 else float("inf")
    return {"observed_max_abs_lag1plus": observed_max_abs, "null_mean": null_mean, "null_std": null_std,
            "p_value": p_value, "z_score": z_score, "relative_effect_pct": rel_effect_pct}


def analyze_residual(name: str, residual: np.ndarray, rng: np.random.Generator) -> dict:
    freqs, mag = fft_magnitude(residual)
    top_idx = int(np.argmax(mag))
    top_peak = {"frequency": float(freqs[top_idx]), "period": float(1.0 / freqs[top_idx]), "magnitude": float(mag[top_idx])}
    fft_perm = permutation_peak_test(residual, top_peak["magnitude"], N_PERM, rng)

    ac = autocorrelation(residual, AUTOCORR_MAX_LAG)
    ac_perm = permutation_autocorr_test(residual, ac, N_PERM, rng)

    kurt_centers, kurt_vals = sliding_window_kurtosis(residual, KURT_WIDTH, KURT_STEP)

    return {
        "name": name, "freqs": freqs, "mag": mag, "top_peak": top_peak, "fft_perm": fft_perm,
        "autocorr": ac, "ac_perm": ac_perm, "kurt_centers": kurt_centers, "kurt_vals": kurt_vals,
    }


def plot_all(mean_res: dict, std_res: dict, ts: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for col, res in enumerate([mean_res, std_res]):
        ax_fft, ax_ac, ax_kurt = axes[0, col], axes[1, col], axes[2, col]

        ax_fft.plot(res["freqs"], res["mag"], color="#4c72b0", lw=0.8)
        ax_fft.axvline(res["top_peak"]["frequency"], color="#d1495b", lw=1.0, ls="--",
                        label=f"top peak, period={res['top_peak']['period']:.1f}, p={res['fft_perm']['p_value']:.3f}")
        ax_fft.set_title(f"{res['name']} residual -- FFT magnitude spectrum")
        ax_fft.set_xlabel("frequency (cycles/gap-index)")
        ax_fft.legend(fontsize=8)

        lags = np.arange(len(res["autocorr"]))
        ax_ac.plot(lags, res["autocorr"], color="#e08214", lw=1.0)
        ax_ac.axhline(0, color="#888888", lw=0.6)
        ax_ac.set_title(f"{res['name']} residual -- autocorrelation "
                         f"(max|r| lag>=1: {res['ac_perm']['observed_max_abs_lag1plus']:.3f} vs. "
                         f"block-shuffle null {res['ac_perm']['null_mean']:.3f}, p={res['ac_perm']['p_value']:.3f}, "
                         f"effect={res['ac_perm']['relative_effect_pct']:.1f}%)")
        ax_ac.set_xlabel("lag")

        ax_kurt.plot(res["kurt_centers"], res["kurt_vals"], color="#2a9d5c", lw=1.2)
        ax_kurt.axhline(0, color="#888888", lw=0.6, ls=":")
        ax_kurt.set_title(f"{res['name']} residual -- sliding-window excess kurtosis "
                           f"(width={KURT_WIDTH}, step={KURT_STEP})")
        ax_kurt.set_xlabel("gap index (window center)")

    fig.suptitle(f"Log-detrended residual structure scan [{ts}]", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "layer3_log_detrend_residual.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path.relative_to(REPO_ROOT)}")


AC_EFFECT_SIZE_THRESHOLD_PCT = 2.0  # relative effect below this = "significant but negligible"


def confidence_label(fft_p: float, ac_p: float, ac_rel_effect_pct: float) -> str:
    """p-values alone aren't enough here: the autocorrelation null's variance
    is extremely tight (block-shuffle preserves nearly all the trivial
    windowing-induced correlation), so a real but tiny departure from it can
    read as p<0.0001 while being practically meaningless. Relative effect
    size (observed vs. null mean, as a percentage) is checked explicitly
    before calling the autocorrelation leg of this test a "confidence"
    contributor -- same discipline as the CLAUDE.md session-handoff note on
    separating statistical significance from practical effect size."""
    ac_negligible = ac_p < 0.05 and abs(ac_rel_effect_pct) < AC_EFFECT_SIZE_THRESHOLD_PCT
    ac_meaningful_sig = ac_p < 0.05 and not ac_negligible
    if fft_p < 0.05 and ac_meaningful_sig:
        return "moderate -- both FFT-peak and autocorrelation significant with non-negligible effect size"
    if fft_p < 0.05 or ac_meaningful_sig:
        return "weak -- one test significant with non-negligible effect size, not both"
    if ac_negligible and fft_p >= 0.05:
        return (f"none -- FFT peak not significant; autocorrelation is statistically significant "
                 f"(p<0.05) but the effect size is negligible (<{AC_EFFECT_SIZE_THRESHOLD_PCT}% relative to "
                 "null mean), consistent with the block-shuffle null's very low variance rather than "
                 "real structure")
    return "none -- neither test distinguishes the residual from a block-shuffled null"


def main() -> None:
    rng = np.random.default_rng(SEED)
    full_gaps = load_full_gaps()
    print(f"Loaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")

    mean_series, std_series = rolling_mean_std(full_gaps, ROLLING_WINDOW)
    centers = np.arange(len(mean_series)) + ROLLING_WINDOW / 2

    a_mean, b_mean, fit_mean, r2_mean = fit_log_linear(centers, mean_series)
    a_std, b_std, fit_std, r2_std = fit_log_linear(centers, std_series)
    print(f"\nmean(N) fit: {a_mean:.6f} * ln(N+2) + {b_mean:.6f}  (R^2={r2_mean:.4f})")
    print(f"std(N)  fit: {a_std:.6f} * ln(N+2) + {b_std:.6f}  (R^2={r2_std:.4f})")

    mean_residual = mean_series - fit_mean
    std_residual = std_series - fit_std

    print("\nAnalyzing mean residual...")
    mean_res = analyze_residual("mean", mean_residual, rng)
    print("Analyzing std residual...")
    std_res = analyze_residual("std", std_residual, rng)

    for res in (mean_res, std_res):
        print(f"\n{res['name']} residual:")
        print(f"  Top FFT peak: period={res['top_peak']['period']:.2f}, "
              f"mag={res['top_peak']['magnitude']:.4f}, permutation p={res['fft_perm']['p_value']:.4f}")
        print(f"  Max |autocorr| (lag>=1): {res['ac_perm']['observed_max_abs_lag1plus']:.4f} "
              f"(null mean={res['ac_perm']['null_mean']:.4f}, std={res['ac_perm']['null_std']:.6f}), "
              f"permutation p={res['ac_perm']['p_value']:.4f}, "
              f"relative effect={res['ac_perm']['relative_effect_pct']:.2f}%")
        label = confidence_label(res["fft_perm"]["p_value"], res["ac_perm"]["p_value"], res["ac_perm"]["relative_effect_pct"])
        print(f"  Confidence: {label}")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_all(mean_res, std_res, ts, out_dir)

    results = {
        "timestamp": ts,
        "gaps_source": str(GAPS_CACHE_PATH.relative_to(REPO_ROOT)),
        "config": {"rolling_window": ROLLING_WINDOW, "autocorr_max_lag": AUTOCORR_MAX_LAG,
                   "kurt_width": KURT_WIDTH, "kurt_step": KURT_STEP, "n_perm": N_PERM, "seed": SEED},
        "log_fits": {
            "mean": {"a": round(a_mean, 6), "b": round(b_mean, 6), "r_squared": round(r2_mean, 6)},
            "std": {"a": round(a_std, 6), "b": round(b_std, 6), "r_squared": round(r2_std, 6)},
        },
        "residual_analysis": {
            res["name"]: {
                "top_fft_peak": res["top_peak"],
                "fft_permutation_test": res["fft_perm"],
                "autocorr_permutation_test": res["ac_perm"],
                "confidence": confidence_label(res["fft_perm"]["p_value"], res["ac_perm"]["p_value"], res["ac_perm"]["relative_effect_pct"]),
                "kurtosis_scan_range": [float(res["kurt_vals"].min()), float(res["kurt_vals"].max())],
                "kurtosis_scan_mean": float(res["kurt_vals"].mean()),
            }
            for res in (mean_res, std_res)
        },
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {json_path.relative_to(REPO_ROOT)}")

    mean_conf = confidence_label(mean_res["fft_perm"]["p_value"], mean_res["ac_perm"]["p_value"], mean_res["ac_perm"]["relative_effect_pct"]).split(" --")[0]
    std_conf = confidence_label(std_res["fft_perm"]["p_value"], std_res["ac_perm"]["p_value"], std_res["ac_perm"]["relative_effect_pct"]).split(" --")[0]
    msg = (f"experiment: log-detrend residual structure scan {ts} -- "
           f"mean-residual confidence={mean_conf}, std-residual confidence={std_conf}, "
           f"mean fit R^2={r2_mean:.3f}, std fit R^2={r2_std:.3f}")
    subprocess.run(["git", "add", str(out_dir.relative_to(REPO_ROOT))], check=True, cwd=REPO_ROOT)
    commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        print(f"\n  Committed: {out_dir.relative_to(REPO_ROOT)}")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"\n  Git commit skipped: {commit.stdout.strip()}")


if __name__ == "__main__":
    main()
