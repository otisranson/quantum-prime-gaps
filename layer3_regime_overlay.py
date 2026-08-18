"""layer3_regime_overlay.py

Layer 3 internal-wave test (hypotheses/regime_internal_wave_structure.md,
"do the regimes rhyme?"): extract the raw prime-gap sequence within each of
the three regimes actually bounded by the three confirmed regime changepoints
(windows 1529, 2501, 4211 -- see hypotheses/second_order_gap_structure.md),
normalize each for length (resample) and amplitude (z-score), overlay them
raw, detrend each (remove its own linear trend) and overlay the residuals
separately, score similarity both ways, and compare against a random-slice
null of matching lengths.

Regime definition: three changepoints delimit exactly three *bounded*
segments when the sequence start is used as the first regime's implicit
start -- [0, 1529), [1529, 2501), [2501, 4211). The tail after the last
changepoint (4211-4999) is intentionally excluded: it has no closing
changepoint, so it isn't "bounded by the changepoints" the way the other
three are. Flagged again in the caveats section below.

Reads the same 5000-prime run other regime-change scripts in this repo use
(GAPS_CACHE_PATH below); reconstructs the continuous raw gap sequence from its
per-window data the same way smoothed_derivative_wave.py does.

Run: python layer3_regime_overlay.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GAPS_CACHE_PATH = Path("data/primes_5000.json")
OUT_PATH = Path("output/prime/analysis/layer3_regime_overlay.png")
JSON_OUT_PATH = Path("output/prime/analysis/layer3_regime_overlay.json")
KNOWN_CHANGEPOINTS = [1529, 2501, 4211]
N_RESAMPLE = 500
N_NULL_TRIALS = 5_000
SEED = 42
REGIME_COLORS = ["#4c72b0", "#e08214", "#2a9d5c"]


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        return np.array(json.load(f)["gaps"])


def resample_and_zscore(y: np.ndarray, n: int) -> np.ndarray:
    x_orig = np.linspace(0.0, 1.0, len(y))
    x_new = np.linspace(0.0, 1.0, n)
    y_rs = np.interp(x_new, x_orig, y)
    std = y_rs.std()
    return (y_rs - y_rs.mean()) / std if std > 0 else y_rs - y_rs.mean()


def detrend_linear(y: np.ndarray) -> np.ndarray:
    x = np.linspace(0.0, 1.0, len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def mean_pairwise_corr(curves: list[np.ndarray]) -> tuple[float, list[float]]:
    pairs = [(0, 1), (0, 2), (1, 2)]
    corrs = [float(np.corrcoef(curves[i], curves[j])[0, 1]) for i, j in pairs]
    return float(np.mean(corrs)), corrs


def main() -> None:
    full_gaps = load_full_gaps()
    print(f"Loaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH}")

    bounds = [(0, KNOWN_CHANGEPOINTS[0]), (KNOWN_CHANGEPOINTS[0], KNOWN_CHANGEPOINTS[1]),
              (KNOWN_CHANGEPOINTS[1], KNOWN_CHANGEPOINTS[2])]
    regimes_raw = [full_gaps[a:b] for a, b in bounds]
    lengths = [len(r) for r in regimes_raw]
    print(f"Regime bounds: {bounds}")
    print(f"Regime lengths: {lengths}")
    print(f"Excluded tail (not bounded by a closing changepoint): "
          f"[{KNOWN_CHANGEPOINTS[2]}, {len(full_gaps)}), length {len(full_gaps) - KNOWN_CHANGEPOINTS[2]}")

    regimes_norm = [resample_and_zscore(r, N_RESAMPLE) for r in regimes_raw]
    regimes_detrended = [detrend_linear(r) for r in regimes_norm]

    obs_raw_sim, obs_raw_pairs = mean_pairwise_corr(regimes_norm)
    obs_det_sim, obs_det_pairs = mean_pairwise_corr(regimes_detrended)
    print(f"\nObserved raw-normalized similarity (mean pairwise corr): {obs_raw_sim:.4f}  pairs={[round(c, 4) for c in obs_raw_pairs]}")
    print(f"Observed detrended similarity (mean pairwise corr):       {obs_det_sim:.4f}  pairs={[round(c, 4) for c in obs_det_pairs]}")

    # Mismatch-vs-regime-index check (n=3 -- see caveats).
    mean_det_curve = np.mean(regimes_detrended, axis=0)
    mismatch = [float(np.sqrt(np.mean((r - mean_det_curve) ** 2))) for r in regimes_detrended]
    regime_idx = np.array([0, 1, 2])
    mismatch_arr = np.array(mismatch)
    mismatch_corr = float(np.corrcoef(regime_idx, mismatch_arr)[0, 1]) if mismatch_arr.std() > 0 else 0.0
    print(f"\nPer-regime detrended-residual mismatch (RMS distance from cross-regime mean): "
          f"{[round(m, 4) for m in mismatch]}")
    print(f"Correlation of mismatch with regime index (n=3, treat with extreme caution): {mismatch_corr:.4f}")

    # Random-slice null: triplets of slices at random offsets, matching the
    # three real regime lengths, normalized/detrended the same way.
    rng = np.random.default_rng(SEED)
    null_raw = np.empty(N_NULL_TRIALS)
    null_det = np.empty(N_NULL_TRIALS)
    for t in range(N_NULL_TRIALS):
        slices = []
        for length in lengths:
            start = rng.integers(0, len(full_gaps) - length + 1)
            slices.append(full_gaps[start : start + length])
        norm_slices = [resample_and_zscore(s, N_RESAMPLE) for s in slices]
        det_slices = [detrend_linear(s) for s in norm_slices]
        null_raw[t], _ = mean_pairwise_corr(norm_slices)
        null_det[t], _ = mean_pairwise_corr(det_slices)

    pct_raw = float(np.mean(null_raw <= obs_raw_sim) * 100)
    pct_det = float(np.mean(null_det <= obs_det_sim) * 100)
    print(f"\nNull distribution (n={N_NULL_TRIALS}, seed={SEED}):")
    print(f"  raw:       mean={null_raw.mean():.4f}  std={null_raw.std():.4f}  observed percentile={pct_raw:.1f}")
    print(f"  detrended: mean={null_det.mean():.4f}  std={null_det.std():.4f}  observed percentile={pct_det:.1f}")

    # ── Plot: 2x2 -- raw overlay, detrended overlay, and both null histograms ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.linspace(0.0, 1.0, N_RESAMPLE)

    ax = axes[0, 0]
    for i, (curve, color) in enumerate(zip(regimes_norm, REGIME_COLORS, strict=True)):
        ax.plot(x, curve, color=color, lw=1.2, label=f"regime {i} (n={lengths[i]})")
    ax.set_title(f"Raw normalized overlay (length+amplitude) -- similarity={obs_raw_sim:.3f}")
    ax.set_xlabel("fractional position within regime")
    ax.set_ylabel("z-scored gap")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, (curve, color) in enumerate(zip(regimes_detrended, REGIME_COLORS, strict=True)):
        ax.plot(x, curve, color=color, lw=1.2, label=f"regime {i} (mismatch={mismatch[i]:.3f})")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_title(f"Detrended residual overlay -- similarity={obs_det_sim:.3f}")
    ax.set_xlabel("fractional position within regime")
    ax.set_ylabel("residual (z-scored, linear trend removed)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(null_raw, bins=60, color="#4c72b0", alpha=0.7, label=f"null (n={N_NULL_TRIALS} random triplets)")
    ax.axvline(obs_raw_sim, color="#d1495b", lw=2, ls="--", label=f"observed = {obs_raw_sim:.3f} (p{pct_raw:.0f})")
    ax.set_title("Raw similarity vs. random-slice null")
    ax.set_xlabel("mean pairwise correlation")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(null_det, bins=60, color="#4c72b0", alpha=0.7, label=f"null (n={N_NULL_TRIALS} random triplets)")
    ax.axvline(obs_det_sim, color="#d1495b", lw=2, ls="--", label=f"observed = {obs_det_sim:.3f} (p{pct_det:.0f})")
    ax.set_title("Detrended similarity vs. random-slice null")
    ax.set_xlabel("mean pairwise correlation")
    ax.legend(fontsize=8)

    fig.suptitle("Layer 3 -- regime internal-wave overlay and random-slice null")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved figure to {OUT_PATH}")

    results = {
        "results_source": str(GAPS_CACHE_PATH),
        "changepoints": KNOWN_CHANGEPOINTS,
        "regime_bounds": bounds,
        "regime_lengths": lengths,
        "excluded_tail_bounds": [KNOWN_CHANGEPOINTS[2], len(full_gaps)],
        "n_resample": N_RESAMPLE,
        "observed": {
            "raw_similarity": round(obs_raw_sim, 6),
            "raw_pairs": [round(c, 6) for c in obs_raw_pairs],
            "detrended_similarity": round(obs_det_sim, 6),
            "detrended_pairs": [round(c, 6) for c in obs_det_pairs],
        },
        "mismatch": {
            "per_regime_rms_from_mean": [round(m, 6) for m in mismatch],
            "correlation_with_regime_index_n3": round(mismatch_corr, 6),
        },
        "null": {
            "n_trials": N_NULL_TRIALS,
            "seed": SEED,
            "raw_mean": round(float(null_raw.mean()), 6),
            "raw_std": round(float(null_raw.std()), 6),
            "raw_observed_percentile": round(pct_raw, 2),
            "detrended_mean": round(float(null_det.mean()), 6),
            "detrended_std": round(float(null_det.std()), 6),
            "detrended_observed_percentile": round(pct_det, 2),
        },
    }
    JSON_OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {JSON_OUT_PATH}")


if __name__ == "__main__":
    main()
