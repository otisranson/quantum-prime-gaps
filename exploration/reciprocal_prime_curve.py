"""reciprocal_prime_curve.py

Quick look at the reciprocal-prime sequence 1/p_n over the first 20,000
primes: index vs. 1/p on linear and log y-axes, plus the running sum
(cumulative area under the curve) and the discrete rate of change between
consecutive terms. A second figure looks at the rolling variance of that
rate of change at two window sizes, and at the residual left after
dividing out the PNT-predicted (1/(n ln n))^2 decay envelope.

A third figure fits f(n) = a/(n ln n) to 1/p_n (PNT-motivated, scaling
constant a free), then inverts the fit to compare predicted vs. actual
prime *positions* -- p_n_actual - n*ln(n)/a -- and normalizes that
position residual by n*ln(n) to check whether the log-scale fit fully
explains prime position or leaves real structure behind.

A fourth figure compares actual gaps p_{n+1}-p_n against the derivative
of n*ln(n), plots the gap prediction error, and checks that error's
autocorrelation out to lag 100 for leftover memory/structure.

Reads data/primes_20000.json (the existing 20k-prime cache built by
build_prime_cache.py) -- does not regenerate primes.

Run: python exploration/reciprocal_prime_curve.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit

PRIMES_CACHE_PATH = Path("data/primes_20000.json")
OUT_DIR = Path("output/prime") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = OUT_DIR / "reciprocal_prime_analysis.png"
VARIANCE_OUT_PATH = OUT_DIR / "reciprocal_prime_variance.png"
RESIDUALS_OUT_PATH = OUT_DIR / "reciprocal_prime_residuals.png"
GAP_PREDICTION_OUT_PATH = OUT_DIR / "reciprocal_prime_gap_prediction.png"


def load_primes() -> np.ndarray:
    with open(PRIMES_CACHE_PATH) as f:
        return np.array(json.load(f)["primes"])


def rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling variance; output[i] covers x[i : i + window]."""
    return sliding_window_view(x, window).var(axis=1, ddof=0)


def reciprocal_model(n: np.ndarray, a: float) -> np.ndarray:
    return a / (n * np.log(n))


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample ACF via Pearson correlation at each lag; acf[0] == 1.0."""
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    return acf


def main() -> None:
    primes = load_primes()
    reciprocals = 1.0 / primes
    index = np.arange(len(primes))

    cumulative = np.cumsum(reciprocals)
    rate_of_change = np.abs(np.diff(reciprocals))

    fig, ((ax_lin, ax_log), (ax_cum, ax_rate)) = plt.subplots(2, 2, figsize=(14, 12))

    ax_lin.plot(index, reciprocals, lw=0.8, color="#4c72b0")
    ax_lin.set_xlabel("index")
    ax_lin.set_ylabel("1/p")
    ax_lin.set_title("Reciprocal primes (linear y-axis)")

    ax_log.plot(index, reciprocals, lw=0.8, color="#e08214")
    ax_log.set_yscale("log")
    ax_log.set_xlabel("index")
    ax_log.set_ylabel("1/p")
    ax_log.set_title("Reciprocal primes (log y-axis)")

    # index+1 so the first point (index 0) is representable on a log x-axis
    ax_cum.plot(index + 1, cumulative, lw=0.8, color="#2a9d5c")
    ax_cum.set_xscale("log")
    ax_cum.set_xlabel("index (log)")
    ax_cum.set_ylabel(r"$\sum 1/p$")
    ax_cum.set_title("Cumulative sum of 1/p")

    ax_rate.plot(index[:-1], rate_of_change, lw=0.8, color="#c44e52")
    ax_rate.set_yscale("log")
    ax_rate.set_xlabel("index")
    ax_rate.set_ylabel(r"$|\Delta(1/p)|$")
    ax_rate.set_title("Rate of change between consecutive 1/p")

    fig.suptitle(f"1/p analysis over first {len(primes):,} primes")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH}")

    # n=1-based position of each rolling window's last element in rate_of_change,
    # which itself is aligned to primes[1:] -- so n ranges over prime index, never 0
    n_100 = np.arange(100, len(rate_of_change) + 1)
    n_500 = np.arange(500, len(rate_of_change) + 1)
    var_100 = rolling_variance(rate_of_change, 100)
    var_500 = rolling_variance(rate_of_change, 500)

    envelope_100 = (1.0 / (n_100 * np.log(n_100))) ** 2
    residual_100 = var_100 / envelope_100

    fig2, (ax_var, ax_resid) = plt.subplots(1, 2, figsize=(14, 6))

    ax_var.plot(n_100, var_100, lw=0.8, color="#4c72b0", label="K=100")
    ax_var.plot(n_500, var_500, lw=0.8, color="#e08214", label="K=500")
    ax_var.set_yscale("log")
    ax_var.set_xlabel("index")
    ax_var.set_ylabel(r"rolling variance of $|\Delta(1/p)|$")
    ax_var.set_title("Rolling variance of rate of change")
    ax_var.legend()

    ax_resid.plot(n_100, residual_100, lw=0.8, color="#c44e52")
    ax_resid.set_yscale("log")
    ax_resid.set_xlabel("index")
    ax_resid.set_ylabel(r"var / $(1/(n \ln n))^2$")
    ax_resid.set_title("K=100 variance normalized by PNT envelope")

    fig2.suptitle(f"Rate-of-change variance over first {len(primes):,} primes")
    fig2.tight_layout()

    fig2.savefig(VARIANCE_OUT_PATH, dpi=150)
    print(f"Saved plot to {VARIANCE_OUT_PATH}")

    # n=1 (p_1=2) is excluded: ln(1)=0 makes a/(n ln n) undefined there
    n = (index + 1)[1:]
    p_actual = primes[1:]
    recip_actual = reciprocals[1:]

    (a,), _ = curve_fit(reciprocal_model, n, recip_actual, p0=[1.0])
    recip_fitted = reciprocal_model(n, a)

    p_predicted = n * np.log(n) / a
    position_residual = p_actual - p_predicted
    normalized_residual = position_residual / (n * np.log(n))

    fig3, (ax_fit, ax_resid, ax_norm) = plt.subplots(1, 3, figsize=(20, 6))

    ax_fit.plot(n, recip_actual, lw=0.8, color="#4c72b0", label="actual 1/p")
    ax_fit.plot(n, recip_fitted, lw=1.2, color="#c44e52", label=f"fit: a/(n ln n), a={a:.4f}")
    ax_fit.set_yscale("log")
    ax_fit.set_xlabel("n")
    ax_fit.set_ylabel("1/p")
    ax_fit.set_title("Curve fit: a/(n ln n)")
    ax_fit.legend()

    ax_resid.plot(n, position_residual, lw=0.8, color="#2a9d5c")
    ax_resid.axhline(0, color="black", lw=0.6, ls="--")
    ax_resid.set_xlabel("n")
    ax_resid.set_ylabel(r"$p_n^{actual} - p_n^{predicted}$")
    ax_resid.set_title("Position residual")

    ax_norm.plot(n, normalized_residual, lw=0.8, color="#e08214")
    ax_norm.axhline(0, color="black", lw=0.6, ls="--")
    ax_norm.set_xlabel("n")
    ax_norm.set_ylabel(r"residual / $(n \ln n)$")
    ax_norm.set_title("Normalized position residual")

    fig3.suptitle(f"Log-fit position residuals over first {len(primes):,} primes")
    fig3.tight_layout()

    fig3.savefig(RESIDUALS_OUT_PATH, dpi=150)
    print(f"Saved plot to {RESIDUALS_OUT_PATH}")

    n_gap = np.arange(1, len(primes))
    predicted_gap = (n_gap + 1) * np.log(n_gap + 1) - n_gap * np.log(n_gap)
    actual_gap = primes[1:] - primes[:-1]
    gap_error = actual_gap - predicted_gap

    max_lag = 100
    acf = autocorrelation(gap_error, max_lag)

    fig4, (ax_gap, ax_err, ax_acf) = plt.subplots(1, 3, figsize=(20, 6))

    ax_gap.plot(n_gap, actual_gap, lw=0.8, color="#4c72b0", alpha=0.3, label="actual gap")
    ax_gap.plot(
        n_gap,
        predicted_gap,
        lw=1.2,
        color="#c44e52",
        label=r"predicted: $(n+1)\ln(n+1) - n\ln n$",
    )
    ax_gap.set_xlabel("n")
    ax_gap.set_ylabel("gap")
    ax_gap.set_title("Predicted vs actual gap")
    ax_gap.legend()

    ax_err.plot(n_gap, gap_error, lw=0.5, color="#2a9d5c")
    ax_err.axhline(0, color="black", lw=0.6, ls="--")
    ax_err.set_xlabel("n")
    ax_err.set_ylabel(r"$g_n^{actual} - g_n^{predicted}$")
    ax_err.set_title("Gap prediction error")

    lags = np.arange(max_lag + 1)
    ax_acf.bar(lags, acf, color="#e08214", width=0.8)
    ax_acf.axhline(0, color="black", lw=0.6)
    ax_acf.set_xlabel("lag")
    ax_acf.set_ylabel("autocorrelation")
    ax_acf.set_title("Autocorrelation of gap prediction error")

    fig4.suptitle(f"Gap prediction over first {len(primes):,} primes")
    fig4.tight_layout()

    fig4.savefig(GAP_PREDICTION_OUT_PATH, dpi=150)
    print(f"Saved plot to {GAP_PREDICTION_OUT_PATH}")


if __name__ == "__main__":
    main()
