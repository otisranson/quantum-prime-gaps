"""reciprocal_prime_curve.py

Quick look at the reciprocal-prime sequence 1/p_n over the first 20,000
primes: index vs. 1/p on linear and log y-axes, plus the running sum
(cumulative area under the curve) and the discrete rate of change between
consecutive terms. A second figure looks at the rolling variance of that
rate of change at two window sizes, and at the residual left after
dividing out the PNT-predicted (1/(n ln n))^2 decay envelope.

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

PRIMES_CACHE_PATH = Path("data/primes_20000.json")
OUT_DIR = Path("output/prime") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = OUT_DIR / "reciprocal_prime_analysis.png"
VARIANCE_OUT_PATH = OUT_DIR / "reciprocal_prime_variance.png"


def load_primes() -> np.ndarray:
    with open(PRIMES_CACHE_PATH) as f:
        return np.array(json.load(f)["primes"])


def rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling variance; output[i] covers x[i : i + window]."""
    return sliding_window_view(x, window).var(axis=1, ddof=0)


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


if __name__ == "__main__":
    main()
