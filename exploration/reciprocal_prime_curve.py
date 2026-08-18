"""reciprocal_prime_curve.py

Quick look at the reciprocal-prime sequence 1/p_n over the first 20,000
primes: index vs. 1/p on linear and log y-axes, plus the running sum
(cumulative area under the curve) and the discrete rate of change between
consecutive terms.

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

PRIMES_CACHE_PATH = Path("data/primes_20000.json")
OUT_DIR = Path("output/prime") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = OUT_DIR / "reciprocal_prime_analysis.png"


def load_primes() -> np.ndarray:
    with open(PRIMES_CACHE_PATH) as f:
        return np.array(json.load(f)["primes"])


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


if __name__ == "__main__":
    main()
