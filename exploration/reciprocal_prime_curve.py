"""reciprocal_prime_curve.py

Quick look at the reciprocal-prime sequence 1/p_n over the first 20,000
primes: plots index vs. 1/p on both a linear and a log y-axis, side by side.

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
OUT_PATH = OUT_DIR / "reciprocal_prime_curve.png"


def load_primes() -> np.ndarray:
    with open(PRIMES_CACHE_PATH) as f:
        return np.array(json.load(f)["primes"])


def main() -> None:
    primes = load_primes()
    reciprocals = 1.0 / primes
    index = np.arange(len(primes))

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6))

    ax_lin.plot(index, reciprocals, lw=0.8, color="#4c72b0")
    ax_lin.set_xlabel("index")
    ax_lin.set_ylabel("1/p")
    ax_lin.set_title("Reciprocal primes (linear y-axis)")

    ax_log.plot(index, reciprocals, lw=0.8, color="#e08214")
    ax_log.set_yscale("log")
    ax_log.set_xlabel("index")
    ax_log.set_ylabel("1/p")
    ax_log.set_title("Reciprocal primes (log y-axis)")

    fig.suptitle(f"1/p over first {len(primes):,} primes")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
