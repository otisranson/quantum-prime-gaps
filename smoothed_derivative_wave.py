"""smoothed_derivative_wave.py

Plain visualization of the K=100-smoothed gap derivative as a continuous
wave, with the three known regime changepoints marked. No statistics, no
null distribution -- just the wave.

Run: python smoothed_derivative_wave.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("output/prime/20260816_010716/terrain_5000primes/results_5000primes.json")
OUT_PATH = Path("output/prime/analysis/smoothed_derivative_wave.png")
KNOWN_CHANGEPOINTS = [1529, 2501, 4211]
K = 100


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def main() -> None:
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    per_window = data["per_window"]
    full_gaps = np.array([r["gaps"][0] for r in per_window] + per_window[-1]["gaps"][1:])

    smoothed = rolling_mean(full_gaps, K)
    deriv = np.diff(smoothed)
    prime_index = np.arange(1, len(deriv) + 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(prime_index, deriv, color="#4c72b0", lw=1)
    ax.axhline(0, color="#888888", lw=1)

    for cp in KNOWN_CHANGEPOINTS:
        ax.axvline(cp, color="#d1495b", lw=1.5, ls="--")
        ax.text(cp, ax.get_ylim()[1], f" {cp}", color="#d1495b", va="top", ha="left", fontsize=9)

    ax.set_xlim(1, 5000)
    ax.set_xlabel("prime index")
    ax.set_ylabel("smoothed gap derivative (K=100)")
    ax.set_title("Smoothed gap derivative — 5000-prime run")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
