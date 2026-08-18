"""layer2_magnitude_test.py

Layer 2 magnitude test (hypotheses/second_order_gap_structure.md,
## Empirical Check -- Layer 2 Magnitude Test).

The two prior Layer 2 tests (proximity of gap-derivative zero-crossings to
the three known regime changepoints, raw and K=100-smoothed) both came
back inconclusive: zero-crossings are so dense in this data that a
fixed-radius proximity check has 100% base rate and can't discriminate
signal from chance.

This test asks a different, sharper question: is the *magnitude* of the
smoothed gap derivative unusually large at the three known changepoints,
compared to a null distribution built from randomly sampled windows in
the same sequence? Magnitude isn't ~50% dense the way zero-crossings are,
so this test can actually fail.

Run: python layer2_magnitude_test.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GAPS_CACHE_PATH = Path("data/primes_5000.json")
OUT_PATH = Path("output/prime/analysis/layer2_magnitude_test.png")
KNOWN_CHANGEPOINTS = [1529, 2501, 4211]
K = 100
N_NULL_SAMPLES = 10_000
SEED = 42


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def main() -> None:
    with open(GAPS_CACHE_PATH) as f:
        full_gaps = np.array(json.load(f)["gaps"])

    smoothed = rolling_mean(full_gaps, K)
    deriv = np.diff(smoothed)
    abs_deriv = np.abs(deriv)
    print(f"Smoothed (K={K}) derivative: {len(deriv)} points")

    cp_mags = {cp: float(abs_deriv[cp]) for cp in KNOWN_CHANGEPOINTS}
    print(f"Changepoint magnitudes: { {k: round(v, 5) for k, v in cp_mags.items()} }")

    rng = np.random.default_rng(SEED)
    null_idx = rng.integers(0, len(abs_deriv), size=N_NULL_SAMPLES)
    null_dist = abs_deriv[null_idx]
    print(f"Null distribution: {N_NULL_SAMPLES} samples, seed={SEED}, "
          f"mean={null_dist.mean():.5f}, std={null_dist.std():.5f}")

    percentiles = {}
    for cp, mag in cp_mags.items():
        pct = float(np.mean(null_dist <= mag) * 100)
        percentiles[cp] = pct
        print(f"  changepoint {cp}: |deriv|={mag:.5f} -> percentile {pct:.1f} in null distribution")

    all_above_90 = all(p >= 90 for p in percentiles.values())
    print(f"\nAll three above 90th percentile: {all_above_90}")
    if all_above_90:
        print("VERDICT: tentative confirmation -- changepoint magnitudes are unusually large.")
    else:
        print("VERDICT: Layer 2 magnitude hypothesis is refuted -- not all changepoints "
              "show unusually large derivative magnitude relative to the null distribution.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null_dist, bins=60, color="#4c72b0", alpha=0.7,
            label=f"null distribution (n={N_NULL_SAMPLES} random windows)")
    colors = ["#d1495b", "#e08214", "#2a9d5c"]
    for (cp, mag), color in zip(cp_mags.items(), colors, strict=True):
        ax.axvline(mag, color=color, lw=2, ls="--",
                   label=f"changepoint {cp}: |deriv|={mag:.4f} (p{percentiles[cp]:.0f})")
    p90 = float(np.percentile(null_dist, 90))
    ax.axvline(p90, color="black", lw=1, ls=":", label=f"90th percentile of null ({p90:.4f})")
    ax.set_xlabel("|smoothed gap derivative|")
    ax.set_ylabel("count (out of 10,000 null samples)")
    ax.set_title("Layer 2 magnitude test: changepoint derivative magnitude vs null distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
