"""gap_derivative_zero_crossings.py

Empirical check for the Layer 2 claim in
hypotheses/second_order_gap_structure.md: that regime changes correspond
to zero-crossings of the prime gap derivative (Dimension 2).

Reconstructs the full prime-gap sequence from the 5000-prime terrain run,
computes the first difference (gap[n+1] - gap[n]), finds sign changes, and
checks proximity to the three known regime-changepoint windows (1529,
2501, 4211) from regime_fit_5k.py.

Run: python gap_derivative_zero_crossings.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GAPS_CACHE_PATH = Path("data/primes_5000.json")
OUT_PATH = Path("output/prime/analysis/gap_derivative_zero_crossings.png")
KNOWN_CHANGEPOINTS = [1529, 2501, 4211]
PROXIMITY = 50


def main() -> None:
    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    full_gaps = np.array(cache["gaps"])
    assert len(full_gaps) == cache["n_gaps"]

    # Step 1: derivative (Dimension 2)
    deriv = np.diff(full_gaps)

    # Step 2: zero-crossings (sign changes, ignoring exact-zero derivative steps)
    signs = np.sign(deriv)
    crossings = []
    last_sign = 0
    for i, s in enumerate(signs):
        if s == 0:
            continue
        if last_sign != 0 and s != last_sign:
            crossings.append(i)
        last_sign = s
    crossings = np.array(crossings)

    density = len(crossings) / len(deriv)
    print(f"Gap sequence length: {len(full_gaps)}")
    print(f"Derivative length: {len(deriv)}")
    print(f"Zero-crossings found: {len(crossings)} / {len(deriv)} indices (density={density:.3f})")

    # Step 3: proximity to known changepoints
    print(f"\nProximity check (within {PROXIMITY} windows):")
    for cp in KNOWN_CHANGEPOINTS:
        dists = np.abs(crossings - cp)
        nearest_idx = np.argmin(dists)
        nearest = crossings[nearest_idx]
        n_within = np.sum(dists <= PROXIMITY)
        print(f"  changepoint {cp}: nearest zero-crossing at {nearest} "
              f"(distance {dists[nearest_idx]}), {n_within} crossings within +/-{PROXIMITY}")

    # Statistical control: at this zero-crossing density, what fraction of
    # ALL indices (not just the 3 changepoints) land within PROXIMITY of a
    # zero-crossing? If that fraction is already ~1, "proximity" to the
    # changepoints is not informative -- it would happen almost everywhere.
    # vectorized: for each crossing, mark +/- PROXIMITY window as covered
    covered = np.zeros(len(deriv) + 1, dtype=int)
    for c in crossings:
        lo = max(0, c - PROXIMITY)
        hi = min(len(deriv), c + PROXIMITY + 1)
        covered[lo] += 1
        covered[hi] -= 1
    coverage = np.cumsum(covered)[:-1] > 0
    base_rate = coverage.mean()
    print(f"\nBase rate: {base_rate:.3f} of ALL indices in [0,{len(deriv)}) are within "
          f"+/-{PROXIMITY} of *some* zero-crossing (density={density:.3f}).")
    print("This is the chance level the 3-changepoint proximity result must be judged against.")

    # Step 4: plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(deriv, color="#4c72b0", lw=0.5, alpha=0.6, label="gap derivative (Dimension 2)")
    ax.scatter(crossings, deriv[crossings], color="#2a9d5c", s=4, alpha=0.4,
               label=f"zero-crossings (n={len(crossings)}, density={density:.2f})")
    for cp in KNOWN_CHANGEPOINTS:
        ax.axvline(cp, color="#d1495b", ls="--", lw=1.5)
    ax.axvline(KNOWN_CHANGEPOINTS[0], color="#d1495b", ls="--", lw=1.5,
               label="known regime changepoints (1529, 2501, 4211)")
    ax.set_xlabel("gap index (~window number)")
    ax.set_ylabel("gap[n+1] - gap[n]")
    ax.set_title(f"Gap derivative zero-crossings vs known regime changepoints\n"
                 f"(zero-crossing base rate within +/-{PROXIMITY}: {base_rate:.0%} of all indices)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
