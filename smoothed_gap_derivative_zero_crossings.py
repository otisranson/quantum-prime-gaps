"""smoothed_gap_derivative_zero_crossings.py

Revised Layer 2 empirical check (hypotheses/second_order_gap_structure.md,
## Empirical Check -- Layer 2 Revised).

The raw gap-derivative test (gap_derivative_zero_crossings.py) failed with
zero discriminating power: zero-crossings covered 100% of all indices
within +/-50 windows, so any 3 points at all would have "passed."

This revised test smooths the gap sequence first (K=100 rolling mean, the
same K used to find the three known regime changepoints via MI
changepoint detection), then looks for zero-crossings in the derivative
of THAT smoothed sequence -- these mark local peaks/troughs of the
smoothed gap curve, and should be much sparser than raw-derivative
crossings. The base rate is computed explicitly before the proximity
test is evaluated, so the test can only claim confirmation if it beats
chance by a stated margin.

Run: python smoothed_gap_derivative_zero_crossings.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("output/prime/20260816_010716/terrain_5000primes/results_5000primes.json")
OUT_PATH = Path("output/prime/analysis/smoothed_gap_derivative_zero_crossings.png")
KNOWN_CHANGEPOINTS = [1529, 2501, 4211]
PROXIMITY = 50
K = 100


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def find_zero_crossings(deriv: np.ndarray) -> np.ndarray:
    signs = np.sign(deriv)
    crossings = []
    last_sign = 0
    for i, s in enumerate(signs):
        if s == 0:
            continue
        if last_sign != 0 and s != last_sign:
            crossings.append(i)
        last_sign = s
    return np.array(crossings)


def coverage_base_rate(n: int, crossings: np.ndarray, proximity: int) -> np.ndarray:
    """Boolean array: for each index in [0,n), is it within +/-proximity of some crossing?"""
    covered = np.zeros(n + 1, dtype=int)
    for c in crossings:
        lo = max(0, c - proximity)
        hi = min(n, c + proximity + 1)
        covered[lo] += 1
        covered[hi] -= 1
    return np.cumsum(covered)[:-1] > 0


def main() -> None:
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    per_window = data["per_window"]
    full_gaps = np.array([r["gaps"][0] for r in per_window] + per_window[-1]["gaps"][1:])

    # Step 1: smoothed gap sequence (K=100, same K as the MI changepoint detection)
    smoothed = rolling_mean(full_gaps, K)
    print(f"Smoothed gap sequence length: {len(smoothed)} (K={K})")

    # Step 2: derivative of the smoothed sequence
    deriv = np.diff(smoothed)
    print(f"Smoothed derivative length: {len(deriv)}")

    # Step 3: zero-crossings of the smoothed derivative
    crossings = find_zero_crossings(deriv)
    density = len(crossings) / len(deriv)
    print(f"Zero-crossings found: {len(crossings)} / {len(deriv)} indices (density={density:.3f})")
    if density > 0.65:
        print("NOTE: this is NOT sparser than the raw-derivative density (0.653) -- "
              "the step 3 expectation in the request does not hold for this data.")

    # Step 4: base rate, computed BEFORE evaluating the proximity test
    coverage = coverage_base_rate(len(deriv), crossings, PROXIMITY)
    base_rate = coverage.mean()
    print(f"\nBase rate: {base_rate:.3f} of ALL indices are within +/-{PROXIMITY} of "
          f"some zero-crossing. This is P(random point passes the proximity test).")

    # Step 5: proximity test for the 3 known changepoints, judged against the base rate
    print(f"\nProximity test for known changepoints (within {PROXIMITY} windows):")
    hits = 0
    for cp in KNOWN_CHANGEPOINTS:
        # smoothed-derivative index i corresponds to raw gap index ~ i + K (offset by the
        # rolling-mean window and the diff); use cp directly as an approximate window-space index
        # since PROXIMITY=50 >> the ~1-window alignment error this introduces.
        idx = cp
        if idx >= len(deriv):
            print(f"  changepoint {cp}: out of range for smoothed-derivative series (len={len(deriv)})")
            continue
        dists = np.abs(crossings - idx)
        nearest = crossings[np.argmin(dists)]
        min_dist = dists.min()
        passes = min_dist <= PROXIMITY
        hits += passes
        print(f"  changepoint {cp}: nearest zero-crossing at {nearest} (distance {min_dist}) "
              f"-> {'PASS' if passes else 'FAIL'} (chance of passing at random: {base_rate:.3f})")

    print(f"\n{hits}/3 changepoints pass. Base rate for a single random point to pass: {base_rate:.3f}.")
    if base_rate > 0.9:
        print("VERDICT: base rate is too high for this test to discriminate. "
              "Passing is not meaningful evidence either way -- INCONCLUSIVE, same failure mode as raw test.")
    elif hits == 3 and base_rate < 0.3:
        print("VERDICT: proximity result is meaningfully better than chance -- weak confirmation.")
    elif hits < 3:
        print("VERDICT: not all changepoints align -- test does not confirm Layer 2.")
    else:
        print("VERDICT: ambiguous -- passes, but base rate is not low enough to call this strong evidence.")

    # Step 6: plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(deriv, color="#4c72b0", lw=0.8, label=f"smoothed (K={K}) gap derivative")
    ax.scatter(crossings, deriv[crossings], color="#2a9d5c", s=10, zorder=3,
               label=f"zero-crossings (n={len(crossings)}, density={density:.2f})")
    for cp in KNOWN_CHANGEPOINTS:
        ax.axvline(cp, color="#d1495b", ls="--", lw=1.5)
    ax.axvline(KNOWN_CHANGEPOINTS[0], color="#d1495b", ls="--", lw=1.5,
               label="known regime changepoints (1529, 2501, 4211)")
    ax.set_xlabel("window index (smoothed-derivative space)")
    ax.set_ylabel("d/dn [K=100 rolling mean of gap size]")
    ax.set_title(f"Smoothed gap derivative zero-crossings vs known changepoints\n"
                 f"(base rate within +/-{PROXIMITY}: {base_rate:.0%} of all indices)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
