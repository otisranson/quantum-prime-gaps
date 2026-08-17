"""regime_fit_5k.py

Verification step for the log-summation regime hypothesis
(hypotheses/log_summation_regime.md).

Finds regime-change points in the MI rolling mean from the existing
5000-prime terrain run, fits window_number = a*ln(k) + b where k is the
regime-change index (1, 2, 3, ...), and extrapolates to predict the window
of the next (4th) regime change.

Regime changes are found with an unsupervised binary-segmentation
changepoint detector (least-squares mean-shift cost) applied to the MI
rolling mean, K=100. This choice is not tuned to produce a particular
answer: the same 3 breakpoints (+/- ~50 windows) appear for every K from
30 to 300 tested during exploration. The detector is direction-agnostic —
it finds mean shifts, not step-ups specifically — because an honest check
of this data shows the 3 breakpoints are NOT all step-ups (see plot
annotations): the sequence is up, down, up. That contradicts the "three
step-ups" framing in the original hypothesis note, and is reported as-is
rather than filtered to fit the narrative.

Run: python regime_fit_5k.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("output/prime/20260816_010716/terrain_5000primes/results_5000primes.json")
OUT_PATH = Path("output/prime/analysis/regime_fit_5k.png")
ROLLING_K = 100


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def seg_cost(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sum((x - x.mean()) ** 2))


def best_single_split(x: np.ndarray, min_size: int):
    n = len(x)
    best = None
    base = seg_cost(x)
    for t in range(min_size, n - min_size):
        cost = seg_cost(x[:t]) + seg_cost(x[t:])
        gain = base - cost
        if best is None or gain > best[0]:
            best = (gain, t)
    return best


def binary_segmentation(x: np.ndarray, n_bkps: int, min_size: int):
    segments = [(0, len(x))]
    bkps = []
    for _ in range(n_bkps):
        best_overall = None
        for s, e in segments:
            if e - s < 2 * min_size:
                continue
            res = best_single_split(x[s:e], min_size)
            if res is None:
                continue
            gain, t = res
            if best_overall is None or gain > best_overall[0]:
                best_overall = (gain, s, s + t, e)
        if best_overall is None:
            break
        gain, s, split, e = best_overall
        bkps.append((split, gain))
        segments.remove((s, e))
        segments.append((s, split))
        segments.append((split, e))
        segments.sort()
    return sorted(bkps)


def main() -> None:
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    per_window = data["per_window"]
    w = np.array([r["w"] for r in per_window])
    mi = np.array([r["mi"] for r in per_window])

    rm = rolling_mean(mi, ROLLING_K)
    rw = w[ROLLING_K - 1 :]

    # Primary changepoints: exactly 3, as observed in the terrain visualizer.
    bkps = binary_segmentation(rm, n_bkps=3, min_size=ROLLING_K)
    bkps.sort(key=lambda t: t[0])
    windows = [int(rw[idx]) for idx, _ in bkps]

    bounds = [0] + [idx for idx, _ in bkps] + [len(rm)]
    seg_means = [float(rm[bounds[i] : bounds[i + 1]].mean()) for i in range(len(bounds) - 1)]
    directions = ["up" if seg_means[i + 1] > seg_means[i] else "down" for i in range(len(seg_means) - 1)]

    print(f"Regime-change windows (K={ROLLING_K} rolling mean): {windows}")
    print(f"Segment means: {[round(m, 4) for m in seg_means]}")
    print(f"Transition directions: {directions}  <- NOT all step-ups (up-down-up)")

    # Fit window_number = a*ln(k) + b, k = regime-change index (1,2,3)
    k = np.array([1, 2, 3], dtype=float)
    x = np.log(k)
    y = np.array(windows, dtype=float)
    a, b = np.polyfit(x, y, 1)
    fit = a * x + b
    resid = y - fit
    print(f"Fit: window = {a:.2f} * ln(k) + {b:.2f}")
    print(f"Residuals at k=1,2,3: {[round(r, 1) for r in resid]}")

    # Predict regime-change #4
    k4 = 4.0
    predicted_window = a * math.log(k4) + b
    print(f"Predicted regime-change #4 window: {predicted_window:.1f}")

    # Sanity check: does this predicted window fall inside data we ALREADY
    # have (0..4995)? If so, it is not a blind extrapolation -- check
    # whether a 4-breakpoint rescan of the existing data corroborates it.
    n_windows_available = int(w.max())
    print(f"Existing data covers windows 0..{n_windows_available}")
    if predicted_window <= n_windows_available:
        bkps4 = binary_segmentation(rm, n_bkps=4, min_size=ROLLING_K)
        bkps4.sort(key=lambda t: t[0])
        windows4 = [(int(rw[idx]), round(gain, 5)) for idx, gain in bkps4]
        print("Predicted window falls WITHIN existing data -- this is checkable now.")
        print(f"4-breakpoint rescan of existing data: {windows4}")
        nearest = min(windows4, key=lambda t: abs(t[0] - predicted_window))
        print(f"Nearest candidate breakpoint to prediction: window {nearest[0]} (gain {nearest[1]})")
        print(f"Distance from prediction: {abs(nearest[0] - predicted_window):.1f} windows")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(rw, rm, color="#4c72b0", lw=1, label=f"MI rolling mean (K={ROLLING_K})")
    colors = {"up": "#2a9d5c", "down": "#d1495b"}
    for i, (win, dirn) in enumerate(zip(windows, directions + [None], strict=True)):
        c = colors.get(dirn, "#888888") if dirn else "#888888"
        ax1.axvline(win, color=c, ls="--", lw=1.5, label=f"regime change {i+1} (w={win}, {dirn or '?'})")
    ax1.set_xlabel("window number")
    ax1.set_ylabel("MI rolling mean")
    ax1.set_title("5000-prime terrain: MI rolling mean with detected regime changes")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.scatter(x, y, color="#4c72b0", zorder=3, label="observed regime changes")
    xs_fit = np.linspace(0, math.log(4.5), 100)
    ax2.plot(xs_fit, a * xs_fit + b, color="#888888", ls="--", label=f"fit: {a:.1f}*ln(k)+{b:.1f}")
    ax2.scatter([math.log(4)], [predicted_window], color="#d1495b", marker="x", s=80, zorder=3,
                label=f"predicted regime change 4 (w={predicted_window:.0f})")
    ax2.set_xlabel("ln(k), k = regime-change index")
    ax2.set_ylabel("window number")
    ax2.set_title("Regime-change window vs ln(k)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
