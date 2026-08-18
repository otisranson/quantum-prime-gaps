"""log_polar_changepoint_remap.py

Remaps the 20,000-prime gap sequence into log-polar coordinates -- radius
= log(n) (n = 1-based position along the sequence), angle = 2*pi *
(gap_n / local_max_gap_in_window(n)) so gap magnitude becomes phase
rather than a second Cartesian axis -- and overlays the 39 confirmed
changepoints from the 20k Scale-Up run
(hypotheses/regime_internal_wave_structure.md, "20k Scale-Up: Intensity
vs. Position"; output/prime/20260818_015045/results.json) to check
whether they cluster in this coordinate system in a way they don't in
flat index/gap space.

Reads data/primes_20000.json for gaps (does not regenerate primes) and
output/prime/20260818_015045/results.json for the 39 changepoint
positions (does not re-detect them).

Run: python exploration/log_polar_changepoint_remap.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter1d

REPO_ROOT = Path(__file__).parent.parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"
CHANGEPOINTS_SOURCE = REPO_ROOT / "output/prime/20260818_015045/results.json"
OUT_DIR = REPO_ROOT / "output/prime" / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = OUT_DIR / "log_polar_changepoint_remap.png"

# Centered window for local_max_gap_in_window; ~2x the K=100 half-width
# convention used elsewhere in this repo's rolling-window analyses.
LOCAL_MAX_WINDOW = 201


def load_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        return np.array(json.load(f)["gaps"])


def load_changepoints() -> np.ndarray:
    with open(CHANGEPOINTS_SOURCE) as f:
        data = json.load(f)
    return np.array([cp["position"] for cp in data["changepoints"]])


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def nearest_neighbor_mean_distance(coords: np.ndarray) -> float:
    """Mean, over all points, of each point's distance to its nearest other point."""
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    return float(dists.min(axis=1).mean())


def main() -> None:
    gaps = load_gaps()
    changepoints = load_changepoints()
    n_gaps = len(gaps)

    index = np.arange(n_gaps)  # 0-based, matches changepoint "position" convention
    n = index + 1  # 1-based, avoids log(0)

    local_max = maximum_filter1d(gaps, size=LOCAL_MAX_WINDOW, mode="nearest")
    radius = np.log(n)
    angle = 2 * np.pi * (gaps / local_max)

    cp_n = changepoints + 1
    cp_radius = np.log(cp_n)
    cp_gaps = gaps[changepoints]
    cp_angle = 2 * np.pi * (cp_gaps / local_max[changepoints])

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.scatter(angle, radius, s=4, alpha=0.08, color="#4c72b0", label="all gaps")
    ax.scatter(
        cp_angle,
        cp_radius,
        s=90,
        color="#c44e52",
        edgecolor="black",
        zorder=5,
        label=f"{len(changepoints)} confirmed changepoints",
    )
    ax.set_title(
        f"Log-polar remap of {n_gaps:,} gaps\n"
        f"(radius=log(n), angle=2π·gap/local_max, K={LOCAL_MAX_WINDOW})",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH.relative_to(REPO_ROOT)}")

    # Nearest-neighbor separation: flat (position, gap) vs log-polar Cartesian
    # (radius*cos(angle), radius*sin(angle)). The two coordinate systems have
    # wildly different natural units (position 0-19999 & gap ~1-90, vs.
    # radius 0-10 & a bounded [-1,1]-ish Cartesian range), so each space's
    # axes are z-scored independently before computing distances -- this
    # compares relative spread (in each space's own standard-deviation
    # units), not absolute distance, which would otherwise be an
    # apples-to-oranges comparison.
    flat_coords = np.column_stack(
        [zscore(changepoints.astype(float)), zscore(cp_gaps.astype(float))]
    )
    polar_x = cp_radius * np.cos(cp_angle)
    polar_y = cp_radius * np.sin(cp_angle)
    polar_coords = np.column_stack([zscore(polar_x), zscore(polar_y)])

    flat_nn_mean = nearest_neighbor_mean_distance(flat_coords)
    polar_nn_mean = nearest_neighbor_mean_distance(polar_coords)
    direction = "increases" if polar_nn_mean > flat_nn_mean else "decreases"
    pct_change = (polar_nn_mean - flat_nn_mean) / flat_nn_mean * 100

    print(
        f"\nMean nearest-neighbor distance among {len(changepoints)} changepoints "
        f"(each space's axes independently z-scored):"
    )
    print(f"  flat (position, gap):      {flat_nn_mean:.4f}")
    print(f"  log-polar (Cartesian x,y): {polar_nn_mean:.4f}")
    print(f"  Log-polar remapping {direction} average changepoint separation by {pct_change:+.1f}%.")


if __name__ == "__main__":
    main()
