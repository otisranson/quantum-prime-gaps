"""log_polar_cluster_test.py

Follow-up to exploration/log_polar_changepoint_remap.py's finding that the
39 confirmed changepoints have 49% tighter mean nearest-neighbor distance
in log-polar space than in flat (position, gap) space. That effect looked,
by eye, driven mostly by one visible cluster around 30-50 degrees at large
radius rather than a uniform effect across all 39 points. This isolates
that cluster, removes it, and checks whether the tightening effect
persists, weakens, or disappears for the remaining changepoints -- plus a
permutation test against random gap-sequence indices.

Reads data/primes_20000.json for gaps and
output/prime/20260818_015045/results.json for the 39 changepoint
positions (same sources as log_polar_changepoint_remap.py, not
re-detected).

Run: python exploration/log_polar_cluster_test.py
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
OUT_PATH = OUT_DIR / "log_polar_cluster_exclusion_test.png"

# Must match log_polar_changepoint_remap.py's window exactly, so radius and
# angle reproduce the same plot and the same visual cluster.
LOCAL_MAX_WINDOW = 201

# Cluster identified by eye in log_polar_changepoint_remap.png: a group of
# changepoints visibly denser than the rest of the 39, around 30-50 degrees
# at large radius (late-sequence positions).
CLUSTER_ANGLE_MIN_DEG = 30.0
CLUSTER_ANGLE_MAX_DEG = 50.0
CLUSTER_RADIUS_MIN = 9.0

N_PERMUTATIONS = 1000
SEED = 42


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


def log_polar(idx: np.ndarray, gaps: np.ndarray, local_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = idx + 1  # 1-based, avoids log(0)
    radius = np.log(n)
    angle = 2 * np.pi * (gaps[idx] / local_max[idx])
    return radius, angle


def flat_vs_polar_nn(
    positions: np.ndarray, gaps: np.ndarray, radius: np.ndarray, angle: np.ndarray
) -> tuple[float, float]:
    """Mean NN distance in flat (position, gap) vs. log-polar Cartesian space, each z-scored per axis."""
    flat_coords = np.column_stack(
        [zscore(positions.astype(float)), zscore(gaps[positions].astype(float))]
    )
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    polar_coords = np.column_stack([zscore(x), zscore(y)])
    return nearest_neighbor_mean_distance(flat_coords), nearest_neighbor_mean_distance(polar_coords)


def main() -> None:
    gaps = load_gaps()
    changepoints = load_changepoints()
    n_gaps = len(gaps)
    local_max = maximum_filter1d(gaps, size=LOCAL_MAX_WINDOW, mode="nearest")

    cp_radius, cp_angle = log_polar(changepoints, gaps, local_max)
    cp_angle_deg = np.degrees(cp_angle) % 360

    in_cluster = (
        (cp_angle_deg >= CLUSTER_ANGLE_MIN_DEG)
        & (cp_angle_deg <= CLUSTER_ANGLE_MAX_DEG)
        & (cp_radius >= CLUSTER_RADIUS_MIN)
    )
    cluster_positions = changepoints[in_cluster]
    remaining_positions = changepoints[~in_cluster]

    print(
        f"Cluster (angle {CLUSTER_ANGLE_MIN_DEG}-{CLUSTER_ANGLE_MAX_DEG} deg, "
        f"radius >= {CLUSTER_RADIUS_MIN}): {len(cluster_positions)} changepoints "
        f"-- {sorted(cluster_positions.tolist())}"
    )
    print(f"Remaining: {len(remaining_positions)} changepoints")

    flat_nn_all, polar_nn_all = flat_vs_polar_nn(changepoints, gaps, cp_radius, cp_angle)
    pct_all = (polar_nn_all - flat_nn_all) / flat_nn_all * 100

    rem_radius, rem_angle = log_polar(remaining_positions, gaps, local_max)
    flat_nn_rem, polar_nn_rem = flat_vs_polar_nn(remaining_positions, gaps, rem_radius, rem_angle)
    pct_rem = (polar_nn_rem - flat_nn_rem) / flat_nn_rem * 100

    print(f"\nAll 39:              flat={flat_nn_all:.4f}  polar={polar_nn_all:.4f}  ({pct_all:+.1f}%)")
    print(
        f"Remaining {len(remaining_positions)}: flat={flat_nn_rem:.4f}  "
        f"polar={polar_nn_rem:.4f}  ({pct_rem:+.1f}%)"
    )

    if pct_rem * pct_all < 0:
        verdict = "reverses direction"
    elif abs(pct_rem) < abs(pct_all) * 0.5:
        verdict = "weakens substantially"
    elif abs(pct_rem) < abs(pct_all) * 0.9:
        verdict = "weakens modestly"
    else:
        verdict = "persists"
    print(f"Verdict: clustering effect {verdict} once the cluster is excluded.")

    # Permutation test: N_PERMUTATIONS resamples of len(remaining_positions)
    # random gap-sequence indices, log-polar Cartesian NN mean distance each
    # time (axes z-scored within each resample, matching the observed-value
    # computation above).
    rng = np.random.default_rng(SEED)
    k = len(remaining_positions)
    null_dists = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        sample_idx = rng.choice(n_gaps, size=k, replace=False)
        s_radius, s_angle = log_polar(sample_idx, gaps, local_max)
        x = s_radius * np.cos(s_angle)
        y = s_radius * np.sin(s_angle)
        coords = np.column_stack([zscore(x), zscore(y)])
        null_dists[i] = nearest_neighbor_mean_distance(coords)

    observed = polar_nn_rem
    percentile = float((null_dists < observed).mean() * 100)
    print(
        f"\nPermutation test (n={N_PERMUTATIONS}, k={k} random indices/draw): "
        f"null mean={null_dists.mean():.4f}, std={null_dists.std():.4f}."
    )
    print(f"Observed remaining-set polar NN mean={observed:.4f}, percentile={percentile:.1f}.")

    # --- Plot ---
    index = np.arange(n_gaps)
    all_radius, all_angle = log_polar(index, gaps, local_max)

    fig = plt.figure(figsize=(16, 8))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_polar.scatter(all_angle, all_radius, s=4, alpha=0.06, color="#4c72b0", label="all gaps")
    ax_polar.scatter(
        rem_angle, rem_radius, s=80, color="#2a9d5c", edgecolor="black", zorder=5,
        label=f"remaining ({len(remaining_positions)})",
    )
    cluster_radius, cluster_angle = log_polar(cluster_positions, gaps, local_max)
    ax_polar.scatter(
        cluster_angle, cluster_radius, s=80, color="#c44e52", edgecolor="black", zorder=6,
        label=f"excluded cluster ({len(cluster_positions)})",
    )
    ax_polar.set_title("Changepoints with excluded cluster highlighted", pad=20)
    ax_polar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    ax_hist = fig.add_subplot(1, 2, 2)
    ax_hist.hist(null_dists, bins=40, color="#e08214", alpha=0.8, label="permutation null")
    ax_hist.axvline(observed, color="#c44e52", lw=2, label=f"observed ({observed:.4f}, {percentile:.1f}th pct)")
    ax_hist.set_xlabel("mean nearest-neighbor distance (log-polar, z-scored)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"Permutation null (n={N_PERMUTATIONS}, k={k})")
    ax_hist.legend()

    fig.suptitle(
        f"Cluster exclusion test: {pct_all:+.1f}% (all 39) -> {pct_rem:+.1f}% (remaining {k}) [{verdict}]"
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved plot to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
