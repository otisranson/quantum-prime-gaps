"""Shared statistical core for hierarchical qubit-correlation analysis: per-qubit
marginals, a pairwise correlation matrix with 2-sigma hardware/simulator divergence
flagging, and a recursive-bisection partition tree with mutual information (bias-
corrected and null-calibrated) as the edge weight.

Data-agnostic by design -- takes bitstring -> count dicts and a qubit count, knows
nothing about which circuit produced them. Two driver scripts use this:
quantum_radio/qubit_hierarchy_analysis.py and
quantum_prime_gaps/qubit_hierarchy_analysis.py. Each owns its own data loading,
report narrative, and highlighting choices; only the math and the shared plot
scaffolding live here, since the math (bias correction, null calibration) is
subtle enough that duplicating it across two files would be a real correctness
risk if one drifted from the other.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Palette (see the project's dataviz skill): fixed categorical slots for identity
# (simulator = blue, hardware = red, matching every hw/sim panel elsewhere in this
# repo), a blue<->red diverging pair with a neutral gray midpoint for polarity
# (correlation sign), and a single blue sequential ramp for magnitude (MI z-score).
SIM_COLOR = "#2a78d6"
HW_COLOR = "#e34948"
NEUTRAL_GRAY = "#f0efec"
DIVERGING_CMAP = LinearSegmentedColormap.from_list("blue_red_diverging", ["#184f95", NEUTRAL_GRAY, "#d03b3b"])
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("sequential_blue", ["#cde2fb", "#0d366b"])
HIGHLIGHT_FILL = "#eda10022"  # translucent yellow wash, when a highlight range is given

Z_SIGNIFICANT = 2.0


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def counts_to_bits_and_probs(counts: dict[str, int], n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Qiskit writes the highest classical bit leftmost (qubit 0 is the rightmost
    character). Returns (bits, probs): bits[state, qubit] in {0,1}, probs[state]
    summing to 1. One row per *distinct* measured bitstring, not one per shot --
    every statistic below is computed as a probability-weighted sum over these,
    exact and avoiding materializing thousands of rows of duplicates."""
    keys = list(counts.keys())
    weights = np.array([counts[k] for k in keys], dtype=np.float64)
    probs = weights / weights.sum()
    bits = np.zeros((len(keys), n_qubits), dtype=np.float64)
    for row, bitstring in enumerate(keys):
        for q in range(n_qubits):
            bits[row, q] = int(bitstring[n_qubits - 1 - q])
    return bits, probs


def marginals(bits: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return bits.T @ probs


def correlation_matrix(bits: np.ndarray, probs: np.ndarray) -> np.ndarray:
    mean = bits.T @ probs
    cov = (bits * probs[:, None]).T @ bits - np.outer(mean, mean)
    var = np.diag(cov)
    denom = np.sqrt(np.outer(var, var))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def flag_divergent_pairs(
    sim_corr: np.ndarray, hw_corr: np.ndarray, n_shots: int, n_qubits: int
) -> list[tuple[int, int, float, float, float]]:
    """Fisher-style approximation: under the null of no true correlation, a
    Pearson correlation estimated from N binary trials has stderr ~ 1/sqrt(N-3).
    Flag pairs where |hw - sim| exceeds 2x the combined stderr of both estimates.
    This is a heuristic threshold (the per-qubit variables aren't bivariate
    normal), good enough to separate "probably real" from "probably shot noise,"
    not a rigorous significance test."""
    se = np.sqrt(2.0 / (n_shots - 3))
    threshold = 2 * se
    flagged = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            delta = hw_corr[i, j] - sim_corr[i, j]
            if abs(delta) > threshold:
                flagged.append((i, j, sim_corr[i, j], hw_corr[i, j], delta))
    flagged.sort(key=lambda row: -abs(row[4]))
    return flagged


# ---------------------------------------------------------------------------
# Entropy / mutual information, bias-corrected and null-calibrated
# ---------------------------------------------------------------------------


def _marginal_joint(bits: np.ndarray, probs: np.ndarray, qubit_indices: list[int]) -> dict[tuple, float]:
    """Joint distribution over a subset of qubits, marginalizing out the rest --
    aggregating probability mass by the reduced tuple of bit values."""
    dist: dict[tuple, float] = defaultdict(float)
    sub = bits[:, qubit_indices]
    for row, p in zip(sub, probs, strict=True):
        dist[tuple(row)] += p
    return dist


def entropy_bits(dist: dict[tuple, float], n_shots: int) -> float:
    """Miller-Madow bias-corrected plug-in entropy. The uncorrected estimator is
    downward-biased by roughly (K_observed - 1) / (2 N ln 2) bits, where K is the
    number of distinct outcomes -- negligible when the outcome space is small
    relative to N, but severe when a joint distribution's outcome space
    approaches the shot count (e.g. a 12-qubit joint has 4096 possible states;
    8192 shots gives only ~2 samples/state). That alone produces spurious
    entropy deficit, which flows straight into mutual information as a false
    positive if left uncorrected."""
    p = np.array(list(dist.values()))
    p = p[p > 0]
    plugin = float(-np.sum(p * np.log2(p)))
    bias = (len(p) - 1) / (2 * n_shots * np.log(2))
    return plugin + bias


def mutual_information(bits: np.ndarray, probs: np.ndarray, left: list[int], right: list[int], n_shots: int) -> float:
    joint = _marginal_joint(bits, probs, left + right)
    left_h = entropy_bits(_marginal_joint(bits, probs, left), n_shots)
    right_h = entropy_bits(_marginal_joint(bits, probs, right), n_shots)
    joint_h = entropy_bits(joint, n_shots)
    # The bias-corrected estimator is unbiased in expectation, not non-negative --
    # it can dip slightly below 0 when the true MI is at/near 0. Clip for display;
    # a real negative MI isn't a meaningful quantity anyway.
    return max(0.0, left_h + right_h - joint_h)


def _entropy_from_uniform_samples(cols: np.ndarray, n_shots: int) -> float:
    """Same Miller-Madow-corrected entropy as entropy_bits, specialized for
    synthetic per-shot samples (one row per shot, equal weight) -- vectorized via
    integer-packing + np.unique instead of a Python dict, since the null
    calibration below needs many of these per node."""
    k = cols.shape[1]
    keys = (cols.astype(np.int64) * (1 << np.arange(k))[None, :]).sum(axis=1)
    _, counts = np.unique(keys, return_counts=True)
    p = counts / n_shots
    plugin = float(-np.sum(p * np.log2(p)))
    bias = (len(p) - 1) / (2 * n_shots * np.log(2))
    return plugin + bias


def null_mi_stats(
    marg_probs: np.ndarray, n_left: int, n_shots: int, rng: np.random.Generator, n_trials: int = 200
) -> tuple[float, float]:
    """What would this node's (bias-corrected) MI look like if its qubits were
    genuinely independent, with the same marginals and the same shot count?
    Answers it by simulation rather than trusting the analytic correction alone
    -- Miller-Madow is only a first-order fix, and stays imperfect when N/K is
    small. Returns (null_mean, null_std) so the observed MI can be turned into a
    z-score against this simulated noise floor."""
    n = len(marg_probs)
    trials = np.empty(n_trials)
    for t in range(n_trials):
        synth = rng.random((n_shots, n)) < marg_probs[None, :]
        trials[t] = (
            _entropy_from_uniform_samples(synth[:, :n_left], n_shots)
            + _entropy_from_uniform_samples(synth[:, n_left:], n_shots)
            - _entropy_from_uniform_samples(synth, n_shots)
        )
    return float(trials.mean()), float(trials.std())


def build_partition_tree(
    bits: np.ndarray, probs: np.ndarray, qubit_indices: list[int], n_shots: int, full_marginals: np.ndarray, rng: np.random.Generator
) -> dict:
    """Recursively bisect `qubit_indices` (in whatever order the caller passes --
    the very first split is exactly the first half vs. second half of that order)
    and compute the mutual information between the two halves at every level,
    calibrated against an independent-qubits null (see null_mi_stats) so a node's
    MI is judged against sampling noise at its own outcome-space size, not a flat
    threshold."""
    if len(qubit_indices) == 1:
        return {"qubits": qubit_indices, "leaf": True}
    mid = (len(qubit_indices) + 1) // 2
    left_idx, right_idx = qubit_indices[:mid], qubit_indices[mid:]
    mi = mutual_information(bits, probs, left_idx, right_idx, n_shots)
    null_mean, null_std = null_mi_stats(full_marginals[qubit_indices], len(left_idx), n_shots, rng)
    z = (mi - null_mean) / null_std if null_std > 0 else 0.0
    return {
        "qubits": qubit_indices,
        "leaf": False,
        "mi": mi,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z,
        "left": build_partition_tree(bits, probs, left_idx, n_shots, full_marginals, rng),
        "right": build_partition_tree(bits, probs, right_idx, n_shots, full_marginals, rng),
    }


def tree_depth(node: dict) -> int:
    if node["leaf"]:
        return 0
    return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))


def significant_nodes(node: dict, label: str, out: list[str]) -> None:
    if node["leaf"]:
        return
    if node["z"] > Z_SIGNIFICANT:
        qs = ",".join(f"q{q}" for q in node["qubits"])
        out.append(f"{label}: [{qs}] split -- MI={node['mi']:.4f} bits, z={node['z']:.2f}")
    significant_nodes(node["left"], label, out)
    significant_nodes(node["right"], label, out)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_marginals(
    sim_p1: np.ndarray, hw_p1: np.ndarray, n_qubits: int, out_path: Path, *, highlight_upto: int | None, hw_label: str, title_note: str
) -> Path:
    fig, ax = plt.subplots(figsize=(max(10, n_qubits * 0.8), 4.5))
    if highlight_upto is not None:
        ax.axvspan(-0.5, highlight_upto - 0.5, color=HIGHLIGHT_FILL, zorder=0)

    x = np.arange(n_qubits)
    width = 0.38
    ax.bar(x - width / 2, sim_p1, width, label="simulator", color=SIM_COLOR)
    ax.bar(x + width / 2, hw_p1, width, label=hw_label, color=HW_COLOR)
    ax.axhline(0.5, color="#898781", linewidth=1, linestyle="--", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"q{i}" for i in range(n_qubits)])
    ax.set_ylabel("P(qubit = |1>)")
    ax.set_title(f"Per-qubit marginals -- hardware vs. simulator{title_note}")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_correlations(
    sim_corr: np.ndarray,
    hw_corr: np.ndarray,
    flagged: list[tuple[int, int]],
    n_qubits: int,
    out_path: Path,
    *,
    highlight_upto: int | None,
    hw_label: str,
    split_label: str | None = None,
) -> Path:
    diff = hw_corr - sim_corr
    panels = [("Simulator", sim_corr), (hw_label, hw_corr), ("Hardware - Simulator (Δ)", diff)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (title, mat) in zip(axes, panels, strict=True):
        im = ax.imshow(mat, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(n_qubits))
        ax.set_yticks(range(n_qubits))
        ax.set_xticklabels([f"q{i}" for i in range(n_qubits)], fontsize=8)
        ax.set_yticklabels([f"q{i}" for i in range(n_qubits)], fontsize=8)
        if highlight_upto is not None:
            ax.axvline(highlight_upto - 0.5, color="#eda100", linewidth=2)
            ax.axhline(highlight_upto - 0.5, color="#eda100", linewidth=2)
        if title.startswith("Hardware -"):
            for i, j in flagged:
                for a, b in ((i, j), (j, i)):
                    ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.6))

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("correlation coefficient")
    split_note = f"orange lines mark the {split_label} split; " if highlight_upto is not None and split_label else ""
    fig.suptitle(f"Pairwise qubit correlation -- {split_note}black boxes on the Δ panel are pairs flagged beyond 2σ sampling noise")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _assign_positions(node: dict, leaf_order: list[int], depth: int, positions: dict[int, tuple[float, float]]) -> float:
    """Recursively assign an x position to every node (mean of its leaves' x) and
    stash (x, depth) per node id; returns this node's x."""
    if node["leaf"]:
        x = float(leaf_order.index(node["qubits"][0]))
        positions[id(node)] = (x, depth)
        return x
    lx = _assign_positions(node["left"], leaf_order, depth + 1, positions)
    rx = _assign_positions(node["right"], leaf_order, depth + 1, positions)
    x = (lx + rx) / 2
    positions[id(node)] = (x, depth)
    return x


def _z_color_frac(z: float, z_cap: float) -> float:
    """log1p-scaled color fraction in [0, 1]. Linear scaling only works when every
    dataset's z-scores happen to land in the same order of magnitude; a strongly
    entangled circuit (z in the hundreds) and a near-independent one (z ~ 1-2) both
    need to be legible without hand-tuning z_cap per dataset, and z itself can span
    3+ orders of magnitude *within* one dataset (see quantum_prime_gaps: z=7.8 at
    one node, z=1465 at another) -- log1p keeps low-z differentiation while still
    letting high-z nodes separate from each other instead of all saturating identically."""
    return float(np.log1p(max(0.0, z)) / np.log1p(z_cap))


def _draw_tree(ax, node: dict, depth: int, positions: dict, max_depth: int, z_cap: float):
    x, _ = positions[id(node)]
    y = max_depth - depth
    if node["leaf"]:
        ax.text(x, y - 0.35, f"q{node['qubits'][0]}", ha="center", va="top", fontsize=9)
        return
    for child in (node["left"], node["right"]):
        cx, _ = positions[id(child)]
        cy = max_depth - (depth + 1)
        z_frac = min(_z_color_frac(node["z"], z_cap), 1.0)
        color = SEQUENTIAL_BLUE(z_frac)
        lw = 1 + 5 * z_frac
        ax.plot([x, x, cx], [y, cy, cy], color=color, linewidth=lw, solid_capstyle="round")
        _draw_tree(ax, child, depth + 1, positions, max_depth, z_cap)
    star = "*" if node["z"] > Z_SIGNIFICANT else ""
    ax.text(x, y + 0.12, f"{node['mi']:.3f} bits{star}\n(z={node['z']:.1f})", ha="center", va="bottom", fontsize=7, color="#52514e")


def plot_dendrograms(
    sim_tree: dict,
    hw_tree: dict,
    n_qubits: int,
    out_path: Path,
    *,
    highlight_upto: int | None,
    hw_label: str,
    title_note: str,
    z_cap: float | None = None,
) -> Path:
    """z_cap sets the color scale's upper reference point (log1p-scaled -- see
    _z_color_frac); pass the largest z-score actually present in this dataset's
    tree (or leave None to auto-detect it) so the color/linewidth encoding uses
    the full range instead of saturating early or barely moving at all."""
    leaf_order = list(range(n_qubits))
    max_depth = max(tree_depth(sim_tree), tree_depth(hw_tree))

    if z_cap is None:
        all_z = []

        def collect_z(node):
            if not node["leaf"]:
                all_z.append(node["z"])
                collect_z(node["left"])
                collect_z(node["right"])

        collect_z(sim_tree)
        collect_z(hw_tree)
        z_cap = max(max(all_z, default=Z_SIGNIFICANT), Z_SIGNIFICANT)

    fig, axes = plt.subplots(1, 2, figsize=(max(15, n_qubits * 1.1), 7.2), sharey=True)
    for ax, tree, title in zip(axes, (sim_tree, hw_tree), ("Simulator", hw_label), strict=True):
        positions: dict = {}
        _assign_positions(tree, leaf_order, 0, positions)
        if highlight_upto is not None:
            ax.axvspan(-0.5, highlight_upto - 0.5, color=HIGHLIGHT_FILL, zorder=0)
        _draw_tree(ax, tree, 0, positions, max_depth, z_cap)
        ax.set_title(title)
        ax.set_xlim(-0.5, n_qubits - 0.5)
        ax.set_ylim(-0.6, max_depth + 0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Hierarchical partition tree -- recursive bisection\n"
        f"MI: Miller-Madow corrected, calibrated vs. an independence null (* = z > 2).{title_note}",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.88))

    # A plain linear Normalize would defeat the log1p color mapping above -- build
    # tick positions in log1p-space but label them with the real z values they
    # came from, so the colorbar stays honestly readable at both ends.
    tick_zs = sorted({0, 1, 2, 5} | {round(z_cap * f) for f in (0.02, 0.1, 0.3, 0.6, 1.0)})
    tick_zs = [z for z in tick_zs if z <= z_cap]
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL_BLUE, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_ticks([_z_color_frac(z, z_cap) for z in tick_zs])
    cbar.set_ticklabels([str(z) for z in tick_zs])
    cbar.set_label("z-score vs. independent-qubits null (log scale; same marginals, same shot count)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
