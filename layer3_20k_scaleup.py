"""layer3_20k_scaleup.py

Scale-up of the Changepoint Character Comparison
(hypotheses/regime_internal_wave_structure.md, "## Changepoint Character
Comparison"): that check found 1529's local kurtosis-peak intensity (3.18)
is a weaker version of the same signature seen at 2501 (7.29) and 4211
(7.27) -- peak intensity climbing 3.18 -> 7.29 -> 7.27 across the three
known changepoints. With only n=3 points, "does intensity climb with
position" has essentially no statistical power. This script re-runs the
same idea at 4x the scale (20,000 primes instead of 5,000, read from
data/primes_20000.json -- see build_prime_cache.py) to get enough
changepoints for a real correlation test.

Method:
  1. Same binary-segmentation, least-squares mean-shift cost as
     regime_fit_5k.py originally used -- but applied to the K=100 rolling
     mean of the raw gap sequence itself, not to quantum-circuit MI (no
     20k-prime quantum run exists or is built here; see caveats in the
     writeup). Re-implemented with a vectorized O(n) best-single-split
     instead of the original's O(n^2) loop -- verified equivalent on a
     shared subset before being trusted at this scale.
  2. Unlike the original fixed-3-breakpoint search, this uses a data-driven
     stopping rule: each candidate split must beat the 99th percentile of
     its own within-segment permutation null (shuffle the segment, rerun
     the same split-finder, many times) before being accepted. This is
     what answers "how many changepoints are found," rather than assuming
     a number in advance.
  3. For every accepted changepoint, the same fine-grained local kurtosis
     scan as layer3_changepoint_1529_investigation.py (width=150, step=25,
     +/-300 gaps around the changepoint) on the raw gap sequence, and its
     peak kurtosis intensity extracted.
  4. Pearson correlation between changepoint position and peak intensity
     across however many changepoints were found, tested against a
     permutation null (shuffle intensities relative to positions) rather
     than an assumed-normal p-value.

Run: python layer3_20k_scaleup.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"

ROLLING_K = 100
MIN_SIZE = 400  # ~2% of the ~19900-point rolling-mean series, matching regime_fit_5k.py's
                # min_size=100 as a fraction of its ~4896-point series (100/4896 ~= 2.04%)
N_PERM_STOP = 200  # permutation-null trials per candidate split, for the stopping rule
STOP_PERCENTILE = 99.0
MAX_SPLITS = 100
SEED = 42

FINE_WIDTH = 150
FINE_STEP = 25
HALF_WINDOW = 300

N_PERM_CORR = 5000  # permutation trials for the position-vs-intensity correlation test

OUT_ROOT = REPO_ROOT / "output" / "prime"


# ── Data loading ─────────────────────────────────────────────────────────


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    gaps = np.array(cache["gaps"])
    assert len(gaps) == cache["n_gaps"]
    return gaps


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


# ── Vectorized binary segmentation (same least-squares mean-shift cost as
# regime_fit_5k.py, O(n) per split instead of that script's O(n^2) loop) ──


def best_single_split_naive(x: np.ndarray, min_size: int):
    """O(n^2) reference implementation, identical to regime_fit_5k.py's
    best_single_split -- kept only to verify the vectorized version below
    agrees with it, not used at full 20k scale (too slow)."""
    n = len(x)
    best = None
    base = float(np.sum((x - x.mean()) ** 2))
    for t in range(min_size, n - min_size):
        cost = float(np.sum((x[:t] - x[:t].mean()) ** 2)) + float(np.sum((x[t:] - x[t:].mean()) ** 2))
        gain = base - cost
        if best is None or gain > best[0]:
            best = (gain, t)
    return best


def best_single_split(x: np.ndarray, min_size: int):
    """Same cost function as best_single_split_naive (least-squares
    mean-shift), computed via cumulative sums in O(n) instead of O(n^2)."""
    n = len(x)
    if n < 2 * min_size:
        return None
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x**2, 0, 0.0))
    total_sum, total_sumsq = c1[n], c2[n]
    base_cost = total_sumsq - total_sum**2 / n
    t = np.arange(min_size, n - min_size)
    left_sum, left_sumsq, left_n = c1[t], c2[t], t
    right_sum, right_sumsq, right_n = total_sum - left_sum, total_sumsq - left_sumsq, n - t
    cost = (left_sumsq - left_sum**2 / left_n) + (right_sumsq - right_sum**2 / right_n)
    gain = base_cost - cost
    best_idx = int(np.argmax(gain))
    return float(gain[best_idx]), int(t[best_idx])


def verify_vectorized_matches_naive(rng: np.random.Generator) -> None:
    x = rng.normal(size=1500)
    x[750:] += 2.0  # inject an obvious mean shift so both finders have something to find
    naive = best_single_split_naive(x, 100)
    fast = best_single_split(x, 100)
    assert naive is not None and fast is not None
    assert abs(naive[0] - fast[0]) < 1e-6 and naive[1] == fast[1], (
        f"Vectorized best_single_split disagrees with the naive O(n^2) reference: {fast} vs {naive}"
    )
    print(f"Verified: vectorized best_single_split matches the naive O(n^2) reference "
          f"(gain={fast[0]:.4f}, split={fast[1]}).")


def significant_binary_segmentation(
    x: np.ndarray, min_size: int, n_perm: int, stop_percentile: float, max_splits: int, rng: np.random.Generator
) -> list[dict]:
    """Greedy binary segmentation that stops accepting splits once the next
    candidate's gain no longer clears the stop_percentile of its own
    within-segment permutation null -- a data-driven changepoint count
    instead of a fixed n_bkps."""
    segments = [(0, len(x))]
    accepted: list[dict] = []
    while len(accepted) < max_splits:
        best_overall = None
        for s, e in segments:
            res = best_single_split(x[s:e], min_size)
            if res is None:
                continue
            gain, t = res
            if best_overall is None or gain > best_overall[0]:
                best_overall = (gain, s, s + t, e)
        if best_overall is None:
            break
        gain, s, split, e = best_overall

        seg = x[s:e]
        null_gains = np.empty(n_perm)
        for p in range(n_perm):
            shuffled = rng.permutation(seg)
            res_null = best_single_split(shuffled, min_size)
            null_gains[p] = res_null[0] if res_null is not None else 0.0
        threshold = float(np.percentile(null_gains, stop_percentile))

        if gain <= threshold:
            break

        accepted.append({"idx": split, "gain": gain, "null_threshold": threshold, "segment": (s, e)})
        segments.remove((s, e))
        segments.append((s, split))
        segments.append((split, e))
        segments.sort()

    return sorted(accepted, key=lambda r: r["idx"])


# ── Fine-grained local kurtosis scan (same as
# layer3_changepoint_1529_investigation.py) ────────────────────────────────


def excess_kurtosis(x: np.ndarray) -> float:
    mean, std = x.mean(), x.std()
    if std == 0:
        return 0.0
    return float(np.mean((x - mean) ** 4) / std**4 - 3.0)


def fine_local_scan(full_gaps: np.ndarray, cp: int) -> tuple[np.ndarray, np.ndarray]:
    domain = full_gaps[cp - HALF_WINDOW : cp + HALF_WINDOW]
    starts = np.arange(0, len(domain) - FINE_WIDTH + 1, FINE_STEP)
    kurt = np.array([excess_kurtosis(domain[s : s + FINE_WIDTH]) for s in starts])
    offsets = starts + FINE_WIDTH / 2 - HALF_WINDOW
    return offsets, kurt


# ── Correlation test ─────────────────────────────────────────────────────


def permutation_correlation_test(positions: np.ndarray, intensities: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    observed_r = float(np.corrcoef(positions, intensities)[0, 1])
    null_r = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(intensities)
        null_r[i] = np.corrcoef(positions, shuffled)[0, 1]
    p_value = float(np.mean(np.abs(null_r) >= abs(observed_r)))
    return {
        "observed_r": observed_r,
        "null_mean": float(null_r.mean()),
        "null_std": float(null_r.std()),
        "p_value_two_tailed": p_value,
    }


# ── Auto-commit / push ──────────────────────────────────────────────────


def auto_commit_push(out_dir: Path, n_cp: int, corr: dict, ts: str) -> None:
    msg = (f"analysis: layer3 20k scale-up {ts} -- {n_cp} changepoints found, "
           f"position-vs-intensity r={corr['observed_r']:.3f} (p={corr['p_value_two_tailed']:.4f})")
    subprocess.run(["git", "add", str(out_dir.relative_to(REPO_ROOT))], check=True, cwd=REPO_ROOT)
    commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        print(f"\n  Committed: {out_dir.relative_to(REPO_ROOT)}")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"\n  Git commit skipped: {commit.stdout.strip()}")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    rng = np.random.default_rng(SEED)
    verify_vectorized_matches_naive(rng)

    full_gaps = load_full_gaps()
    print(f"\nLoaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")

    rm = rolling_mean(full_gaps, ROLLING_K)
    print(f"K={ROLLING_K} rolling mean: {len(rm)} points. Running data-driven binary segmentation "
          f"(min_size={MIN_SIZE}, stop at {STOP_PERCENTILE}th percentile of a {N_PERM_STOP}-trial "
          f"within-segment permutation null)...")

    accepted = significant_binary_segmentation(rm, MIN_SIZE, N_PERM_STOP, STOP_PERCENTILE, MAX_SPLITS, rng)
    # Map rolling-mean index -> raw gap index using the same convention as
    # regime_fit_5k.py: rolling_mean output position idx represents the
    # window [idx, idx+K), reported at its last raw index, idx+K-1.
    for row in accepted:
        row["raw_position"] = row["idx"] + ROLLING_K - 1

    print(f"\n{len(accepted)} changepoints found (expected more than the original 3).")
    for row in accepted:
        print(f"  position={row['raw_position']:6d}  gain={row['gain']:.5f}  null_threshold={row['null_threshold']:.5f}")

    print(f"\nFine-grained local kurtosis scan (width={FINE_WIDTH}, step={FINE_STEP}, +/-{HALF_WINDOW}) "
          f"at each changepoint...")
    for row in accepted:
        cp = row["raw_position"]
        assert cp - HALF_WINDOW >= 0 and cp + HALF_WINDOW <= len(full_gaps), (
            f"changepoint {cp} too close to a sequence boundary for a +/-{HALF_WINDOW} scan"
        )
        offsets, kurt = fine_local_scan(full_gaps, cp)
        peak_idx = int(np.argmax(kurt))
        row["peak_offset"] = float(offsets[peak_idx])
        row["peak_kurtosis"] = float(kurt[peak_idx])
        row["local_kurtosis_min"] = float(kurt.min())
        row["local_kurtosis_max"] = float(kurt.max())
        print(f"  position={cp:6d}  peak_kurtosis={row['peak_kurtosis']:.4f}  peak_offset={row['peak_offset']:+.0f}")

    positions = np.array([row["raw_position"] for row in accepted], dtype=float)
    intensities = np.array([row["peak_kurtosis"] for row in accepted], dtype=float)
    print(f"\nCorrelation test: changepoint position vs. peak kurtosis intensity (n={len(accepted)}), "
          f"permutation null (n_perm={N_PERM_CORR}, seed={SEED})...")
    corr = permutation_correlation_test(positions, intensities, N_PERM_CORR, rng)
    print(f"  observed r = {corr['observed_r']:.4f}")
    print(f"  null: mean={corr['null_mean']:.4f}  std={corr['null_std']:.4f}")
    print(f"  two-tailed p-value = {corr['p_value_two_tailed']:.4f}")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(positions, intensities, color="#4c72b0", s=30, zorder=3)
    if len(positions) >= 2:
        fit = np.polyfit(positions, intensities, 1)
        xs = np.linspace(positions.min(), positions.max(), 100)
        ax1.plot(xs, np.polyval(fit, xs), color="#d1495b", ls="--", lw=1.5,
                  label=f"linear fit (slope={fit[0]:.5f})")
    ax1.set_xlabel("changepoint position (raw gap index)")
    ax1.set_ylabel("peak kurtosis intensity")
    ax1.set_title(f"Peak kurtosis intensity vs. position (n={len(accepted)}, r={corr['observed_r']:.3f}, "
                  f"p={corr['p_value_two_tailed']:.4f})")
    ax1.legend(fontsize=8)

    ax2.hist(np.array([np.corrcoef(positions, rng.permutation(intensities))[0, 1] for _ in range(2000)]),
              bins=50, color="#4c72b0", alpha=0.7, label="null r (resampled for display)")
    ax2.axvline(corr["observed_r"], color="#d1495b", lw=2, ls="--", label=f"observed r = {corr['observed_r']:.3f}")
    ax2.set_xlabel("Pearson r (position vs. intensity)")
    ax2.set_title("Observed correlation vs. permutation null")
    ax2.legend(fontsize=8)

    fig.suptitle(f"20k scale-up -- changepoint intensity vs. position [{ts}]")
    fig.tight_layout()
    plot_path = out_dir / "layer3_20k_intensity_vs_position.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {plot_path.relative_to(REPO_ROOT)}")

    fig, ax = plt.subplots(figsize=(max(10, len(accepted) * 0.6), 4 + 0.3 * len(accepted)))
    ax.axis("off")
    row_labels = [f"cp {i}" for i in range(len(accepted))]
    cell_text = [
        [f"{row['raw_position']}", f"{row['gain']:.4f}", f"{row['peak_kurtosis']:.3f}", f"{row['peak_offset']:+.0f}"]
        for row in accepted
    ]
    table = ax.table(cellText=cell_text, rowLabels=row_labels,
                      colLabels=["position", "binary-seg gain", "peak kurtosis", "peak offset"],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title(f"All {len(accepted)} changepoints -- position / gain / kurtosis-peak [{ts}]", pad=20)
    fig.tight_layout()
    table_path = out_dir / "layer3_20k_changepoint_table.png"
    fig.savefig(table_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {table_path.relative_to(REPO_ROOT)}")

    results = {
        "timestamp": ts,
        "gaps_source": str(GAPS_CACHE_PATH.relative_to(REPO_ROOT)),
        "config": {
            "rolling_k": ROLLING_K, "min_size": MIN_SIZE, "n_perm_stop": N_PERM_STOP,
            "stop_percentile": STOP_PERCENTILE, "max_splits": MAX_SPLITS, "seed": SEED,
            "fine_width": FINE_WIDTH, "fine_step": FINE_STEP, "half_window": HALF_WINDOW,
            "n_perm_corr": N_PERM_CORR,
        },
        "n_changepoints": len(accepted),
        "changepoints": [
            {
                "position": row["raw_position"],
                "binary_seg_gain": round(row["gain"], 6),
                "null_threshold": round(row["null_threshold"], 6),
                "peak_kurtosis": round(row["peak_kurtosis"], 6),
                "peak_offset": row["peak_offset"],
                "local_kurtosis_min": round(row["local_kurtosis_min"], 6),
                "local_kurtosis_max": round(row["local_kurtosis_max"], 6),
            }
            for row in accepted
        ],
        "position_vs_intensity_correlation": corr,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    auto_commit_push(out_dir, len(accepted), corr, ts)


if __name__ == "__main__":
    main()
