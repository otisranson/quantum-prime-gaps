"""experiments/mi_changepoints_sim_20k.py

MI-based changepoint detection at 20k scale, run against the quantum MI
series produced by experiments/terrain_20000primes_sim.py.

**What "same config as regime_fit_5k.py" actually means here, checked
directly rather than assumed:** regime_fit_5k.py's binary segmentation is a
*fixed*-count search -- `binary_segmentation(rm, n_bkps=3, min_size=100)` --
with no permutation-null stopping rule and no seed anywhere in that script
(it's a purely deterministic least-squares cost, no randomness). There is no
`n_perm_stop` or `stop_percentile` parameter in regime_fit_5k.py to carry
over; assuming exactly 3 changepoints would still hold at 4x the data is an
unexamined assumption, not a "same config" choice -- and this repo already
flagged that exact problem once before: layer3_20k_scaleup.py hit the
identical issue scaling the *raw-gap* version of this same detector to 20k,
and replaced the fixed n_bkps=3 assumption with a data-driven stopping rule
(each candidate split must beat the 99th percentile of a 200-trial
within-segment permutation null before being accepted).

**Decision:** this script reuses that already-validated stopping rule
(rolling_k=100, min_size=400, n_perm_stop=200, stop_percentile=99.0,
max_splits=100, seed=42 -- copied verbatim from layer3_20k_scaleup.py,
including its vectorized O(n) best_single_split, verified there against the
naive O(n^2) reference) applied to the *MI rolling mean* instead of the
raw-gap rolling mean. The underlying cost function (least-squares mean-shift)
is unchanged from regime_fit_5k.py's original -- what changed is only the
stopping rule, and only because a fixed count of 3 has no principled basis
at 4x the series length. min_size=400 already matches ~2% of the ~19,897-
point MI rolling-mean series here, the same proportional choice
layer3_20k_scaleup.py made for its ~19,899-point raw-gap series -- no further
adjustment was needed.

Copied here, not imported, per this repo's standalone-script convention.

Run: python experiments/mi_changepoints_sim_20k.py <path to results_20000primes.json>
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
OUT_ROOT = REPO_ROOT / "output" / "prime"

ROLLING_K = 100
MIN_SIZE = 400
N_PERM_STOP = 200
STOP_PERCENTILE = 99.0
MAX_SPLITS = 100
SEED = 42


def rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def best_single_split_naive(x: np.ndarray, min_size: int):
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
    x[750:] += 2.0
    naive = best_single_split_naive(x, 100)
    fast = best_single_split(x, 100)
    assert naive is not None and fast is not None
    assert abs(naive[0] - fast[0]) < 1e-6 and naive[1] == fast[1], (
        f"Vectorized best_single_split disagrees with the naive O(n^2) reference: {fast} vs {naive}"
    )
    print(f"Verified: vectorized best_single_split matches the naive O(n^2) reference "
          f"(gain={fast[0]:.4f}, split={fast[1]}).")


def significant_binary_segmentation(x, min_size, n_perm, stop_percentile, max_splits, rng) -> list[dict]:
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


def auto_commit_push(out_dir: Path, n_cp: int, ts: str) -> None:
    msg = f"experiment: MI changepoints sim 20k {ts} -- {n_cp} changepoints found (data-driven stopping)"
    subprocess.run(["git", "add", str(out_dir.relative_to(REPO_ROOT))], check=True, cwd=REPO_ROOT)
    commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        print(f"\n  Committed: {out_dir.relative_to(REPO_ROOT)}")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"\n  Git commit skipped: {commit.stdout.strip()}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python experiments/mi_changepoints_sim_20k.py <path to results_20000primes.json>")
        sys.exit(1)
    terrain_results_path = Path(sys.argv[1])

    rng = np.random.default_rng(SEED)
    verify_vectorized_matches_naive(rng)

    with open(terrain_results_path) as f:
        terrain_data = json.load(f)
    mi_series = np.array([r["mi"] for r in terrain_data["per_window"]])
    print(f"\nLoaded {len(mi_series)} MI values from {terrain_results_path}")

    rm = rolling_mean(mi_series, ROLLING_K)
    print(f"K={ROLLING_K} rolling mean: {len(rm)} points ({100 * MIN_SIZE / len(rm):.2f}% min_size fraction). "
          f"Running data-driven binary segmentation (min_size={MIN_SIZE}, stop at {STOP_PERCENTILE}th "
          f"percentile of a {N_PERM_STOP}-trial within-segment permutation null)...")

    accepted = significant_binary_segmentation(rm, MIN_SIZE, N_PERM_STOP, STOP_PERCENTILE, MAX_SPLITS, rng)
    for row in accepted:
        row["raw_position"] = row["idx"] + ROLLING_K - 1

    print(f"\n{len(accepted)} MI-based changepoints found at 20k scale "
          f"(vs. the original fixed count of 3 at 5k scale).")
    for row in accepted:
        print(f"  position={row['raw_position']:6d}  gain={row['gain']:.5f}  null_threshold={row['null_threshold']:.5f}")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_by": "experiments/mi_changepoints_sim_20k.py",
        "signal_type": "quantum-measured MI (AerSimulator, terrain_20000primes_sim.py) -- real MI, not a gap-space proxy",
        "source": str(terrain_results_path),
        "note": (
            "Same least-squares mean-shift binary-segmentation cost as regime_fit_5k.py's original "
            "3-changepoint detection, but with a data-driven permutation-null stopping rule (reused "
            "verbatim from layer3_20k_scaleup.py) instead of a fixed n_bkps=3 -- see this file's "
            "docstring for why the fixed count wasn't carried over unmodified."
        ),
        "config": {
            "rolling_k": ROLLING_K, "min_size": MIN_SIZE, "n_perm_stop": N_PERM_STOP,
            "stop_percentile": STOP_PERCENTILE, "max_splits": MAX_SPLITS, "seed": SEED,
        },
        "n_changepoints": len(accepted),
        "changepoints": [
            {"position": row["raw_position"], "binary_seg_gain": round(row["gain"], 6),
             "null_threshold": round(row["null_threshold"], 6)}
            for row in accepted
        ],
        "original_fixed_3_changepoints": [1529, 2501, 4211],
        "original_valid_range": 4999,
    }
    out_path = out_dir / "mi_changepoints_sim_20k.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {out_path.relative_to(REPO_ROOT)}")

    auto_commit_push(out_dir, len(accepted), ts)


if __name__ == "__main__":
    main()
