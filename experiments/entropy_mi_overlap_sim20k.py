"""experiments/entropy_mi_overlap_sim20k.py

Overlap test between the flagged entropy-deviation windows
(experiments/gap_entropy_windows.py, output/prime/20260818_224457/results.json)
and the sim-based, quantum-MI-derived changepoint list at 20k scale
(experiments/mi_changepoints_sim_20k.py output), with the same
permutation-null methodology already established in
experiments/entropy_mi_overlap.py (100k+ trials, shared null sample reused
across changepoints since the null distribution doesn't depend on which
changepoint is being tested).

Also compares this sim-based MI overlap rate directly against the
already-computed gap-space overlap rate (11/39, output/prime/20260818_225919/
results.json) -- same flagged-window set, same null methodology, different
changepoint source (real quantum MI here vs. raw-gap rolling mean there) --
so the two rates are on equal footing for comparison.

Copied logic from experiments/entropy_mi_overlap.py rather than imported,
per this repo's standalone-script convention; the changepoint source and
output paths are the only substantive differences.

Run: python experiments/entropy_mi_overlap_sim20k.py <path to mi_changepoints_sim_20k.json>
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

ENTROPY_RESULTS_PATH = REPO_ROOT / "output/prime/20260818_224457/results.json"
GAP_SPACE_OVERLAP_RESULTS_PATH = REPO_ROOT / "output/prime/20260818_225919/results.json"
DOMAIN_MAX = 19999

N_PERM = 100_000
SEED = 42
SIG_ALPHA = 0.05


def load_flagged_windows() -> list[dict]:
    with open(ENTROPY_RESULTS_PATH) as f:
        return json.load(f)["flagged_windows"]


def load_sim_mi_changepoints(path: Path) -> tuple[list[int], dict]:
    with open(path) as f:
        data = json.load(f)
    positions = [int(c["position"]) for c in data["changepoints"]]
    assert len(positions) == data["n_changepoints"]
    return positions, data["config"]


def load_gap_space_comparison() -> dict:
    with open(GAP_SPACE_OVERLAP_RESULTS_PATH) as f:
        data = json.load(f)
    return {
        "n_changepoints": data["n_changepoints"],
        "n_contained": data["n_contained"],
        "closed_form_domain_coverage": data["closed_form_domain_coverage"],
        "n_significant_p_lt_0_05": data["n_significant_p_lt_0_05"],
    }


def nearest_distance(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diffs = np.abs(points[:, None] - centers[None, :])
    return diffs.min(axis=1)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python experiments/entropy_mi_overlap_sim20k.py <path to mi_changepoints_sim_20k.json>")
        sys.exit(1)
    mi_changepoints_path = Path(sys.argv[1])

    flagged = load_flagged_windows()
    flagged_centers = np.array([w["center"] for w in flagged])
    flagged_bounds = [(w["start"], w["end"]) for w in flagged]

    changepoints, mi_detector_config = load_sim_mi_changepoints(mi_changepoints_path)
    gap_space = load_gap_space_comparison()

    print(f"Loaded {len(flagged)} flagged entropy windows from {ENTROPY_RESULTS_PATH.relative_to(REPO_ROOT)}")
    print(f"Loaded {len(changepoints)} sim-based MI changepoints from {mi_changepoints_path}")
    print(f"Loaded gap-space comparison baseline from {GAP_SPACE_OVERLAP_RESULTS_PATH.relative_to(REPO_ROOT)}: "
          f"{gap_space['n_contained']}/{gap_space['n_changepoints']} contained")

    cp_arr = np.array(changepoints)
    obs_distances = nearest_distance(cp_arr, flagged_centers)
    obs_contained = np.array([any(s <= cp < e for s, e in flagged_bounds) for cp in changepoints])
    n_contained = int(obs_contained.sum())

    rng = np.random.default_rng(SEED)
    null_points = rng.integers(0, DOMAIN_MAX + 1, size=N_PERM).astype(float)
    null_distances = nearest_distance(null_points, flagged_centers)

    covered = np.zeros(DOMAIN_MAX + 1, dtype=bool)
    for s, e in flagged_bounds:
        covered[s:min(e, DOMAIN_MAX + 1)] = True
    closed_form_coverage = float(covered.mean())

    p_values = np.array([float((null_distances <= d).mean()) for d in obs_distances])
    n_significant = int((p_values < SIG_ALPHA).sum())
    expected_false_positives = SIG_ALPHA * len(changepoints)

    print(f"\n== Sim-based MI overlap (n={len(changepoints)} changepoints) ==")
    print(f"  Contained in a flagged window: {n_contained} / {len(changepoints)}")
    print(f"  Null-expected: {closed_form_coverage * len(changepoints):.2f} / {len(changepoints)} "
          f"(domain coverage = {closed_form_coverage:.4f})")
    print(f"  Individually significant (p<{SIG_ALPHA}): {n_significant} / {len(changepoints)} "
          f"vs. {expected_false_positives:.2f} expected by chance")

    sim_mi_rate = n_contained / len(changepoints)
    gap_space_rate = gap_space["n_contained"] / gap_space["n_changepoints"]
    print("\n== Comparison: sim-based MI vs. gap-space proxy ==")
    print(f"  Sim-based MI containment rate:  {sim_mi_rate:.4f} ({n_contained}/{len(changepoints)})")
    print(f"  Gap-space containment rate:     {gap_space_rate:.4f} "
          f"({gap_space['n_contained']}/{gap_space['n_changepoints']})")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap_rows = []
    for cp, d, contained, p in zip(changepoints, obs_distances, obs_contained, p_values, strict=True):
        nearest_idx = int(np.argmin(np.abs(flagged_centers - cp)))
        overlap_rows.append({
            "changepoint": int(cp),
            "nearest_flagged_window_center": float(flagged_centers[nearest_idx]),
            "nearest_flagged_window_start": int(flagged[nearest_idx]["start"]),
            "nearest_flagged_window_end": int(flagged[nearest_idx]["end"]),
            "distance": float(d),
            "contained": bool(contained),
            "null_p_value": round(float(p), 5),
        })

    results = {
        "timestamp": ts,
        "entropy_results_source": str(ENTROPY_RESULTS_PATH.relative_to(REPO_ROOT)),
        "mi_changepoints_source": str(mi_changepoints_path),
        "mi_detector_config": mi_detector_config,
        "n_flagged_windows": len(flagged),
        "n_changepoints": len(changepoints),
        "n_contained": n_contained,
        "sim_mi_containment_rate": round(sim_mi_rate, 6),
        "null_permutations": N_PERM,
        "seed": SEED,
        "closed_form_domain_coverage": round(closed_form_coverage, 6),
        "expected_contained_under_null": round(closed_form_coverage * len(changepoints), 3),
        "n_significant_p_lt_0_05": n_significant,
        "expected_false_positives_no_correction": round(expected_false_positives, 2),
        "overlap": overlap_rows,
        "gap_space_comparison": {
            **gap_space,
            "containment_rate": round(gap_space_rate, 6),
        },
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {json_path.relative_to(REPO_ROOT)}")

    md_lines = [
        f"# Sim-based MI overlap vs. gap-space proxy, 20k scale -- {ts}",
        "",
        "## Sim-based MI overlap rate vs. null baseline",
        "",
        f"- {len(changepoints)} MI-based changepoints (data-driven stopping, see "
        f"`{mi_changepoints_path}`)",
        f"- Contained in a flagged entropy window: **{n_contained} / {len(changepoints)}** "
        f"({sim_mi_rate:.1%})",
        f"- Null-expected containment: **{closed_form_coverage * len(changepoints):.2f} / {len(changepoints)}** "
        f"(domain coverage = {closed_form_coverage:.4f})",
        f"- Individually significant at p<{SIG_ALPHA} (uncorrected): **{n_significant} / {len(changepoints)}** "
        f"vs. **{expected_false_positives:.2f}** expected by chance alone",
        "",
        "## Sim-based MI overlap rate vs. gap-space proxy rate",
        "",
        "| | n changepoints | contained | containment rate | significant (p<0.05) |",
        "|---|---|---|---|---|",
        f"| **Sim-based MI (this run)** | {len(changepoints)} | {n_contained} | {sim_mi_rate:.4f} | {n_significant} |",
        f"| **Gap-space proxy (prior run)** | {gap_space['n_changepoints']} | {gap_space['n_contained']} | "
        f"{gap_space_rate:.4f} | {gap_space['n_significant_p_lt_0_05']} |",
        "",
        "## Interpretation",
        "",
    ]
    if sim_mi_rate > gap_space_rate * 1.15:
        rel = "stronger"
        detail = (f"Sim-based MI containment ({sim_mi_rate:.1%}) is meaningfully higher than the "
                   f"gap-space proxy's ({gap_space_rate:.1%}) -- real quantum-measured MI shows a "
                   "stronger relationship with the entropy-deviation windows than the classical raw-gap "
                   "proxy did.")
    elif sim_mi_rate < gap_space_rate * 0.85:
        rel = "weaker"
        detail = (f"Sim-based MI containment ({sim_mi_rate:.1%}) is meaningfully lower than the "
                   f"gap-space proxy's ({gap_space_rate:.1%}) -- the raw-gap proxy showed a stronger "
                   "relationship with the entropy-deviation windows than real quantum-measured MI does "
                   "here.")
    else:
        rel = "comparable"
        detail = (f"Sim-based MI containment ({sim_mi_rate:.1%}) and the gap-space proxy's "
                   f"({gap_space_rate:.1%}) are close enough (within 15% relative) that this reads as "
                   "comparable, not a clear win for either signal.")
    md_lines.append(
        f"**Sim-based MI shows {rel} correlation with entropy regime boundaries relative to the "
        f"classical gap-space proxy.** {detail} Neither comparison here applies a multiple-comparison "
        "correction across the two detector types being compared, and both overlap tests share the "
        "same underlying limitation (the flagged entropy windows themselves are candidates, not "
        "confirmed regime boundaries -- see experiments/gap_entropy_windows.py) -- read this as a "
        "relative comparison between two proxies for the same underlying question, not a confirmation "
        "of either one in isolation."
    )
    md_path = out_dir / "sim_mi_vs_gap_space_comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved summary to {md_path.relative_to(REPO_ROOT)}")

    msg = (f"experiment: sim MI vs entropy overlap {ts} -- "
           f"sim-MI={n_contained}/{len(changepoints)} ({sim_mi_rate:.3f}) vs. "
           f"gap-space={gap_space['n_contained']}/{gap_space['n_changepoints']} ({gap_space_rate:.3f})")
    subprocess.run(["git", "add", str(out_dir.relative_to(REPO_ROOT))], check=True, cwd=REPO_ROOT)
    commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        print(f"\n  Committed: {out_dir.relative_to(REPO_ROOT)}")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"\n  Git commit skipped: {commit.stdout.strip()}")


if __name__ == "__main__":
    main()
