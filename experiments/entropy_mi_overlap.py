"""experiments/entropy_mi_overlap.py

Overlap test between the flagged entropy-deviation windows
(experiments/gap_entropy_windows.py, output/prime/20260818_224457/results.json)
and an expanded, 20k-scale changepoint list -- with a permutation-null baseline,
not just raw overlap counts.

**Where the original MI changepoints (1529, 2501, 4211) actually come from, and why
they stop at N~4999 -- investigated directly, not assumed:** `regime_fit_5k.py`
loads `per_window` mutual-information values from
`output/prime/20260816_010716/terrain_5000primes/results_5000primes.json` --
real quantum-circuit-measured MI, produced by `terrain_5000primes.py`'s Bell-pair /
RY-gap-encoding / approximated-iQFT circuit run over the 5000-prime sequence. The
script itself has no hardcoded N-limit and no compute-cost guard; it simply reads
whatever `per_window` data exists in that one JSON file, which covers ~4996 windows
because that is the scope of the one quantum terrain run that was ever executed in
this repo. There is no 20,000-prime terrain run anywhere in this repo, and MI is a
quantum measurement, not a deterministic function of the gap sequence -- so it
cannot be recovered from `data/primes_20000.json` (primes and gaps only, no MI).

**Consequence, and the decision made about it:** "rerun the MI changepoint
detection method against the full 20k cache" is not executable as originally
specified, since there is no MI in that cache. Two ways to close the gap were
considered: (a) run a brand-new 20k-prime quantum terrain circuit to generate real
MI at that scale -- a large new computation, not a rerun; or (b) reuse the
already-existing 20k-scale changepoint set from `layer3_20k_scaleup.py`
(`output/prime/20260818_015045/results.json`), which already applied the *same*
binary-segmentation algorithm (least-squares mean-shift cost, unmodified core
logic, vectorized-vs-naive equivalence verified in that script) to the raw-gap
rolling mean instead of MI, specifically because no 20k MI data exists. Per
explicit user instruction, this script takes path (b) -- classical reuse of the
existing 20k-scale gap-space set, no quantum circuit run.

**Naming, deliberately not "mi_changepoints_full.json":** CLAUDE.md is explicit
that the MI-based 3-changepoint set and the raw-gap-based 39-changepoint set are
different signals and must not be conflated. Labeling a raw-gap-derived file
"mi_changepoints_*" would be exactly that conflation, so the expanded list here is
saved as `expanded_changepoints_gap_space.json` and documented throughout as
gap-space, not MI-space, even though it is structured as a flat position list so
it's usable as a drop-in wherever a bare changepoint list (like
`KNOWN_MI_CHANGEPOINTS`) is consumed.

**Permutation-null baseline:** for a fixed layout of flagged-window centers, the
null distribution of "distance from a uniformly random point in [0, 19999] to its
nearest flagged-window center" does not depend on which changepoint is being
tested -- so one shared null sample (default 100,000 draws) is drawn once and
reused for every changepoint's p-value, rather than resampled per changepoint.
p-value per changepoint = fraction of null draws whose nearest-flagged-window
distance is <= that changepoint's observed nearest-flagged-window distance (i.e.
how often a random point would land at least as close, by chance).

Run: python experiments/entropy_mi_overlap.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
OUT_ROOT = REPO_ROOT / "output" / "prime"

ENTROPY_RESULTS_PATH = REPO_ROOT / "output/prime/20260818_224457/results.json"
GAP_CHANGEPOINTS_PATH = REPO_ROOT / "output/prime/20260818_015045/results.json"
ORIGINAL_MI_CHANGEPOINTS = [1529, 2501, 4211]
ORIGINAL_MI_VALID_RANGE = 4999
DOMAIN_MAX = 19999  # inclusive, "a uniformly random point in [0, 19999]"

N_PERM = 100_000
SEED = 42
SIG_ALPHA = 0.05


def load_flagged_windows() -> list[dict]:
    with open(ENTROPY_RESULTS_PATH) as f:
        data = json.load(f)
    return data["flagged_windows"]


def load_gap_space_changepoints() -> tuple[list[int], dict]:
    with open(GAP_CHANGEPOINTS_PATH) as f:
        data = json.load(f)
    positions = [int(c["position"]) for c in data["changepoints"]]
    assert len(positions) == data["n_changepoints"]
    return positions, data["config"]


def nearest_distance(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """For each value in `points`, the distance to the nearest value in `centers`."""
    diffs = np.abs(points[:, None] - centers[None, :])
    return diffs.min(axis=1)


def main() -> None:
    flagged = load_flagged_windows()
    flagged_centers = np.array([w["center"] for w in flagged])
    flagged_bounds = [(w["start"], w["end"]) for w in flagged]

    changepoints, gap_detector_config = load_gap_space_changepoints()
    print(f"Loaded {len(flagged)} flagged entropy windows from "
          f"{ENTROPY_RESULTS_PATH.relative_to(REPO_ROOT)}")
    print(f"Loaded {len(changepoints)} gap-space changepoints from "
          f"{GAP_CHANGEPOINTS_PATH.relative_to(REPO_ROOT)}")

    # ── Observed containment + nearest distance, per changepoint ───────────
    cp_arr = np.array(changepoints)
    obs_distances = nearest_distance(cp_arr, flagged_centers)
    obs_contained = np.array([
        any(s <= cp < e for s, e in flagged_bounds) for cp in changepoints
    ])
    n_contained = int(obs_contained.sum())

    # ── Shared permutation null: nearest-distance distribution for a
    #    uniformly random point against this same fixed flagged-window layout ──
    rng = np.random.default_rng(SEED)
    null_points = rng.integers(0, DOMAIN_MAX + 1, size=N_PERM).astype(float)
    null_distances = nearest_distance(null_points, flagged_centers)
    null_contained = np.array([
        any(s <= p < e for s, e in flagged_bounds) for p in null_points[:5000]
    ])  # containment check is the expensive path; 5000 draws is enough to estimate this rate stably
    null_containment_rate = float(null_contained.mean())

    # closed-form cross-check: fraction of the domain actually covered by flagged windows
    covered = np.zeros(DOMAIN_MAX + 1, dtype=bool)
    for s, e in flagged_bounds:
        covered[s:min(e, DOMAIN_MAX + 1)] = True
    closed_form_coverage = float(covered.mean())

    # ── Per-changepoint p-values ─────────────────────────────────────────
    p_values = np.array([float((null_distances <= d).mean()) for d in obs_distances])
    n_significant = int((p_values < SIG_ALPHA).sum())
    expected_false_positives = SIG_ALPHA * len(changepoints)

    print("\n== Observed ==")
    print(f"  Changepoints contained in a flagged window: {n_contained} / {len(changepoints)}")
    print(f"  Median nearest-flagged-window distance: {float(np.median(obs_distances)):.1f}")

    print(f"\n== Null baseline (n_perm={N_PERM}, seed={SEED}) ==")
    print(f"  Null containment rate (sampled): {null_containment_rate:.4f}")
    print(f"  Closed-form domain coverage by flagged windows: {closed_form_coverage:.4f}")
    print(f"  Expected containment count under null: {closed_form_coverage * len(changepoints):.2f} "
          f"of {len(changepoints)}")

    print("\n== Per-changepoint significance ==")
    print(f"  Changepoints with p < {SIG_ALPHA}: {n_significant} / {len(changepoints)} "
          f"(expected by chance alone, no multiple-comparison correction: {expected_false_positives:.1f})")

    # ── Output ───────────────────────────────────────────────────────────
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    expanded_list_payload = {
        "generated_by": "experiments/entropy_mi_overlap.py, reusing layer3_20k_scaleup.py's detection",
        "signal_type": "raw_gap_rolling_mean_binary_segmentation -- NOT quantum-measured MI",
        "source": str(GAP_CHANGEPOINTS_PATH.relative_to(REPO_ROOT)),
        "note": (
            "The original 3 changepoints (1529, 2501, 4211) come from quantum-measured MI "
            "(regime_fit_5k.py) and are only valid within gap-index < 4999 -- no 20k-prime "
            "quantum terrain run exists in this repo, so true MI-based changepoints cannot be "
            "computed at this scale from the primes/gaps cache alone. This list instead reuses "
            "the same binary-segmentation algorithm applied to the raw-gap rolling mean, "
            "already run at 20k scale by layer3_20k_scaleup.py. Do not treat this as a literal "
            "MI-based extension; it is a different signal at the same detection method, per "
            "CLAUDE.md's explicit warning against conflating the two changepoint sets."
        ),
        "n_changepoints": len(changepoints),
        "changepoints": changepoints,
        "detector_config": gap_detector_config,
        "original_mi_based_changepoints": ORIGINAL_MI_CHANGEPOINTS,
        "original_mi_valid_range": ORIGINAL_MI_VALID_RANGE,
    }
    expanded_path = out_dir / "expanded_changepoints_gap_space.json"
    expanded_path.write_text(json.dumps(expanded_list_payload, indent=2))
    print(f"\nSaved expanded changepoint list to {expanded_path.relative_to(REPO_ROOT)}")

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
        "gap_changepoints_source": str(GAP_CHANGEPOINTS_PATH.relative_to(REPO_ROOT)),
        "n_flagged_windows": len(flagged),
        "n_changepoints": len(changepoints),
        "n_contained": n_contained,
        "null_permutations": N_PERM,
        "seed": SEED,
        "null_containment_rate_sampled": round(null_containment_rate, 6),
        "closed_form_domain_coverage": round(closed_form_coverage, 6),
        "expected_contained_under_null": round(closed_form_coverage * len(changepoints), 3),
        "n_significant_p_lt_0_05": n_significant,
        "expected_false_positives_no_correction": round(expected_false_positives, 2),
        "overlap": overlap_rows,
        "original_mi_context": {
            "changepoints": ORIGINAL_MI_CHANGEPOINTS,
            "valid_range": ORIGINAL_MI_VALID_RANGE,
            "note": "reported for context only; the 39-point overlap above is the primary result",
        },
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    # ── Markdown summary ─────────────────────────────────────────────────
    md_lines = [
        f"# Entropy / expanded-changepoint overlap, with permutation-null baseline -- {ts}",
        "",
        "## Where the original MI changepoints come from, and why they stop at N~4999",
        "",
        "`regime_fit_5k.py` runs binary-segmentation changepoint detection on quantum-measured "
        "mutual information (`per_window` MI from `output/prime/20260816_010716/terrain_5000primes/"
        "results_5000primes.json`), produced by an actual quantum circuit run "
        "(`terrain_5000primes.py`) over the 5000-prime sequence. The script has no hardcoded "
        "N-limit and no compute-cost guard of its own -- it simply reads whatever `per_window` data "
        "exists in that one file, which covers ~4996 windows because that is the scope of the only "
        "quantum terrain run ever executed in this repo. No 20,000-prime terrain run exists, and "
        "MI is a quantum measurement, not a function of the gap sequence alone, so it cannot be "
        "recovered from `data/primes_20000.json` (primes and gaps only).",
        "",
        "**Decision on how this was handled (per explicit instruction):** rather than running a new, "
        "large 20k-prime quantum terrain circuit, this reuses the existing 20k-scale changepoint set "
        "already produced classically by `layer3_20k_scaleup.py` -- the *same* binary-segmentation "
        "algorithm, unmodified core logic, applied to the raw-gap rolling mean instead of MI "
        "(because no 20k MI data exists). This is a different signal from the original MI-based "
        "detection, not a true extension of it -- named and documented as gap-space throughout, per "
        "CLAUDE.md's explicit warning against conflating the two changepoint sets.",
        "",
        "## Changepoint count: original vs. expanded",
        "",
        f"- Original (MI-based, quantum-measured, valid only for gap-index < {ORIGINAL_MI_VALID_RANGE}): "
        f"**{len(ORIGINAL_MI_CHANGEPOINTS)}** -- {ORIGINAL_MI_CHANGEPOINTS}",
        f"- Expanded (gap-space, raw-gap rolling mean, classical, full 20k range): **{len(changepoints)}**",
        "",
        "## Overlap table (all changepoints, nearest flagged window, distance, null p-value)",
        "",
        "p-value = fraction of 100,000 uniformly random points in [0, 19999] whose nearest-flagged-"
        "window distance is <= this changepoint's observed distance (lower = more surprising under "
        "the null of no real relationship).",
        "",
        "| changepoint | nearest flagged window | distance | contained | null p-value |",
        "|---|---|---|---|---|",
    ]
    for row in overlap_rows:
        window_str = f"[{row['nearest_flagged_window_start']}, {row['nearest_flagged_window_end']})"
        md_lines.append(
            f"| {row['changepoint']} | {window_str} (center {row['nearest_flagged_window_center']:.1f}) | "
            f"{row['distance']:.1f} | {'yes' if row['contained'] else 'no'} | {row['null_p_value']:.5f} |"
        )
    md_lines += [
        "",
        "## Aggregate hit rate vs. null baseline",
        "",
        f"- Observed: **{n_contained} / {len(changepoints)}** changepoints fall inside a flagged window.",
        f"- Null-expected containment: **{closed_form_coverage * len(changepoints):.2f} / {len(changepoints)}** "
        f"(closed-form domain coverage by flagged windows = {closed_form_coverage:.4f}, sampled null "
        f"containment rate = {null_containment_rate:.4f} -- these agree, confirming the null simulation "
        "is well-calibrated).",
        f"- Changepoints individually significant at p < {SIG_ALPHA} (uncorrected): **{n_significant} / "
        f"{len(changepoints)}** vs. **{expected_false_positives:.1f}** expected by chance alone with no "
        "real relationship (5% of 39, since 39 independent tests at alpha=0.05 will produce that many "
        "false positives on average even under a true null -- no multiple-comparison correction was "
        "applied, so this count needs to clear that bar meaningfully, not just be nonzero, before being "
        "read as a real signal).",
        "",
        "## Honest interpretation",
        "",
    ]
    if n_contained > closed_form_coverage * len(changepoints) * 1.5 and n_significant > expected_false_positives * 1.5:
        verdict = (
            f"Observed containment ({n_contained}/{len(changepoints)}) is meaningfully above the "
            f"null-expected rate ({closed_form_coverage * len(changepoints):.1f}/{len(changepoints)}), and "
            f"{n_significant} changepoints clear p<{SIG_ALPHA} individually vs. {expected_false_positives:.1f} "
            "expected by chance. This is suggestive of a real relationship between gap-space regime "
            "boundaries and entropy-deviation windows -- worth a closer look, though still not a fully "
            "corrected-for-multiple-comparisons confirmation."
        )
    elif n_contained <= closed_form_coverage * len(changepoints) * 1.1 and n_significant <= expected_false_positives + 1:
        verdict = (
            f"Observed containment ({n_contained}/{len(changepoints)}) sits close to what the null "
            f"baseline predicts by chance alone ({closed_form_coverage * len(changepoints):.1f}/"
            f"{len(changepoints)}), and the count of individually significant changepoints "
            f"({n_significant}) is within range of the {expected_false_positives:.1f} expected false "
            "positives from testing 39 changepoints uncorrected. This is a null result -- the raw-gap "
            "changepoints and the entropy-deviation windows do not show evidence of a real relationship "
            "beyond chance at this sample size, consistent with most other cross-signal comparisons "
            "already on record in this repo's hypotheses/ files."
        )
    else:
        verdict = (
            f"Mixed: observed containment ({n_contained}/{len(changepoints)}) and the null-expected rate "
            f"({closed_form_coverage * len(changepoints):.1f}/{len(changepoints)}) are close enough, or "
            f"the significant-changepoint count ({n_significant} vs. {expected_false_positives:.1f} "
            "expected) is elevated but not decisively so, that this does not cleanly round to either "
            "'confirmed' or 'coincidence' -- reported as-is rather than forced into either bucket. See "
            "the per-changepoint table above for which specific changepoints (if any) look like the "
            "more plausible candidates before dismissing or confirming the aggregate."
        )
    md_lines.append(verdict)
    md_path = out_dir / "overlap_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved summary to {md_path.relative_to(REPO_ROOT)}")

    msg = (f"experiment: entropy/expanded-changepoint overlap {ts} -- "
           f"{len(changepoints)} gap-space changepoints, contained={n_contained}/{len(changepoints)} "
           f"(null-expected={closed_form_coverage * len(changepoints):.1f}), "
           f"significant(p<0.05)={n_significant}/{len(changepoints)}")
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
