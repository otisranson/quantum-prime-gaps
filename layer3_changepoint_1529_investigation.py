"""layer3_changepoint_1529_investigation.py

Follow-up to the Kurtosis Robustness Check
(hypotheses/regime_internal_wave_structure.md, "## Kurtosis Robustness
Check"): the sliding-window kurtosis test there found real, localized jumps
at changepoints 2501 and 4211 (4.9x and 4.3x the typical local step size)
but no unusual jump at 1529 (0.56x, below the median step). This script asks
what, if anything, distinguishes 1529 from the other two: a weaker version
of the same kind of transition, or a fundamentally different kind of
changepoint.

Four independent angles, all descriptive/comparative (no cross-regime
similarity retest -- that stays refuted, see the Regime Overlay check):

  1. Original detection evidence -- re-run the same binary-segmentation
     changepoint detector (least-squares mean-shift cost on the MI rolling
     mean, K=100) that regime_fit_5k.py originally used, and pull the
     per-changepoint gain (cost reduction) and MI-level-shift magnitude,
     plus the ln(k) envelope-fit residual at each of the three points. Was
     1529 the weakest of the three by any of these original criteria?
  2. Local before/after comparison at all three changepoints (window=300
     gaps each side) -- level (mean), scale (variance), and shape
     (kurtosis) each tested separately via bootstrap CI overlap (before vs
     after), so it's possible to see whether a changepoint is a level-only
     shift versus a level+shape shift.
  3. A fine-grained local kurtosis scan (width=150, step=25) across a wider
     +/-300-gap window around each changepoint, aligned on offset-from-
     changepoint so all three are directly comparable side by side -- looks
     for a real but smaller/delayed transition near 1529 that the coarse
     1000+-point regime cut might have missed.

Same three changepoints and regime definitions as every other Layer 3
script in this file: 1529, 2501, 4211 in raw gap-index space.

Run: python layer3_changepoint_1529_investigation.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent
RESULTS_PATH = REPO_ROOT / "output/prime/20260816_010716/terrain_5000primes/results_5000primes.json"
KNOWN_CHANGEPOINTS = [1529, 2501, 4211]
CP_LABELS = [f"changepoint {cp}" for cp in KNOWN_CHANGEPOINTS]
CP_COLORS = ["#4c72b0", "#e08214", "#2a9d5c"]

MI_ROLLING_K = 100  # matches regime_fit_5k.py, the original detector
HALF_WINDOW = 300  # before/after comparison window, each side
FINE_WIDTH = 150
FINE_STEP = 25
N_BOOT = 2000
SEED = 42

OUT_ROOT = REPO_ROOT / "output" / "prime"


# ── Data loading ─────────────────────────────────────────────────────────


def load_full_gaps() -> np.ndarray:
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    per_window = data["per_window"]
    return np.array([r["gaps"][0] for r in per_window] + per_window[-1]["gaps"][1:])


def load_mi_series() -> tuple[np.ndarray, np.ndarray]:
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    per_window = data["per_window"]
    w = np.array([r["w"] for r in per_window])
    mi = np.array([r["mi"] for r in per_window])
    return w, mi


# ── Reimplementation of regime_fit_5k.py's original detector ───────────────
# Duplicated deliberately (self-contained scripts, repo convention) rather
# than imported -- this needs to reproduce the *original* evidence exactly,
# so any drift from regime_fit_5k.py's algorithm would be a bug, not a
# feature.


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


def original_detection_evidence() -> dict:
    w, mi = load_mi_series()
    rm = rolling_mean(mi, MI_ROLLING_K)
    rw = w[MI_ROLLING_K - 1 :]

    bkps = binary_segmentation(rm, n_bkps=3, min_size=MI_ROLLING_K)
    bkps.sort(key=lambda t: t[0])
    windows = [int(rw[idx]) for idx, _ in bkps]
    assert windows == KNOWN_CHANGEPOINTS, (
        f"re-derived changepoints {windows} don't match the known {KNOWN_CHANGEPOINTS} -- "
        "the original detector's evidence would not correspond to the changepoints under investigation."
    )

    bounds = [0] + [idx for idx, _ in bkps] + [len(rm)]
    seg_means = [float(rm[bounds[i] : bounds[i + 1]].mean()) for i in range(len(bounds) - 1)]

    per_cp = []
    for i, (_idx, gain) in enumerate(bkps):
        mean_before, mean_after = seg_means[i], seg_means[i + 1]
        per_cp.append({
            "changepoint": windows[i],
            "binary_seg_gain": float(gain),
            "mi_mean_before": mean_before,
            "mi_mean_after": mean_after,
            "mi_level_shift_abs": abs(mean_after - mean_before),
        })

    # ln(k) envelope fit -- same as regime_fit_5k.py.
    k = np.array([1.0, 2.0, 3.0])
    x = np.log(k)
    y = np.array(windows, dtype=float)
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    for i in range(3):
        per_cp[i]["envelope_fit_residual"] = float(resid[i])

    weakest_by_gain = min(per_cp, key=lambda r: r["binary_seg_gain"])["changepoint"]
    weakest_by_level_shift = min(per_cp, key=lambda r: r["mi_level_shift_abs"])["changepoint"]
    weakest_by_residual = max(per_cp, key=lambda r: abs(r["envelope_fit_residual"]))["changepoint"]

    return {
        "per_changepoint": per_cp,
        "envelope_fit": {"a": float(a), "b": float(b)},
        "weakest_by_binary_seg_gain": weakest_by_gain,
        "weakest_by_mi_level_shift": weakest_by_level_shift,
        "worst_envelope_fit_residual": weakest_by_residual,
    }


# ── Local level / scale / shape comparison ──────────────────────────────


def excess_kurtosis(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    std_safe = np.where(std == 0, 1.0, std)
    k = np.mean((x - mean) ** 4, axis=axis) / std_safe.squeeze(axis) ** 4 - 3.0
    return np.where(std.squeeze(axis) == 0, 0.0, k)


def bootstrap_stats(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = x[idx]
    means = resamples.mean(axis=1)
    variances = resamples.var(axis=1)
    kurts = excess_kurtosis(resamples, axis=1)

    def ci(dist: np.ndarray, point: float) -> dict:
        return {"point": float(point), "ci_lo": float(np.percentile(dist, 2.5)), "ci_hi": float(np.percentile(dist, 97.5))}

    return {
        "mean": ci(means, x.mean()),
        "variance": ci(variances, x.var()),
        "kurtosis": ci(kurts, excess_kurtosis(x)),
    }


def intervals_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    return max(lo1, lo2) <= min(hi1, hi2)


def level_scale_shape_breakdown(full_gaps: np.ndarray, cp: int, rng: np.random.Generator) -> dict:
    before = full_gaps[cp - HALF_WINDOW : cp]
    after = full_gaps[cp : cp + HALF_WINDOW]
    before_stats = bootstrap_stats(before, rng, N_BOOT)
    after_stats = bootstrap_stats(after, rng, N_BOOT)

    breakdown = {}
    for key, out_key in [("mean", "level"), ("variance", "scale"), ("kurtosis", "shape")]:
        b, a = before_stats[key], after_stats[key]
        overlap = intervals_overlap(b["ci_lo"], b["ci_hi"], a["ci_lo"], a["ci_hi"])
        breakdown[out_key] = {
            "before_point": b["point"],
            "before_ci95": [b["ci_lo"], b["ci_hi"]],
            "after_point": a["point"],
            "after_ci95": [a["ci_lo"], a["ci_hi"]],
            "ci_overlap": bool(overlap),
            "shifted_significantly": not overlap,
        }
    return breakdown


# ── Fine-grained local kurtosis scan, aligned on offset-from-changepoint ──


def fine_local_scan(full_gaps: np.ndarray, cp: int) -> tuple[np.ndarray, np.ndarray]:
    domain = full_gaps[cp - HALF_WINDOW : cp + HALF_WINDOW]
    starts = np.arange(0, len(domain) - FINE_WIDTH + 1, FINE_STEP)
    kurt = np.array([excess_kurtosis(domain[s : s + FINE_WIDTH]) for s in starts])
    offsets = starts + FINE_WIDTH / 2 - HALF_WINDOW
    return offsets, kurt


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_comparison_figure(full_gaps: np.ndarray, fine_scans: list[tuple[np.ndarray, np.ndarray]], ts: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col")
    offset_x = np.arange(-HALF_WINDOW, HALF_WINDOW)
    for col, cp in enumerate(KNOWN_CHANGEPOINTS):
        domain = full_gaps[cp - HALF_WINDOW : cp + HALF_WINDOW]
        ax = axes[0, col]
        ax.plot(offset_x[:HALF_WINDOW], domain[:HALF_WINDOW], color="#94a3b8", lw=0.7)
        ax.plot(offset_x[HALF_WINDOW:], domain[HALF_WINDOW:], color=CP_COLORS[col], lw=0.7)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_title(f"{CP_LABELS[col]} -- raw gaps (offset from cp)")
        if col == 0:
            ax.set_ylabel("gap size")

        ax = axes[1, col]
        offsets, kurt = fine_scans[col]
        ax.plot(offsets, kurt, color=CP_COLORS[col], lw=1.3, marker="o", markersize=3)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_xlabel("offset from changepoint (gap-steps)")
        ax.set_title(f"fine local kurtosis (width={FINE_WIDTH}, step={FINE_STEP})")
        if col == 0:
            ax.set_ylabel("excess kurtosis")

    fig.suptitle(f"Local behavior around all three changepoints, aligned on offset [{ts}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_breakdown_table(breakdowns: list[dict], out_path: Path, ts: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    row_labels = ["level (mean)", "scale (variance)", "shape (kurtosis)"]
    cell_text = []
    for key in ["level", "scale", "shape"]:
        row = []
        for bd in breakdowns:
            b = bd[key]
            mark = "SHIFTED" if b["shifted_significantly"] else "no sig. shift"
            row.append(f"{b['before_point']:.3f} -> {b['after_point']:.3f}  ({mark})")
        cell_text.append(row)
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=CP_LABELS, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
    ax.set_title(f"Level / scale / shape breakdown (before->after, window={HALF_WINDOW}) [{ts}]", pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Auto-commit / push ──────────────────────────────────────────────────


def auto_commit_push(out_dir: Path, evidence: dict, breakdowns: list[dict], ts: str) -> None:
    shifted_summary = []
    for cp, bd in zip(KNOWN_CHANGEPOINTS, breakdowns, strict=True):
        shifted = [k for k in ["level", "scale", "shape"] if bd[k]["shifted_significantly"]]
        shifted_summary.append(f"{cp}:{'+'.join(shifted) if shifted else 'none'}")
    msg = (f"analysis: layer3 changepoint 1529 investigation {ts} -- "
           f"weakest by original gain={evidence['weakest_by_binary_seg_gain']}, "
           f"shifts={shifted_summary}")
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
    full_gaps = load_full_gaps()
    assert len(full_gaps) == 4999, f"expected 4999 raw gaps, got {len(full_gaps)}"

    print("== 1. Original detection evidence (re-run of regime_fit_5k.py's binary segmentation) ==")
    evidence = original_detection_evidence()
    for row in evidence["per_changepoint"]:
        print(f"  changepoint {row['changepoint']}: gain={row['binary_seg_gain']:.5f}  "
              f"MI level shift={row['mi_level_shift_abs']:.5f}  envelope residual={row['envelope_fit_residual']:+.2f}")
    print(f"  Weakest by binary-seg gain: {evidence['weakest_by_binary_seg_gain']}")
    print(f"  Weakest by MI level-shift magnitude: {evidence['weakest_by_mi_level_shift']}")
    print(f"  Worst envelope-fit residual (largest |residual|): {evidence['worst_envelope_fit_residual']}")

    print(f"\n== 2. Level / scale / shape breakdown (window={HALF_WINDOW} each side, n_boot={N_BOOT}) ==")
    rng = np.random.default_rng(SEED)
    breakdowns = [level_scale_shape_breakdown(full_gaps, cp, rng) for cp in KNOWN_CHANGEPOINTS]
    for cp, bd in zip(KNOWN_CHANGEPOINTS, breakdowns, strict=True):
        print(f"  changepoint {cp}:")
        for key in ["level", "scale", "shape"]:
            b = bd[key]
            print(f"    {key}: {b['before_point']:.4f} -> {b['after_point']:.4f}  "
                  f"(before CI {b['before_ci95']}, after CI {b['after_ci95']})  "
                  f"shifted_significantly={b['shifted_significantly']}")

    print(f"\n== 3. Fine-grained local kurtosis scan (width={FINE_WIDTH}, step={FINE_STEP}, +/-{HALF_WINDOW} around each cp) ==")
    fine_scans = [fine_local_scan(full_gaps, cp) for cp in KNOWN_CHANGEPOINTS]
    fine_scan_summary = []
    for cp, (offsets, kurt) in zip(KNOWN_CHANGEPOINTS, fine_scans, strict=True):
        peak_idx = int(np.argmax(kurt))
        print(f"  changepoint {cp}: local kurtosis range=[{kurt.min():.4f}, {kurt.max():.4f}]  "
              f"peak at offset={offsets[peak_idx]:+.0f}  kurtosis={kurt[peak_idx]:.4f}")
        fine_scan_summary.append({
            "changepoint": cp,
            "kurtosis_min": float(kurt.min()),
            "kurtosis_max": float(kurt.max()),
            "peak_offset": float(offsets[peak_idx]),
            "peak_kurtosis": float(kurt[peak_idx]),
        })

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = out_dir / "layer3_changepoint_comparison.png"
    plot_comparison_figure(full_gaps, fine_scans, ts, comparison_path)
    print(f"\nSaved figure to {comparison_path.relative_to(REPO_ROOT)}")

    table_path = out_dir / "layer3_changepoint_breakdown_table.png"
    plot_breakdown_table(breakdowns, table_path, ts)
    print(f"Saved figure to {table_path.relative_to(REPO_ROOT)}")

    results = {
        "timestamp": ts,
        "results_source": str(RESULTS_PATH.relative_to(REPO_ROOT)),
        "changepoints": KNOWN_CHANGEPOINTS,
        "config": {"mi_rolling_k": MI_ROLLING_K, "half_window": HALF_WINDOW, "fine_width": FINE_WIDTH,
                   "fine_step": FINE_STEP, "n_boot": N_BOOT, "seed": SEED},
        "original_detection_evidence": evidence,
        "level_scale_shape_breakdown": {
            str(cp): bd for cp, bd in zip(KNOWN_CHANGEPOINTS, breakdowns, strict=True)
        },
        "fine_local_scan": fine_scan_summary,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    auto_commit_push(out_dir, evidence, breakdowns, ts)


if __name__ == "__main__":
    main()
