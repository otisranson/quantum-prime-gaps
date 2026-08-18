"""layer3_kurtosis_robustness.py

Robustness check on the per-regime characterization follow-up
(hypotheses/regime_internal_wave_structure.md, "Follow-up -- Per-Regime
Characterization"): that script found excess kurtosis climbing across the
three regimes (2.42 -> 3.22 -> 6.04) while 2-sigma spike density stayed flat
(0.0517 -> 0.0453 -> 0.0427) -- outlier gaps aren't more frequent later, but
they're more extreme when they occur. This script tests whether that pattern
is robust or an artifact, along five independent axes:

  1. Bootstrap CIs on kurtosis and spike density per regime (resample with
     replacement, check whether the CIs actually separate the regimes).
  2. Threshold robustness -- does "flat density, rising kurtosis" hold at
     2.0, 2.5, and 3.0 sigma, or only at the one threshold originally used?
  3. Sliding-window kurtosis (width=500, step=100) across the full
     changepoint-bounded range [0, 4211), ignoring regime boundaries --
     tests whether kurtosis rises smoothly with position (a general drift)
     or jumps specifically at the three changepoints (regime-specific).
  4. Background-growth control -- divide out a local log-fit trend
     (gap ~ a*ln(global_index+2)+b, fit separately per regime) before
     recomputing kurtosis, to see whether the rise survives ordinary
     gap-size growth being accounted for.
  5. Max/mean gap size per regime reported alongside, for growth context.

Same regime definitions as every other Layer 3 script in this file:
[0,1529), [1529,2501), [2501,4211) in raw gap-index space, post-4211 tail
excluded (no closing changepoint).

Run: python layer3_kurtosis_robustness.py
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
REGIME_LABELS = ["regime 0", "regime 1", "regime 2"]
REGIME_COLORS = ["#4c72b0", "#e08214", "#2a9d5c"]

N_BOOT = 2000
SEED = 42
THRESHOLDS = [2.0, 2.5, 3.0]
SLIDE_WIDTH = 500
SLIDE_STEP = 100

OUT_ROOT = REPO_ROOT / "output" / "prime"


def load_full_gaps() -> np.ndarray:
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    per_window = data["per_window"]
    return np.array([r["gaps"][0] for r in per_window] + per_window[-1]["gaps"][1:])


def excess_kurtosis(x: np.ndarray, axis: int = -1) -> np.ndarray:
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    std_safe = np.where(std == 0, 1.0, std)
    k = np.mean((x - mean) ** 4, axis=axis) / std_safe.squeeze(axis) ** 4 - 3.0
    return np.where(std.squeeze(axis) == 0, 0.0, k)


def spike_density(x: np.ndarray, z_thresh: float, axis: int = -1) -> np.ndarray:
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    std_safe = np.where(std == 0, 1.0, std)
    z = (x - mean) / std_safe
    density = np.mean(np.abs(z) > z_thresh, axis=axis)
    return np.where(std.squeeze(axis) == 0, 0.0, density)


def bootstrap_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = x[idx]
    kurt_dist = excess_kurtosis(resamples, axis=1)
    density_dist = spike_density(resamples, 2.0, axis=1)
    return {
        "kurtosis": {
            "point": float(excess_kurtosis(x)),
            "ci_lo": float(np.percentile(kurt_dist, 2.5)),
            "ci_hi": float(np.percentile(kurt_dist, 97.5)),
            "dist": kurt_dist,
        },
        "spike_density": {
            "point": float(spike_density(x, 2.0)),
            "ci_lo": float(np.percentile(density_dist, 2.5)),
            "ci_hi": float(np.percentile(density_dist, 97.5)),
            "dist": density_dist,
        },
    }


def intervals_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    return max(lo1, lo2) <= min(hi1, hi2)


def log_fit_detrend(x: np.ndarray, global_start: int) -> tuple[np.ndarray, float, float]:
    """Fit gap ~ a*ln(global_index+2)+b (least squares, local to this regime)
    and return the ratio series x/trend along with the fit params."""
    global_idx = np.arange(global_start, global_start + len(x))
    log_idx = np.log(global_idx + 2)
    a, b = np.polyfit(log_idx, x, 1)
    trend = a * log_idx + b
    trend_safe = np.where(trend > 0, trend, np.nan)
    ratio = x / trend_safe
    ratio = ratio[np.isfinite(ratio)]
    return ratio, float(a), float(b)


def sliding_window_kurtosis(x: np.ndarray, width: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(x) - width + 1, step)
    kurt = np.array([excess_kurtosis(x[s : s + width]) for s in starts])
    centers = starts + width / 2
    return centers, kurt


def changepoint_jump_ratio(centers: np.ndarray, kurt: np.ndarray, changepoints: list[int]) -> dict:
    step_positions = (centers[:-1] + centers[1:]) / 2
    steps = np.diff(kurt)
    median_abs_step = float(np.median(np.abs(steps))) if len(steps) else 0.0
    report = {}
    for cp in changepoints:
        i = int(np.argmin(np.abs(step_positions - cp)))
        jump = float(abs(steps[i]))
        ratio = jump / median_abs_step if median_abs_step > 0 else float("nan")
        report[cp] = {"jump": jump, "median_abs_step_elsewhere": median_abs_step, "ratio": ratio}
    return report


def main() -> None:
    full_gaps = load_full_gaps()
    assert len(full_gaps) == 4999, f"expected 4999 raw gaps, got {len(full_gaps)}"

    bounds = [(0, KNOWN_CHANGEPOINTS[0]), (KNOWN_CHANGEPOINTS[0], KNOWN_CHANGEPOINTS[1]),
              (KNOWN_CHANGEPOINTS[1], KNOWN_CHANGEPOINTS[2])]
    regimes = [full_gaps[a:b] for a, b in bounds]
    assert sum(len(r) for r in regimes) == KNOWN_CHANGEPOINTS[2]
    print(f"Regime bounds: {bounds}")

    rng = np.random.default_rng(SEED)

    # ── 1. Bootstrap CIs ─────────────────────────────────────────────────
    print(f"\n== Bootstrap CIs (n_boot={N_BOOT}, seed={SEED}) ==")
    boot = [bootstrap_ci(r, rng, N_BOOT) for r in regimes]
    for i, b in enumerate(boot):
        k, d = b["kurtosis"], b["spike_density"]
        print(f"  {REGIME_LABELS[i]}: kurtosis={k['point']:.4f}  95% CI=[{k['ci_lo']:.4f}, {k['ci_hi']:.4f}]   "
              f"spike_density={d['point']:.4f}  95% CI=[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]")

    overlap_report = {}
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        k_overlap = intervals_overlap(boot[i]["kurtosis"]["ci_lo"], boot[i]["kurtosis"]["ci_hi"],
                                       boot[j]["kurtosis"]["ci_lo"], boot[j]["kurtosis"]["ci_hi"])
        d_overlap = intervals_overlap(boot[i]["spike_density"]["ci_lo"], boot[i]["spike_density"]["ci_hi"],
                                       boot[j]["spike_density"]["ci_lo"], boot[j]["spike_density"]["ci_hi"])
        overlap_report[f"{i}v{j}"] = {"kurtosis_ci_overlap": bool(k_overlap), "spike_density_ci_overlap": bool(d_overlap)}
        print(f"  regime {i} vs {j}: kurtosis CI overlap={k_overlap}  spike-density CI overlap={d_overlap}")

    # ── 2. Threshold robustness ─────────────────────────────────────────
    print("\n== Threshold robustness ==")
    threshold_table = {}
    for thresh in THRESHOLDS:
        row = [float(spike_density(r, thresh)) for r in regimes]
        threshold_table[thresh] = row
        print(f"  sigma={thresh}: " + "  ".join(f"{REGIME_LABELS[i]}={row[i]:.4f}" for i in range(3)))

    # ── 3. Sliding window kurtosis over [0, 4211) ───────────────────────
    print(f"\n== Sliding window kurtosis (width={SLIDE_WIDTH}, step={SLIDE_STEP}, range=[0,{KNOWN_CHANGEPOINTS[2]})) ==")
    slide_domain = full_gaps[: KNOWN_CHANGEPOINTS[2]]
    centers, kurt_series = sliding_window_kurtosis(slide_domain, SLIDE_WIDTH, SLIDE_STEP)
    print(f"  {len(centers)} windows, kurtosis range=[{kurt_series.min():.4f}, {kurt_series.max():.4f}]")
    jump_report = changepoint_jump_ratio(centers, kurt_series, KNOWN_CHANGEPOINTS)
    for cp, rep in jump_report.items():
        print(f"  changepoint {cp}: step jump={rep['jump']:.4f}  median |step| elsewhere={rep['median_abs_step_elsewhere']:.4f}  "
              f"ratio={rep['ratio']:.2f}x")

    # ── 4. Background-growth (log-fit) detrending ───────────────────────
    print("\n== Background-growth control (local log-fit detrend) ==")
    detrended_kurt = []
    fit_params = []
    for i, r in enumerate(regimes):
        global_start = bounds[i][0]
        ratio, a, b = log_fit_detrend(r, global_start)
        dk = float(excess_kurtosis(ratio))
        detrended_kurt.append(dk)
        fit_params.append({"a": a, "b": b})
        print(f"  {REGIME_LABELS[i]}: trend fit a={a:.4f} b={b:.4f}  "
              f"original kurtosis={float(excess_kurtosis(r)):.4f}  detrended kurtosis={dk:.4f}")

    # ── 5/6. Max/mean context ───────────────────────────────────────────
    print("\n== Max / mean gap size per regime ==")
    max_gaps = [float(r.max()) for r in regimes]
    mean_gaps = [float(r.mean()) for r in regimes]
    for i in range(3):
        print(f"  {REGIME_LABELS[i]}: max={max_gaps[i]:.1f}  mean={mean_gaps[i]:.4f}")

    # ── Output ───────────────────────────────────────────────────────────
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overview figure: bootstrap CIs, threshold comparison, detrended comparison.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    x = np.arange(3)
    kurt_points = [b["kurtosis"]["point"] for b in boot]
    kurt_los = [b["kurtosis"]["point"] - b["kurtosis"]["ci_lo"] for b in boot]
    kurt_his = [b["kurtosis"]["ci_hi"] - b["kurtosis"]["point"] for b in boot]
    ax.bar(x, kurt_points, color=REGIME_COLORS, alpha=0.85, yerr=[kurt_los, kurt_his], capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_ylabel("excess kurtosis")
    ax.set_title(f"Bootstrap 95% CI -- kurtosis (n_boot={N_BOOT})")

    ax = axes[0, 1]
    dens_points = [b["spike_density"]["point"] for b in boot]
    dens_los = [b["spike_density"]["point"] - b["spike_density"]["ci_lo"] for b in boot]
    dens_his = [b["spike_density"]["ci_hi"] - b["spike_density"]["point"] for b in boot]
    ax.bar(x, dens_points, color=REGIME_COLORS, alpha=0.85, yerr=[dens_los, dens_his], capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_ylabel("spike density (|z|>2)")
    ax.set_title(f"Bootstrap 95% CI -- spike density (n_boot={N_BOOT})")

    ax = axes[1, 0]
    width = 0.25
    for i in range(3):
        vals = [threshold_table[t][i] for t in THRESHOLDS]
        ax.bar(np.arange(len(THRESHOLDS)) + (i - 1) * width, vals, width, color=REGIME_COLORS[i], label=REGIME_LABELS[i])
    ax.set_xticks(np.arange(len(THRESHOLDS)))
    ax.set_xticklabels([f"{t}sigma" for t in THRESHOLDS])
    ax.set_ylabel("spike density")
    ax.set_title("Threshold robustness -- spike density at 2.0 / 2.5 / 3.0 sigma")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    orig_kurt = [float(excess_kurtosis(r)) for r in regimes]
    bar_w = 0.35
    ax.bar(x - bar_w / 2, orig_kurt, bar_w, label="original kurtosis", color="#94a3b8")
    ax.bar(x + bar_w / 2, detrended_kurt, bar_w, label="detrended (log-fit removed)", color="#334155")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_ylabel("excess kurtosis")
    ax.set_title("Background-growth control -- original vs. detrended kurtosis")
    ax.legend(fontsize=8)

    fig.suptitle(f"Layer 3 kurtosis robustness -- overview [{ts}]")
    fig.tight_layout()
    overview_path = out_dir / "layer3_kurtosis_robustness_overview.png"
    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {overview_path.relative_to(REPO_ROOT)}")

    # Sliding-window figure.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(centers, kurt_series, color="#4c72b0", lw=1.3, marker="o", markersize=3)
    for cp in KNOWN_CHANGEPOINTS:
        ax.axvline(cp, color="#d1495b", lw=1.5, ls="--")
        ax.text(cp, ax.get_ylim()[1], f" {cp}", color="#d1495b", va="top", ha="left", fontsize=9)
    ax.set_xlim(0, KNOWN_CHANGEPOINTS[2])
    ax.set_xlabel("gap index (window center)")
    ax.set_ylabel(f"excess kurtosis (width={SLIDE_WIDTH})")
    ax.set_title(f"Sliding-window kurtosis vs. position, regime boundaries ignored [{ts}]")
    fig.tight_layout()
    slide_path = out_dir / "layer3_kurtosis_sliding_window.png"
    fig.savefig(slide_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {slide_path.relative_to(REPO_ROOT)}")

    results = {
        "timestamp": ts,
        "results_source": str(RESULTS_PATH.relative_to(REPO_ROOT)),
        "regime_bounds": bounds,
        "config": {"n_boot": N_BOOT, "seed": SEED, "thresholds": THRESHOLDS,
                   "slide_width": SLIDE_WIDTH, "slide_step": SLIDE_STEP},
        "bootstrap": [
            {
                "regime": REGIME_LABELS[i],
                "kurtosis_point": round(boot[i]["kurtosis"]["point"], 6),
                "kurtosis_ci95": [round(boot[i]["kurtosis"]["ci_lo"], 6), round(boot[i]["kurtosis"]["ci_hi"], 6)],
                "spike_density_point": round(boot[i]["spike_density"]["point"], 6),
                "spike_density_ci95": [round(boot[i]["spike_density"]["ci_lo"], 6), round(boot[i]["spike_density"]["ci_hi"], 6)],
            }
            for i in range(3)
        ],
        "ci_overlap": overlap_report,
        "threshold_table": {str(t): [round(v, 6) for v in threshold_table[t]] for t in THRESHOLDS},
        "sliding_window": {
            "n_windows": len(centers),
            "kurtosis_min": round(float(kurt_series.min()), 6),
            "kurtosis_max": round(float(kurt_series.max()), 6),
            "changepoint_jump_ratios": {str(cp): {k: round(v, 6) if isinstance(v, float) else v for k, v in rep.items()}
                                         for cp, rep in jump_report.items()},
        },
        "background_growth_control": [
            {
                "regime": REGIME_LABELS[i],
                "fit_a": round(fit_params[i]["a"], 6),
                "fit_b": round(fit_params[i]["b"], 6),
                "original_kurtosis": round(orig_kurt[i], 6),
                "detrended_kurtosis": round(detrended_kurt[i], 6),
            }
            for i in range(3)
        ],
        "size_context": [
            {"regime": REGIME_LABELS[i], "max_gap": max_gaps[i], "mean_gap": round(mean_gaps[i], 6)}
            for i in range(3)
        ],
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    msg = (f"analysis: layer3 kurtosis robustness {ts} -- "
           f"kurtosis CIs {[round(k, 2) for k in kurt_points]}, "
           f"overlap(0v2)={overlap_report['0v2']['kurtosis_ci_overlap']}, "
           f"detrended kurtosis {[round(k, 2) for k in detrended_kurt]}")
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
