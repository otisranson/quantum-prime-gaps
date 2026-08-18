"""experiments/gap_entropy_windows.py

Sliding-window entropy of the prime gap sequence, two measures side by side:

  - Shannon entropy H(N) = -sum(p_i * log2(p_i)), bits, over the observed gap-value
    distribution within each window.
  - Boltzmann-style entropy S(N) = ln(Omega), nats, where Omega is the count of
    distinct gap values observed in that window (a microstate-count entropy, not a
    probability-weighted one).

These are reported in different units on purpose (bits vs. nats) since that's what
each formula's own convention calls for -- see the "different units" note in the
summary output before comparing their magnitudes directly.

**Binning choice, stated explicitly since it changes H directly:** each distinct
integer gap value observed in a window is its own bin (a categorical/count
histogram over the discrete gap values actually seen), not a fixed-width numeric
bin. This is the natural choice here because gaps are already a small set of
discrete integers (the vast majority even, since consecutive odd primes differ by
an even number; only the single 2->3 gap is odd) -- collapsing them into
fixed-width numeric bins would merge distinct gap values together for no reason
and make H depend on an arbitrary bin-width choice instead of the data itself.
Applied identically to every window.

**Known confound, worth flagging up front:** average gap size grows ~ln(N) (PNT),
already confirmed for this dataset in hypotheses/regime_internal_wave_structure.md's
40-Regime Characterization (mean gap r=0.85, variance r=0.88, both p<0.0001). A
wider typical gap magnitude gives a fixed-size window more distinct integer values
to draw from, so both H(N) and S(N) are expected to trend upward with N even under
a "boring" gap distribution with no real structural change -- this is the same
log-growth confound already flagged in CLAUDE.md's Next Session note. The trend fit
and 2-sigma deviation flagging below exist specifically to separate that expected
growth from anything that looks like a real, localized departure from it; flagged
windows are reported as *candidates*, not confirmed regime boundaries.

**Data source:** follows this repo's cache convention (data/primes_20000.json,
falling back to data/primes_5000.json, falling back to a self-contained sieve only
if neither cache exists -- matching the sieve_primes() pattern already used in
build_prime_cache.py and several other scripts in this repo, not imported from any
of them per this repo's standalone-script convention).

**Cross-reference scope:** the "prior MI drift analysis" markers are the 3
quantum-measured-MI changepoints (windows 1529, 2501, 4211) from regime_fit_5k.py,
which are only defined/valid within the first 4999 gaps (the 5000-prime run this
detector actually ran on). This is deliberately NOT compared against the separate
39-point gap-space changepoint set from layer3_20k_scaleup.py -- CLAUDE.md is
explicit that the two changepoint sets come from different signals (quantum MI vs.
raw-gap rolling mean) and shouldn't be conflated; the request here was specifically
for the MI-drift markers.

Run: python experiments/gap_entropy_windows.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "output" / "prime"

WINDOW_SIZE = 100
STEP = 25
DEVIATION_SIGMA = 2.0

# The 3 quantum-MI-based regime changepoints (regime_fit_5k.py, binary segmentation
# on MI rolling mean, K=100, from the 5000-prime run) -- same hardcoded constant
# used throughout this repo's layer3_*.py scripts. Only valid within [0, 4999).
KNOWN_MI_CHANGEPOINTS = [1529, 2501, 4211]
MI_CHANGEPOINT_VALID_RANGE = 4999


def sieve_primes(count: int) -> list[int]:
    """Sieve of Eratosthenes, doubling the search limit until enough primes are
    found. Same pattern used in build_prime_cache.py and elsewhere in this repo."""
    limit = 10
    while True:
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        primes = [i for i, p in enumerate(is_p) if p]
        if len(primes) >= count:
            return primes[:count]
        limit *= 2


def load_gaps() -> tuple[np.ndarray, str]:
    """Prefer the 20k-prime cache, fall back to the 5k cache, fall back to a
    fresh sieve only if neither cache file exists on disk."""
    for path, label in [(DATA_DIR / "primes_20000.json", "cache:20000"),
                         (DATA_DIR / "primes_5000.json", "cache:5000")]:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return np.array(data["gaps"], dtype=np.int64), label
    primes = sieve_primes(20000)
    gaps = np.array([primes[i + 1] - primes[i] for i in range(len(primes) - 1)], dtype=np.int64)
    return gaps, "fresh_sieve:20000"


def window_entropy(gaps: np.ndarray, start: int, size: int) -> tuple[float, float, int]:
    window = gaps[start:start + size]
    values, counts = np.unique(window, return_counts=True)
    probs = counts / counts.sum()
    shannon_h = float(-np.sum(probs * np.log2(probs)))
    omega = len(values)
    boltzmann_s = float(np.log(omega))
    return shannon_h, boltzmann_s, omega


def fit_log_linear(centers: np.ndarray, values: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """H(N) ~ a*ln(N+2) + b, matching the ln(global_index+2) convention already
    used in layer3_kurtosis_robustness.py's log_fit_detrend. Returns (a, b,
    predicted values, R^2)."""
    x = np.log(centers + 2)
    a, b = np.polyfit(x, values, 1)
    predicted = a * x + b
    residuals = values - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((values - values.mean())**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), predicted, r_squared


def main() -> None:
    gaps, source_label = load_gaps()
    n_gaps = len(gaps)
    print(f"Loaded {n_gaps} gaps from {source_label}")

    starts = list(range(0, n_gaps - WINDOW_SIZE + 1, STEP))
    centers = np.array([s + WINDOW_SIZE / 2 for s in starts])
    shannon_vals = np.empty(len(starts))
    boltzmann_vals = np.empty(len(starts))
    omega_vals = np.empty(len(starts), dtype=np.int64)
    for i, s in enumerate(starts):
        h, sB, omega = window_entropy(gaps, s, WINDOW_SIZE)
        shannon_vals[i] = h
        boltzmann_vals[i] = sB
        omega_vals[i] = omega

    print(f"{len(starts)} windows (size={WINDOW_SIZE}, step={STEP})")

    # ── Trend fit + deviation flagging on H(N) ──────────────────────────────
    a, b, predicted, r_squared = fit_log_linear(centers, shannon_vals)
    residuals = shannon_vals - predicted
    resid_std = float(residuals.std())
    flagged_mask = np.abs(residuals) > DEVIATION_SIGMA * resid_std
    flagged_indices = np.where(flagged_mask)[0]
    flagged_centers = centers[flagged_indices]
    flagged_starts = [starts[i] for i in flagged_indices]
    flagged_ends = [s + WINDOW_SIZE for s in flagged_starts]

    print(f"\nLog-linear fit: H(N) ~ {a:.6f} * ln(N+2) + {b:.6f}  (R^2={r_squared:.4f})")
    print(f"Residual std: {resid_std:.4f}, flagging |residual| > {DEVIATION_SIGMA}*std")
    print(f"Flagged windows: {len(flagged_indices)} of {len(starts)}")

    hs_corr = float(np.corrcoef(shannon_vals, boltzmann_vals)[0, 1])
    print(f"Pearson r(H, S) across windows: {hs_corr:.4f} (expected high -- both track window value-diversity)")

    # ── Cross-reference against known MI-drift changepoints ────────────────
    in_range_flags = [(s, e, c) for s, e, c in zip(flagged_starts, flagged_ends, flagged_centers, strict=True)
                       if s < MI_CHANGEPOINT_VALID_RANGE]
    overlap_report = []
    for cp in KNOWN_MI_CHANGEPOINTS:
        containing = [(s, e, c) for s, e, c in in_range_flags if s <= cp < e]
        if in_range_flags:
            nearest = min(in_range_flags, key=lambda t: abs(t[2] - cp))
            nearest_dist = float(abs(nearest[2] - cp))
        else:
            nearest_dist = None
        overlap_report.append({
            "changepoint": cp,
            "inside_a_flagged_window": len(containing) > 0,
            "containing_flagged_window_centers": [float(c) for _, _, c in containing],
            "nearest_flagged_window_center": float(nearest[2]) if in_range_flags else None,
            "nearest_distance": nearest_dist,
        })

    print(f"\n== Cross-reference vs. known MI-drift changepoints (valid range: gap-index < {MI_CHANGEPOINT_VALID_RANGE}) ==")
    print(f"Flagged windows starting within valid range: {len(in_range_flags)} of {len(flagged_indices)} total flagged")
    for rep in overlap_report:
        status = "INSIDE a flagged window" if rep["inside_a_flagged_window"] else "no overlap"
        near = f"(nearest flagged center: {rep['nearest_flagged_window_center']}, dist={rep['nearest_distance']})" \
            if rep["nearest_flagged_window_center"] is not None else "(no in-range flagged windows to compare)"
        print(f"  changepoint {rep['changepoint']}: {status} {near}")

    # ── Plot ─────────────────────────────────────────────────────────────
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax1.plot(centers, shannon_vals, color="#4c72b0", lw=1.0, label="H(N) -- Shannon entropy (bits)")
    ax1.plot(centers, predicted, color="#d1495b", lw=1.5, ls="--", label=f"log-linear fit (R^2={r_squared:.3f})")
    if len(flagged_centers) > 0:
        ax1.scatter(flagged_centers, shannon_vals[flagged_indices], color="#d1495b", zorder=5, s=30,
                     label=f"flagged (|resid|>{DEVIATION_SIGMA}$\\sigma$), n={len(flagged_indices)}")
    for cp in KNOWN_MI_CHANGEPOINTS:
        ax1.axvline(cp, color="#2a9d5c", lw=1.2, ls=":", alpha=0.8)
        ax1.text(cp, ax1.get_ylim()[1], f" MI cp {cp}", color="#2a9d5c", va="top", ha="left", fontsize=8)
    ax1.set_ylabel("H(N) [bits]")
    ax1.set_title(f"Shannon entropy vs. window center, window={WINDOW_SIZE} step={STEP} [{ts}]")
    ax1.legend(fontsize=8, loc="lower right")

    ax2.plot(centers, boltzmann_vals, color="#e08214", lw=1.0, label="S(N) = ln(Omega) -- Boltzmann-style entropy (nats)")
    for cp in KNOWN_MI_CHANGEPOINTS:
        ax2.axvline(cp, color="#2a9d5c", lw=1.2, ls=":", alpha=0.8)
    ax2.set_xlabel("gap index (window center)")
    ax2.set_ylabel("S(N) [nats]")
    ax2.set_title("Boltzmann-style entropy (distinct gap-value count) vs. window center")
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    png_path = out_dir / "gap_entropy_windows.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {png_path.relative_to(REPO_ROOT)}")

    # ── Markdown summary ─────────────────────────────────────────────────
    md_lines = [
        f"# Gap-sequence sliding-window entropy -- {ts}",
        "",
        f"Source: `{source_label}` ({n_gaps} gaps). Window size={WINDOW_SIZE}, step={STEP}, "
        f"{len(starts)} windows total.",
        "",
        "## Binning methodology",
        "",
        "Each distinct integer gap value observed within a window is its own histogram bin "
        "(a categorical count over observed values, not fixed-width numeric binning) -- applied "
        "identically across every window. Shannon entropy H(N) is reported in bits (log base 2); "
        "Boltzmann-style entropy S(N)=ln(Omega) is reported in nats (natural log) -- these use "
        "different units by the definitions requested and should not be compared directly without "
        "accounting for that.",
        "",
        "## Entropy growth rate",
        "",
        f"Log-linear fit: **H(N) ~ {a:.6f} * ln(N+2) + {b:.6f}** (R^2={r_squared:.4f})",
        f"Residual std: {resid_std:.4f}; windows flagged where |residual| > {DEVIATION_SIGMA} sigma.",
        f"Pearson r(H, S) across all windows: {hs_corr:.4f} (both track window value-diversity by "
        "construction, so a high correlation here is an expected consistency check, not an "
        "independent finding).",
        "",
        "**Confound note:** average gap size is already known to grow ~ln(N) for this dataset "
        "(hypotheses/regime_internal_wave_structure.md, 40-Regime Characterization: mean gap "
        "r=0.85, p<0.0001). A wider typical gap gives each fixed-size window more distinct integer "
        "values to draw from, so upward H(N)/S(N) trend with N is expected under a structurally "
        "boring gap distribution -- the log-linear fit above is intended to characterize and "
        "remove exactly that expected growth before anything is flagged as unusual.",
        "",
        f"## Flagged windows (candidate regime boundaries, n={len(flagged_indices)})",
        "",
        "Flagged because H(N) deviates from the fitted trend by more than "
        f"{DEVIATION_SIGMA} residual standard deviations. These are candidates, not confirmed "
        "regime boundaries.",
        "",
    ]
    if len(flagged_indices) > 0:
        md_lines += ["| window start | window end | center | H(N) | fit | residual |",
                     "|---|---|---|---|---|---|"]
        for i in flagged_indices:
            md_lines.append(f"| {starts[i]} | {starts[i] + WINDOW_SIZE} | {centers[i]:.1f} | "
                             f"{shannon_vals[i]:.4f} | {predicted[i]:.4f} | {residuals[i]:+.4f} |")
    else:
        md_lines.append("None.")
    md_lines += [
        "",
        "## Overlap with known MI-drift regime markers",
        "",
        f"Comparison scope: only the 3 quantum-MI-based changepoints from `regime_fit_5k.py` "
        f"(windows {KNOWN_MI_CHANGEPOINTS}), valid only within gap-index < {MI_CHANGEPOINT_VALID_RANGE} "
        "(the 5000-prime run this detector actually ran on). This is deliberately not compared "
        "against the separate 39-point gap-space changepoint set from `layer3_20k_scaleup.py` -- "
        "CLAUDE.md documents those as a different signal (raw-gap rolling mean, not MI) and warns "
        "against conflating the two sets.",
        "",
        f"{len(in_range_flags)} of the {len(flagged_indices)} flagged windows fall within the valid "
        "comparison range.",
        "",
        "| MI changepoint | inside a flagged window? | nearest flagged window center | distance |",
        "|---|---|---|---|",
    ]
    for rep in overlap_report:
        inside = "yes" if rep["inside_a_flagged_window"] else "no"
        near_c = f"{rep['nearest_flagged_window_center']:.1f}" if rep["nearest_flagged_window_center"] is not None else "n/a"
        dist = f"{rep['nearest_distance']:.1f}" if rep["nearest_distance"] is not None else "n/a"
        md_lines.append(f"| {rep['changepoint']} | {inside} | {near_c} | {dist} |")
    n_inside = sum(1 for rep in overlap_report if rep["inside_a_flagged_window"])
    md_lines += [
        "",
        f"**Result, stated plainly:** {n_inside} of {len(KNOWN_MI_CHANGEPOINTS)} known MI-drift "
        "changepoints fall inside a flagged entropy-deviation window. Reported as overlap/"
        "non-overlap only -- no null distribution was computed for how often a random window "
        "would contain a given changepoint by chance, so this should not be read as a significance "
        "claim either way, only as a descriptive comparison.",
        "",
    ]
    md_path = out_dir / "gap_entropy_windows_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved summary to {md_path.relative_to(REPO_ROOT)}")

    results = {
        "timestamp": ts,
        "source": source_label,
        "n_gaps": int(n_gaps),
        "window_size": WINDOW_SIZE,
        "step": STEP,
        "n_windows": len(starts),
        "binning": "one bin per distinct observed integer gap value, identical across windows",
        "shannon_units": "bits (log2)",
        "boltzmann_units": "nats (ln)",
        "trend_fit": {"form": "H ~ a*ln(N+2) + b", "a": round(a, 6), "b": round(b, 6), "r_squared": round(r_squared, 6)},
        "residual_std": round(resid_std, 6),
        "deviation_sigma_threshold": DEVIATION_SIGMA,
        "n_flagged": int(len(flagged_indices)),
        "flagged_windows": [
            {"start": int(starts[i]), "end": int(starts[i] + WINDOW_SIZE), "center": float(centers[i]),
             "H": round(float(shannon_vals[i]), 6), "fit": round(float(predicted[i]), 6),
             "residual": round(float(residuals[i]), 6)}
            for i in flagged_indices
        ],
        "hs_correlation": round(hs_corr, 6),
        "known_mi_changepoints": KNOWN_MI_CHANGEPOINTS,
        "mi_changepoint_valid_range": MI_CHANGEPOINT_VALID_RANGE,
        "overlap_report": overlap_report,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    msg = (f"experiment: gap entropy sliding windows {ts} -- "
           f"H(N) growth rate a={a:.4f} (R^2={r_squared:.3f}), flagged={len(flagged_indices)} windows, "
           f"MI-changepoint overlap={n_inside}/{len(KNOWN_MI_CHANGEPOINTS)}")
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
