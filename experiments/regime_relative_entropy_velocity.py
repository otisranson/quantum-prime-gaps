"""experiments/regime_relative_entropy_velocity.py

Tests the "burst then decay" (sawtooth) hypothesis on entropy velocity
within each gap-space regime: does dH/dN' spike near a regime's start and
decay exponentially toward its end, or is that shape indistinguishable from
noise?

**H(N) series:** recomputed deterministically from data/primes_20000.json
(same window_size=100/step=25/one-bin-per-observed-gap-value binning as
experiments/gap_entropy_windows.py), verified against the stored
flagged-window values and trend fit in
output/prime/20260818_224457/results.json -- same pattern as
experiments/entropy_velocity_residual_decay.py, copied here rather than
imported per this repo's standalone-script convention.

**Regime boundaries:** the 39-point gap-space changepoint set
(output/prime/20260818_015045/results.json, layer3_20k_scaleup.py's
data-driven binary segmentation on the raw-gap rolling mean) -- explicitly
NOT the sim-based MI set, per instruction, since gap-space is the
higher-count, already-validated set. 39 changepoints bound 40 regimes
(leading and trailing segments included, same convention as the "40-Regime
Characterization" already in hypotheses/regime_internal_wave_structure.md):
regime i = [boundary[i], boundary[i+1]) in raw gap-index space, boundary[0]=0
and boundary[40]=domain end.

**Minimum window count (>=8), justified:** the exponential-decay model has 2
free parameters (v0, k); a flat model has 1. Fewer than ~3 points make any
2-parameter fit degenerate (perfect or undefined), and very few points give
essentially no power to distinguish "decay" from "noise that happens to
slope down." >=8 local-velocity points per regime gives at least 6 residual
degrees of freedom for the exponential fit (8-2) and 7 for the flat model
(8-1) -- thin, but enough for a real per-regime R^2 comparison rather than a
guaranteed-degenerate one. Regimes below this are excluded and the excluded
count is reported, not silently dropped.

**Fit direction matters, not just R^2:** v(N') = v0*exp(-k*N') can fit
growth as well as decay depending on the sign of the fitted k. A regime only
counts as showing the "burst then decay" pattern if the exponential model
beats the flat baseline on R^2 *and* the fitted k is positive (genuine
decay) -- a high-R^2 fit with negative k (growth) is not evidence for this
hypothesis and is reported as such, not folded into the "wins" count.

**Permutation null for the aggregate fraction:** shuffles the *order* of
the 40 real regime lengths (a random permutation of the same length
multiset, laid out cumulatively from N=0) rather than drawing arbitrary
boundary positions -- this holds the empirical length distribution fixed
(so the null accounts for the fact that shorter regimes give noisier,
lower-power fits) and asks whether the *specific real boundary positions*
matter, or whether any regime segmentation with the same length statistics
would show the same fraction of "decay-wins" regimes by chance alone.

Run: python experiments/regime_relative_entropy_velocity.py
"""

from __future__ import annotations

import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

REPO_ROOT = Path(__file__).parent.parent
DATA_PATH = REPO_ROOT / "data/primes_20000.json"
ENTROPY_RESULTS_PATH = REPO_ROOT / "output/prime/20260818_224457/results.json"
GAP_CHANGEPOINTS_PATH = REPO_ROOT / "output/prime/20260818_015045/results.json"
OUT_ROOT = REPO_ROOT / "output" / "prime"

WINDOW_SIZE = 100
STEP = 25
MIN_WINDOWS_PER_REGIME = 8
N_PERM = 1000
SEED = 42


# ── H(N) recomputation (identical to experiments/entropy_velocity_residual_decay.py) ─


def load_gaps() -> np.ndarray:
    with open(DATA_PATH) as f:
        data = json.load(f)
    return np.array(data["gaps"], dtype=np.int64)


def window_shannon_entropy(gaps: np.ndarray, start: int, size: int) -> float:
    window = gaps[start:start + size]
    _, counts = np.unique(window, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def recompute_full_series(gaps: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    starts = list(range(0, len(gaps) - WINDOW_SIZE + 1, STEP))
    centers = np.array([s + WINDOW_SIZE / 2 for s in starts])
    shannon = np.array([window_shannon_entropy(gaps, s, WINDOW_SIZE) for s in starts])
    return centers, shannon, starts


def verify_matches_stored_run(centers: np.ndarray, shannon: np.ndarray, starts: list[int],
                               stored: dict) -> dict:
    assert len(starts) == stored["n_windows"]
    index_by_start = {s: i for i, s in enumerate(starts)}
    for fw in stored["flagged_windows"]:
        idx = index_by_start[fw["start"]]
        assert abs(shannon[idx] - fw["H"]) < 1e-4, (
            f"recomputed H at start={fw['start']} doesn't match stored value -- "
            "recomputation is not reproducing the original run"
        )
    a, b = np.polyfit(np.log(centers + 2), shannon, 1)
    assert abs(a - stored["trend_fit"]["a"]) < 1e-3
    print(f"Verified: recomputed full series ({len(starts)} windows) exactly reproduces "
          f"stored flagged-window H values and trend fit (a={a:.6f}).")
    return {"a": float(a), "b": float(b)}


# ── Regime segmentation ─────────────────────────────────────────────────


def load_regime_boundaries(domain_end: float) -> list[int]:
    with open(GAP_CHANGEPOINTS_PATH) as f:
        data = json.load(f)
    cps = sorted(c["position"] for c in data["changepoints"])
    assert len(cps) == 39 == data["n_changepoints"]
    return [0] + cps + [int(domain_end) + 1]


def assign_windows_to_regimes(centers: np.ndarray, boundaries: list[int]) -> np.ndarray:
    regime_idx = np.searchsorted(boundaries, centers, side="right") - 1
    regime_idx = np.clip(regime_idx, 0, len(boundaries) - 2)
    return regime_idx


# ── Fitting ──────────────────────────────────────────────────────────────


def r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def exp_model(n_prime: np.ndarray, v0: float, k: float) -> np.ndarray:
    return v0 * np.exp(-k * n_prime)


def fit_regime_velocity(n_prime: np.ndarray, velocity: np.ndarray) -> dict:
    flat_pred = np.full_like(velocity, velocity.mean())
    flat_r2 = r_squared(velocity, flat_pred)

    v0_guess = velocity[0] if abs(velocity[0]) > 1e-9 else (velocity.mean() or 1e-6)
    converged = False
    exp_r2 = float("-inf")
    v0_fit = k_fit = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # silence exp() overflow spam; scipy
                                                            # turns the resulting nan/inf into a
                                                            # ValueError, which is caught below
        warnings.simplefilter("error", OptimizeWarning)
        try:
            popt, _ = curve_fit(
                exp_model, n_prime, velocity, p0=[v0_guess, 0.001],
                maxfev=5000, bounds=([-np.inf, -1.0], [np.inf, 1.0]),
            )
            v0_fit, k_fit = float(popt[0]), float(popt[1])
            exp_pred = exp_model(n_prime, v0_fit, k_fit)
            exp_r2 = r_squared(velocity, exp_pred)
            converged = True
        except (RuntimeError, OptimizeWarning, ValueError):
            converged = False

    decay_wins = converged and exp_r2 > flat_r2 and k_fit is not None and k_fit > 0

    return {
        "flat_r2": flat_r2,
        "exp_r2": exp_r2 if converged else None,
        "v0": v0_fit,
        "k": k_fit,
        "converged": converged,
        "decay_wins": decay_wins,
    }


def run_regime_pass(centers: np.ndarray, shannon: np.ndarray, boundaries: list[int],
                     min_windows: int) -> list[dict]:
    regime_idx = assign_windows_to_regimes(centers, boundaries)
    n_regimes = len(boundaries) - 1
    results = []
    for r in range(n_regimes):
        mask = regime_idx == r
        n_windows = int(mask.sum())
        if n_windows < min_windows:
            results.append({"regime": r, "n_windows": n_windows, "qualifies": False})
            continue
        regime_centers = centers[mask]
        regime_h = shannon[mask]
        n_prime = regime_centers - boundaries[r]
        velocity = np.gradient(regime_h, n_prime)
        fit = fit_regime_velocity(n_prime, velocity)
        results.append({
            "regime": r, "n_windows": n_windows, "qualifies": True,
            "regime_start": int(boundaries[r]), "regime_end": int(boundaries[r + 1]),
            "n_prime": n_prime, "velocity": velocity, **fit,
        })
    return results


# ── Permutation null (shuffle regime-length order, matched spacing) ────


def permutation_null_fraction(centers: np.ndarray, shannon: np.ndarray, lengths: np.ndarray,
                               min_windows: int, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    null_fractions = np.empty(n_perm)
    for p in range(n_perm):
        shuffled_lengths = rng.permutation(lengths)
        synth_boundaries = [0] + list(np.cumsum(shuffled_lengths))  # sum of permuted lengths
                                                                      # equals the real domain span exactly
        pass_results = run_regime_pass(centers, shannon, [int(b) for b in synth_boundaries], min_windows)
        qualifying = [r for r in pass_results if r["qualifies"]]
        n_decay = sum(1 for r in qualifying if r["decay_wins"])
        null_fractions[p] = n_decay / len(qualifying) if qualifying else 0.0
    return null_fractions


def auto_commit_push(out_dir: Path, fraction: float, null_p: float, ts: str) -> None:
    msg = (f"experiment: regime-relative entropy velocity {ts} -- "
           f"burst-decay fraction={fraction:.3f}, permutation-null p={null_p:.4f}")
    subprocess.run(["git", "add", str(out_dir.relative_to(REPO_ROOT))], check=True, cwd=REPO_ROOT)
    commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        print(f"\n  Committed: {out_dir.relative_to(REPO_ROOT)}")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"\n  Git commit skipped: {commit.stdout.strip()}")


def main() -> None:
    with open(ENTROPY_RESULTS_PATH) as f:
        stored = json.load(f)

    gaps = load_gaps()
    centers, shannon, starts = recompute_full_series(gaps)
    verify_matches_stored_run(centers, shannon, starts, stored)

    boundaries = load_regime_boundaries(domain_end=centers[-1])
    n_regimes = len(boundaries) - 1
    lengths = np.diff(boundaries)
    print(f"\nLoaded {n_regimes - 1} changepoints -> {n_regimes} regimes from "
          f"{GAP_CHANGEPOINTS_PATH.relative_to(REPO_ROOT)}")
    print(f"Regime lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")

    results = run_regime_pass(centers, shannon, boundaries, MIN_WINDOWS_PER_REGIME)
    qualifying = [r for r in results if r["qualifies"]]
    excluded = [r for r in results if not r["qualifies"]]
    n_decay_wins = sum(1 for r in qualifying if r["decay_wins"])
    fraction = n_decay_wins / len(qualifying) if qualifying else 0.0

    print("\n== Regime pass ==")
    print(f"  Total regimes: {n_regimes}, excluded (< {MIN_WINDOWS_PER_REGIME} windows): {len(excluded)}, "
          f"qualifying: {len(qualifying)}")
    print(f"  Decay-wins (exp beats flat AND k>0): {n_decay_wins} / {len(qualifying)} = {fraction:.4f}")
    for r in qualifying:
        status = "DECAY" if r["decay_wins"] else ("no-converge" if not r["converged"] else "flat/growth wins")
        exp_r2_str = f"{r['exp_r2']:.4f}" if r["exp_r2"] is not None else "n/a"
        k_str = f"{r['k']:.5f}" if r["k"] is not None else "n/a"
        print(f"    regime {r['regime']:2d} [{r['regime_start']:6d},{r['regime_end']:6d}) "
              f"n={r['n_windows']:3d}  exp_r2={exp_r2_str:>8}  flat_r2={r['flat_r2']:.4f}  "
              f"k={k_str:>9}  -> {status}")

    print("\n== Permutation null for the fraction ==")
    null_fractions = permutation_null_fraction(centers, shannon, lengths, MIN_WINDOWS_PER_REGIME, N_PERM, SEED)
    null_p = float((null_fractions >= fraction).mean())
    print(f"  Null (n_perm={N_PERM}): mean={null_fractions.mean():.4f}  std={null_fractions.std():.4f}")
    print(f"  Observed fraction {fraction:.4f} vs. null -> p={null_p:.4f}")

    decay_regimes = [r for r in qualifying if r["decay_wins"]]
    k_values = np.array([r["k"] for r in decay_regimes])
    if len(k_values) > 1:
        k_cv = float(k_values.std() / k_values.mean()) if k_values.mean() != 0 else float("inf")
    else:
        k_cv = None
    print(f"\n== k-value distribution ({len(k_values)} decay-winning regimes) ==")
    if len(k_values) > 0:
        print(f"  k: mean={k_values.mean():.5f}  std={k_values.std():.5f}  "
              f"median={np.median(k_values):.5f}  min={k_values.min():.5f}  max={k_values.max():.5f}")
        if k_cv is not None:
            print(f"  Coefficient of variation (std/mean): {k_cv:.3f} "
                  f"({'clustered' if k_cv < 0.5 else 'scattered'} by a CV<0.5 rule of thumb)")
    else:
        print("  No regimes showed the decay pattern -- no k-distribution to report.")

    # ── Plot ─────────────────────────────────────────────────────────────
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    n_q = len(qualifying)
    n_cols = 6
    n_rows = int(np.ceil(n_q / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False)
    for i, r in enumerate(qualifying):
        ax = axes[i // n_cols][i % n_cols]
        ax.plot(r["n_prime"], r["velocity"], color="#4c72b0", lw=0.9, marker="o", markersize=2)
        ax.axhline(0, color="#cbd5e1", lw=0.6)
        if r["decay_wins"]:
            n_fit = np.linspace(r["n_prime"].min(), r["n_prime"].max(), 100)
            ax.plot(n_fit, exp_model(n_fit, r["v0"], r["k"]), color="#d1495b", lw=1.3,
                     label=f"exp (R2={r['exp_r2']:.2f}, k={r['k']:.4f})")
        else:
            ax.axhline(r["velocity"].mean(), color="#94a3b8", lw=1.2, ls="--",
                        label=f"flat (R2={r['flat_r2']:.2f})")
        ax.set_title(f"regime {r['regime']} (n={r['n_windows']})", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=6)
    for j in range(n_q, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(f"Regime-relative entropy velocity: local dH/dN' per regime "
                 f"({n_decay_wins}/{n_q} show burst-decay) [{ts}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = out_dir / "regime_relative_velocity.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {png_path.relative_to(REPO_ROOT)}")

    # ── Markdown summary ─────────────────────────────────────────────────
    if fraction == 0:
        verdict = (
            "**Not supported.** No qualifying regime shows the exponential-decay velocity pattern "
            "beating a flat baseline with genuine decay (k>0) -- the sawtooth/burst-decay hypothesis "
            "finds no support in this data."
        )
    elif null_p >= 0.05:
        verdict = (
            f"**Not supported (statistically).** {n_decay_wins}/{len(qualifying)} regimes "
            f"({fraction:.1%}) nominally show the pattern, but the {N_PERM}-trial permutation null "
            f"(matched regime-length distribution, shuffled order) puts this at p={null_p:.4f} -- not "
            "distinguishable from what a random segmentation with the same length statistics would "
            "produce by chance. The real boundary positions don't appear to matter for this fraction."
        )
    elif fraction < 0.3:
        verdict = (
            f"**Weakly supported / inconclusive.** {n_decay_wins}/{len(qualifying)} regimes "
            f"({fraction:.1%}) show the pattern, and this clears the permutation-null bar "
            f"(p={null_p:.4f}) -- statistically real, but the fraction itself is modest (well under "
            "half of qualifying regimes), so this reads as a real but partial effect, not a general "
            "property of regimes in this data."
        )
    else:
        verdict = (
            f"**Supported.** {n_decay_wins}/{len(qualifying)} regimes ({fraction:.1%}) show the "
            f"burst-decay velocity pattern, clearing the permutation-null bar (p={null_p:.4f}) -- more "
            "than would be expected from boundary position alone under a length-matched null, and "
            "affecting a majority of qualifying regimes."
        )

    md_lines = [
        f"# Regime-relative entropy velocity: burst-decay (sawtooth) test -- {ts}",
        "",
        "## Regime count",
        "",
        f"- Total regimes (39 gap-space changepoints -> 40 regimes): **{n_regimes}**",
        f"- Excluded (< {MIN_WINDOWS_PER_REGIME} windows): **{len(excluded)}**",
        f"- Qualifying: **{len(qualifying)}**",
        "",
    ]
    if excluded:
        md_lines += ["| regime | n_windows |", "|---|---|"]
        for r in excluded:
            md_lines.append(f"| {r['regime']} | {r['n_windows']} |")
        md_lines.append("")
    md_lines += [
        "## Per-regime fit results",
        "",
        "| regime | range | n windows | exp R^2 | flat R^2 | k | v0 | result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in qualifying:
        status = "DECAY" if r["decay_wins"] else ("no-converge" if not r["converged"] else "flat/growth wins")
        exp_r2_str = f"{r['exp_r2']:.4f}" if r["exp_r2"] is not None else "n/a"
        k_str = f"{r['k']:.5f}" if r["k"] is not None else "n/a"
        v0_str = f"{r['v0']:.5f}" if r["v0"] is not None else "n/a"
        md_lines.append(f"| {r['regime']} | [{r['regime_start']},{r['regime_end']}) | {r['n_windows']} | "
                         f"{exp_r2_str} | {r['flat_r2']:.4f} | {k_str} | {v0_str} | {status} |")
    md_lines += [
        "",
        "## Aggregate: fraction showing burst-decay",
        "",
        f"- Decay-wins: **{n_decay_wins} / {len(qualifying)} = {fraction:.4f}**",
        f"- Permutation null ({N_PERM} trials, shuffled regime-length order, matched spacing): "
        f"mean={null_fractions.mean():.4f}, std={null_fractions.std():.4f}",
        f"- p-value (P[null fraction >= observed]): **{null_p:.4f}**",
        "",
        "## k-value distribution (decay-winning regimes only)",
        "",
    ]
    if len(k_values) > 0:
        md_lines += [
            f"- n = {len(k_values)}",
            f"- mean={k_values.mean():.5f}, std={k_values.std():.5f}, median={np.median(k_values):.5f}, "
            f"min={k_values.min():.5f}, max={k_values.max():.5f}",
        ]
        if k_cv is not None:
            md_lines.append(
                f"- Coefficient of variation: {k_cv:.3f} -- "
                f"{'clustered (CV<0.5), suggesting a consistent underlying rate' if k_cv < 0.5 else 'scattered (CV>=0.5), suggesting overfitting noise rather than a shared rate'}"
            )
    else:
        md_lines.append("No decay-winning regimes -- no distribution to report.")
    md_lines += [
        "",
        "## Verdict: sawtooth / burst-decay hypothesis",
        "",
        verdict,
        "",
    ]
    md_path = out_dir / "regime_relative_velocity_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved summary to {md_path.relative_to(REPO_ROOT)}")

    results_json = {
        "timestamp": ts,
        "entropy_results_source": str(ENTROPY_RESULTS_PATH.relative_to(REPO_ROOT)),
        "gap_changepoints_source": str(GAP_CHANGEPOINTS_PATH.relative_to(REPO_ROOT)),
        "min_windows_per_regime": MIN_WINDOWS_PER_REGIME,
        "n_regimes": n_regimes,
        "n_excluded": len(excluded),
        "n_qualifying": len(qualifying),
        "n_decay_wins": n_decay_wins,
        "fraction": fraction,
        "n_perm": N_PERM,
        "seed": SEED,
        "null_mean": float(null_fractions.mean()),
        "null_std": float(null_fractions.std()),
        "null_p_value": null_p,
        "k_values": k_values.tolist(),
        "k_cv": k_cv,
        "regimes": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            for r in results
        ],
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results_json, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    auto_commit_push(out_dir, fraction, null_p, ts)


if __name__ == "__main__":
    main()
