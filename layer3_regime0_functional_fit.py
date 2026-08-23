"""layer3_regime0_functional_fit.py

Stage 5 of this session: fit four candidate functional forms (log, power law,
linear, exponential) to the raw gap sequence within "regime 0", under the two
boundary definitions that exist in this repo for that regime, and compare
them by AIC/BIC.

**The two boundary definitions, both already established elsewhere in this
repo, not invented here:**
- `[0, 499)` -- regime 0 of the 40-regime carving from the 39 gap-space
  changepoints (`output/prime/20260818_015045/results.json`, first
  changepoint at position 499; used throughout
  `layer3_regime_characterization_20k.py`).
- `[0, 1529)` -- regime 0 of the original 3-regime carving from the 3
  MI-space changepoints (1529, 2501, 4211; used throughout the original
  Regime Overlay / Per-Regime Characterization / Kurtosis Robustness checks
  in `hypotheses/regime_internal_wave_structure.md`).

These come from two different changepoint-detection signals (raw-gap rolling
mean vs. quantum-measured MI) applied at two different scales, per CLAUDE.md's
"Two changepoint sets" note -- comparing fits across them tests whether the
choice of boundary changes which functional form looks best, not whether the
two detectors agree with each other.

**Candidate forms** (all 2-parameter, so AIC/BIC differences reduce to a
comparison of residual sum of squares -- k=2 for every candidate):
- log:         y = a*ln(N) + b
- power law:   y = a*N^b
- linear:      y = a*N + b
- exponential: y = a*exp(b*N)

N = 1-based gap index within the regime slice (N=1 for the regime's first
gap), to keep ln(N) and N^b well-defined. Fit via `scipy.optimize.curve_fit`
(nonlinear least squares) for all four so they're compared on equal footing;
`log` and `linear` also have closed-form OLS solutions but curve_fit
converges to the same values for these simpler forms and this keeps the
fitting code uniform across all four.

AIC = 2k + n*ln(RSS/n); BIC = k*ln(n) + n*ln(RSS/n), both computed on raw
(non-standardized) residuals -- lower is better for both.

Run: python layer3_regime0_functional_fit.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"
OUT_ROOT = REPO_ROOT / "output" / "prime"

BOUNDARIES = {"[0, 499)": (0, 499), "[0, 1529)": (0, 1529)}


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    gaps = np.array(cache["gaps"], dtype=float)
    assert len(gaps) == cache["n_gaps"]
    return gaps


def form_log(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.log(n) + b


def form_power(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.power(n, b)


def form_linear(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * n + b


def form_exp(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(b * n)


CANDIDATES = [
    {"name": "log", "func": form_log, "p0": (1.0, 1.0)},
    {"name": "power", "func": form_power, "p0": (1.0, 0.3)},
    {"name": "linear", "func": form_linear, "p0": (0.001, 5.0)},
    {"name": "exponential", "func": form_exp, "p0": (5.0, 0.0001)},
]


def aic_bic(y: np.ndarray, y_pred: np.ndarray, k: int) -> tuple[float, float]:
    n = len(y)
    rss = float(np.sum((y - y_pred) ** 2))
    rss = max(rss, 1e-12)
    log_likelihood_term = n * np.log(rss / n)
    aic = 2 * k + log_likelihood_term
    bic = k * np.log(n) + log_likelihood_term
    return float(aic), float(bic)


def fit_boundary(y: np.ndarray) -> list[dict]:
    n_idx = np.arange(1, len(y) + 1, dtype=float)
    results = []
    for cand in CANDIDATES:
        try:
            popt, _ = curve_fit(cand["func"], n_idx, y, p0=cand["p0"], maxfev=20000)
            y_pred = cand["func"](n_idx, *popt)
            if not np.all(np.isfinite(y_pred)):
                raise RuntimeError("non-finite prediction")
            aic, bic = aic_bic(y, y_pred, k=2)
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            results.append({"name": cand["name"], "params": [float(p) for p in popt],
                             "aic": aic, "bic": bic, "r_squared": r_squared,
                             "y_pred": y_pred, "converged": True})
        except (RuntimeError, ValueError) as exc:
            results.append({"name": cand["name"], "params": None, "aic": float("inf"), "bic": float("inf"),
                             "r_squared": float("nan"), "y_pred": None, "converged": False, "error": str(exc)})
    return results


def main() -> None:
    full_gaps = load_full_gaps()
    print(f"Loaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")

    all_results = {}
    for label, (start, end) in BOUNDARIES.items():
        y = full_gaps[start:end]
        print(f"\n=== Regime 0, boundary {label} (n={len(y)}) ===")
        fits = fit_boundary(y)
        for f in fits:
            if f["converged"]:
                print(f"  {f['name']:12s} AIC={f['aic']:9.2f}  BIC={f['bic']:9.2f}  R^2={f['r_squared']:.4f}  "
                      f"params={[round(p, 5) for p in f['params']]}")
            else:
                print(f"  {f['name']:12s} FAILED TO CONVERGE ({f.get('error', '?')})")
        converged = [f for f in fits if f["converged"]]
        winner_aic = min(converged, key=lambda f: f["aic"])["name"] if converged else None
        winner_bic = min(converged, key=lambda f: f["bic"])["name"] if converged else None
        print(f"  Winner by AIC: {winner_aic}   Winner by BIC: {winner_bic}")
        all_results[label] = {"y": y, "fits": fits, "winner_aic": winner_aic, "winner_bic": winner_bic}

    winners_aic = {label: r["winner_aic"] for label, r in all_results.items()}
    boundary_changes_winner = len(set(winners_aic.values())) > 1
    print(f"\nDoes the boundary definition change the AIC winner? "
          f"{'YES' if boundary_changes_winner else 'NO'} ({winners_aic})")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    colors = {"log": "#4c72b0", "power": "#e08214", "linear": "#2a9d5c", "exponential": "#d1495b"}
    for ax, (label, r) in zip(axes, all_results.items(), strict=True):
        y = r["y"]
        n_idx = np.arange(1, len(y) + 1)
        ax.plot(n_idx, y, color="#94a3b8", lw=0.6, alpha=0.7, label="raw gaps")
        for f in r["fits"]:
            if f["converged"]:
                marker = " *AIC winner*" if f["name"] == r["winner_aic"] else ""
                ax.plot(n_idx, f["y_pred"], color=colors[f["name"]], lw=1.6,
                         label=f"{f['name']}{marker} (AIC={f['aic']:.1f})")
        ax.set_title(f"Regime 0, boundary {label} (n={len(y)})")
        ax.set_xlabel("N (1-based index within regime)")
        ax.set_ylabel("gap size")
        ax.legend(fontsize=8)
    fig.suptitle(f"Regime 0 functional form fits -- two boundary definitions [{ts}]", fontsize=13)
    fig.tight_layout()
    png_path = out_dir / "layer3_regime0_functional_fit.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {png_path.relative_to(REPO_ROOT)}")

    results_json = {
        "timestamp": ts,
        "gaps_source": str(GAPS_CACHE_PATH.relative_to(REPO_ROOT)),
        "boundaries": {
            label: {
                "range": list(BOUNDARIES[label]), "n": len(r["y"]),
                "fits": [
                    {"name": f["name"], "converged": f["converged"],
                     "params": f["params"], "aic": f["aic"], "bic": f["bic"], "r_squared": f["r_squared"]}
                    for f in r["fits"]
                ],
                "winner_aic": r["winner_aic"], "winner_bic": r["winner_bic"],
            }
            for label, r in all_results.items()
        },
        "boundary_changes_aic_winner": bool(boundary_changes_winner),
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results_json, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    msg = (f"analysis: regime 0 functional form fit {ts} -- "
           f"winners (AIC): {winners_aic}, boundary changes winner: {boundary_changes_winner}")
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
