"""layer3_pi_coefficient_test.py

Priority test for this session: does the log-linear growth coefficient of the
full 20k gap sequence's rolling std (and, secondarily, its rolling variance)
line up with a simple constant involving pi, e, or ln(2), rather than being an
arbitrary fitted number?

**Naming clash flagged up front:** the session prompt that requested this test
labeled the rolling-std log fit "H(N)". That notation collides with the
Shannon entropy H(N) already defined and fitted in
experiments/gap_entropy_windows.py (a completely different quantity: bits of
entropy over a window's gap-value distribution, not the standard deviation of
gap size). This script fits **std(N)**, not entropy, and uses sigma(N) /
var(N) notation throughout to avoid the collision. No entropy computation
happens here.

**Method:** rolling std and rolling variance (K=100, leading-window
convention, identical to layer3_full_sequence_overview.py's rolling_std) over
the full 20,000-prime gap sequence. Each series is fit to a·ln(center+2)+b via
ordinary least squares (same ln(x+2) convention as gap_entropy_windows.py and
layer3_kurtosis_robustness.py's log_fit_detrend), giving one free-fit slope
for std(N) and one for var(N). Five candidate constants are compared against
whichever fit they were nominated for (four against the std slope, pi^2/6
against the variance slope per the request's own labeling), using relative
percent error |candidate - empirical| / |empirical| * 100. A candidate within
1% of the empirical slope is flagged.

**This is a coefficient-matching exercise, not a mechanism test.** Even a
flagged "match" only says the fitted slope happens to sit near a fixed
constant to within noise -- it is not evidence the constant is causally
responsible for the growth rate (the actual mechanism for gap-size growth is
already established: PNT-driven ~ln(N) growth, confirmed independently via
mean/variance permutation-correlation in
hypotheses/regime_internal_wave_structure.md's 40-Regime Characterization).
Reported plainly either way, following this repo's standing discipline of
reporting null/non-matches with the same rigor as matches.

Run: python layer3_pi_coefficient_test.py
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"
OUT_ROOT = REPO_ROOT / "output" / "prime"

ROLLING_WINDOW = 100
FLAG_THRESHOLD_PCT = 1.0

CANDIDATES = [
    {"name": "1/(2*pi)", "value": 1.0 / (2.0 * math.pi), "target": "std"},
    {"name": "1/pi", "value": 1.0 / math.pi, "target": "std"},
    {"name": "pi^2/6", "value": (math.pi ** 2) / 6.0, "target": "variance"},
    {"name": "1/(2*e)", "value": 1.0 / (2.0 * math.e), "target": "std"},
    {"name": "ln(2)/pi", "value": math.log(2.0) / math.pi, "target": "std"},
]


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    gaps = np.array(cache["gaps"])
    assert len(gaps) == cache["n_gaps"]
    return gaps


def rolling_std(x: np.ndarray, k: int) -> np.ndarray:
    """Leading-window rolling std: value at position i is std(x[i:i+k]) --
    identical convention to layer3_full_sequence_overview.py's rolling_std."""
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x ** 2, 0, 0.0))
    mean = (c1[k:] - c1[:-k]) / k
    mean_sq = (c2[k:] - c2[:-k]) / k
    return np.sqrt(np.clip(mean_sq - mean ** 2, 0.0, None))


def fit_log_linear(centers: np.ndarray, values: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """values ~ a*ln(center+2) + b, same ln(x+2) convention used throughout
    this repo (gap_entropy_windows.py, layer3_kurtosis_robustness.py)."""
    x = np.log(centers + 2)
    a, b = np.polyfit(x, values, 1)
    predicted = a * x + b
    residuals = values - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), predicted, r_squared


def main() -> None:
    full_gaps = load_full_gaps()
    print(f"Loaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")

    sigma = rolling_std(full_gaps, ROLLING_WINDOW)
    variance = sigma ** 2
    centers = np.arange(len(sigma)) + ROLLING_WINDOW / 2

    a_std, b_std, pred_std, r2_std = fit_log_linear(centers, sigma)
    a_var, b_var, pred_var, r2_var = fit_log_linear(centers, variance)

    print(f"\nFree fit -- std(N)      ~ {a_std:.6f} * ln(N+2) + {b_std:.6f}  (R^2={r2_std:.4f})")
    print(f"Free fit -- variance(N) ~ {a_var:.6f} * ln(N+2) + {b_var:.6f}  (R^2={r2_var:.4f})")

    report = []
    for cand in CANDIDATES:
        empirical = a_std if cand["target"] == "std" else a_var
        pct_error = abs(cand["value"] - empirical) / abs(empirical) * 100.0
        flagged = pct_error < FLAG_THRESHOLD_PCT
        report.append({
            "name": cand["name"], "value": cand["value"], "target": cand["target"],
            "empirical_slope": empirical, "pct_error": pct_error, "flagged": flagged,
        })
        flag_str = "  <== FLAGGED (within 1%)" if flagged else ""
        print(f"  {cand['name']:10s} = {cand['value']:.5f}  vs. {cand['target']:8s} slope "
              f"{empirical:.5f}  ->  {pct_error:6.2f}% error{flag_str}")

    n_flagged = sum(r["flagged"] for r in report)
    print(f"\n{n_flagged} of {len(CANDIDATES)} candidates flagged within {FLAG_THRESHOLD_PCT}%.")

    # ── Plot ─────────────────────────────────────────────────────────────
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax1.plot(centers, sigma, color="#4c72b0", lw=0.6, alpha=0.6, label="rolling std, K=100")
    ax1.plot(centers, pred_std, color="#111111", lw=1.8, label=f"free fit (a={a_std:.5f}, R^2={r2_std:.3f})")
    colors = plt.cm.tab10.colors
    ci = 0
    for cand in CANDIDATES:
        if cand["target"] != "std":
            continue
        cand_line = cand["value"] * np.log(centers + 2) + b_std
        ax1.plot(centers, cand_line, lw=1.2, ls="--", color=colors[ci % 10],
                  label=f"{cand['name']}={cand['value']:.5f} ({[r for r in report if r['name'] == cand['name']][0]['pct_error']:.1f}% err)")
        ci += 1
    ax1.set_ylabel("rolling std")
    ax1.set_title(f"std(N) log-linear fit vs. pi/e candidate slopes (same intercept b={b_std:.3f}) [{ts}]")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.plot(centers, variance, color="#e08214", lw=0.6, alpha=0.6, label="rolling variance, K=100")
    ax2.plot(centers, pred_var, color="#111111", lw=1.8, label=f"free fit (a={a_var:.5f}, R^2={r2_var:.3f})")
    var_cand = [c for c in CANDIDATES if c["target"] == "variance"][0]
    var_cand_line = var_cand["value"] * np.log(centers + 2) + b_var
    var_err = [r for r in report if r["name"] == var_cand["name"]][0]["pct_error"]
    ax2.plot(centers, var_cand_line, lw=1.2, ls="--", color="#d1495b",
              label=f"{var_cand['name']}={var_cand['value']:.5f} ({var_err:.1f}% err)")
    ax2.set_xlabel("gap index (window center)")
    ax2.set_ylabel("rolling variance")
    ax2.set_title("variance(N) log-linear fit vs. pi^2/6")
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    png_path = out_dir / "layer3_pi_coefficient_test.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {png_path.relative_to(REPO_ROOT)}")

    results = {
        "timestamp": ts,
        "gaps_source": str(GAPS_CACHE_PATH.relative_to(REPO_ROOT)),
        "rolling_window": ROLLING_WINDOW,
        "flag_threshold_pct": FLAG_THRESHOLD_PCT,
        "free_fit": {
            "std": {"a": round(a_std, 6), "b": round(b_std, 6), "r_squared": round(r2_std, 6)},
            "variance": {"a": round(a_var, 6), "b": round(b_var, 6), "r_squared": round(r2_var, 6)},
        },
        "candidates": [
            {"name": r["name"], "value": round(r["value"], 6), "target": r["target"],
             "empirical_slope": round(r["empirical_slope"], 6), "pct_error": round(r["pct_error"], 4),
             "flagged": bool(r["flagged"])}
            for r in report
        ],
        "n_flagged": int(n_flagged),
        "note": "std(N) fit here is unrelated to the H(N) Shannon-entropy fit in "
                "experiments/gap_entropy_windows.py -- naming collision only, different quantities.",
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    flagged_names = [r["name"] for r in report if r["flagged"]]
    msg = (f"analysis: pi/e coefficient test on 20k rolling std/variance {ts} -- "
           f"std slope a={a_std:.5f} (R^2={r2_std:.3f}), var slope a={a_var:.5f} (R^2={r2_var:.3f}), "
           f"flagged candidates: {flagged_names or 'none'}")
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
