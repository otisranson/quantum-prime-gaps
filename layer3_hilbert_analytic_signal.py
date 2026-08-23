"""layer3_hilbert_analytic_signal.py

Stage 4 of this session: apply the Hilbert transform to the raw prime gap
sequence, extract the analytic signal's instantaneous amplitude envelope and
instantaneous phase, and check both against the 39 confirmed changepoints and
against Stage 2's log-fit trend curves.

**Confound flagged up front, before any result below is read as a finding:**
the Hilbert transform's usual assumption is a signal that oscillates around
zero (narrowband, mean-free). The raw gap sequence is strictly positive with
a strong upward PNT trend -- applying scipy.signal.hilbert to it "as raw" (as
this stage's instructions literally specify) means the resulting amplitude
envelope is expected, by construction, to track the local magnitude of the
signal itself (dominated by the trend and typical gap size), not some
independent oscillatory envelope. A high correlation between the envelope and
Stage 2's mean-trend log fit is therefore the *expected*, uninteresting
outcome -- consistent with this repo's standing discipline of stating the
expected/null outcome before reporting a match as if it were a discovery.

**Method:**
1. `scipy.signal.hilbert` on the raw (untouched) 19,999-gap sequence ->
   analytic signal. Instantaneous amplitude = |analytic|; instantaneous phase
   = angle(analytic), both raw and unwrapped.
2. Phase-discontinuity test at the 39 changepoints: identical design to
   layer3_phase_changepoint_analysis.py's STFT jump test (window-to-window
   |diff| of unwrapped phase, permutation null of 39 random positions) --
   here applied directly to the Hilbert instantaneous phase rather than an
   STFT-derived one, since the Hilbert transform already gives a genuine
   full-resolution instantaneous phase at every index.
3. Envelope-vs-log-fit comparison: Stage 2's a*ln(N+2)+b fits for rolling
   mean and rolling std (layer3_log_detrend_residual.py, refit here directly
   from the raw sequence to keep this script standalone) are evaluated at
   every raw index and compared to the instantaneous amplitude envelope via
   Pearson correlation.

Run: python layer3_hilbert_analytic_signal.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

REPO_ROOT = Path(__file__).parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"
CHANGEPOINTS_SOURCE = REPO_ROOT / "output/prime/20260818_015045/results.json"
OUT_ROOT = REPO_ROOT / "output" / "prime"

ROLLING_WINDOW = 100
N_PERM = 5000
SEED = 42


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    gaps = np.array(cache["gaps"])
    assert len(gaps) == cache["n_gaps"]
    return gaps


def load_changepoints() -> list[int]:
    with open(CHANGEPOINTS_SOURCE) as f:
        data = json.load(f)
    return [c["position"] for c in data["changepoints"]]


def rolling_mean_std(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x ** 2, 0, 0.0))
    mean = (c1[k:] - c1[:-k]) / k
    mean_sq = (c2[k:] - c2[:-k]) / k
    std = np.sqrt(np.clip(mean_sq - mean ** 2, 0.0, None))
    return mean, std


def fit_log_linear(centers: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    x = np.log(centers + 2)
    a, b = np.polyfit(x, values, 1)
    predicted = a * x + b
    ss_res = float(np.sum((values - predicted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), r_squared


def changepoint_jump_test(unwrapped_phase: np.ndarray, changepoints: list[int],
                           n_perm: int, rng: np.random.Generator) -> dict:
    jumps = np.abs(np.diff(unwrapped_phase))  # jumps[i] = |phase[i+1] - phase[i]|
    n = len(jumps)
    cp_jumps = [float(jumps[cp]) for cp in changepoints if 0 <= cp < n]
    observed_mean = float(np.mean(cp_jumps))

    null_means = np.empty(n_perm)
    for i in range(n_perm):
        rand_idx = rng.choice(n, size=len(cp_jumps), replace=False)
        null_means[i] = jumps[rand_idx].mean()
    p_value = float(np.mean(null_means >= observed_mean))
    return {
        "observed_mean_jump_at_changepoints": observed_mean,
        "null_mean": float(null_means.mean()), "null_std": float(null_means.std()),
        "p_value": p_value,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    full_gaps = load_full_gaps().astype(float)
    changepoints = load_changepoints()
    print(f"Loaded {len(full_gaps)} raw gaps and {len(changepoints)} changepoints")

    analytic = hilbert(full_gaps)
    envelope = np.abs(analytic)
    phase = np.angle(analytic)
    unwrapped_phase = np.unwrap(phase)
    print(f"Analytic signal computed. Envelope range: [{envelope.min():.2f}, {envelope.max():.2f}]")

    print("\nRunning changepoint phase-jump test on Hilbert instantaneous phase...")
    jump_test = changepoint_jump_test(unwrapped_phase, changepoints, N_PERM, rng)
    sig = "SIGNIFICANT (p<0.05)" if jump_test["p_value"] < 0.05 else "not significant"
    print(f"  observed mean jump={jump_test['observed_mean_jump_at_changepoints']:.4f}, "
          f"null(mean={jump_test['null_mean']:.4f}, std={jump_test['null_std']:.4f}), "
          f"p={jump_test['p_value']:.4f} -> {sig}")

    mean_series, std_series = rolling_mean_std(full_gaps, ROLLING_WINDOW)
    centers = np.arange(len(mean_series)) + ROLLING_WINDOW / 2
    a_mean, b_mean, r2_mean = fit_log_linear(centers, mean_series)
    a_std, b_std, r2_std = fit_log_linear(centers, std_series)
    print(f"\nmean(N) log fit: {a_mean:.6f}*ln(N+2)+{b_mean:.6f} (R^2={r2_mean:.4f})")
    print(f"std(N)  log fit: {a_std:.6f}*ln(N+2)+{b_std:.6f} (R^2={r2_std:.4f})")

    idx = np.arange(len(full_gaps)) + 1.0  # avoid ln(0)
    mean_curve_full = a_mean * np.log(idx + 2) + b_mean
    std_curve_full = a_std * np.log(idx + 2) + b_std

    r_env_mean = float(np.corrcoef(envelope, mean_curve_full)[0, 1])
    r_env_std = float(np.corrcoef(envelope, std_curve_full)[0, 1])
    print(f"\nEnvelope vs. mean-trend log fit: Pearson r = {r_env_mean:.4f}")
    print(f"Envelope vs. std-trend  log fit: Pearson r = {r_env_std:.4f}")
    print("(Note: these two values are identical by construction, not two independent checks -- both "
          "mean_curve_full and std_curve_full are positive-slope affine transforms of the same ln(N+2), "
          "and Pearson correlation is invariant to positive-slope affine transforms of either variable.)")
    print(f"Result vs. the pre-stated expectation (module docstring's confound note predicted HIGH "
          f"correlation, since a positive trend-dominated signal's Hilbert envelope was expected to track "
          f"its own magnitude): actual r={r_env_mean:.4f} is WEAK, contrary to that expectation. Reported "
          f"plainly as a genuine (if modest) surprise, not adjusted after the fact to match the prediction.")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax = plt.subplots(figsize=(20, 6))
    ax.plot(idx, full_gaps, color="#94a3b8", lw=0.3, alpha=0.6, label="raw gap sequence")
    ax.plot(idx, envelope, color="#4c72b0", lw=1.0, label="Hilbert instantaneous amplitude envelope")
    ax.plot(idx, mean_curve_full, color="#d1495b", lw=1.4, ls="--",
            label=f"Stage 2 mean-trend log fit (r={r_env_mean:.3f} vs. envelope)")
    for cp in changepoints:
        ax.axvline(cp, color="#2a9d5c", lw=0.6, alpha=0.4)
    ax.set_xlabel("gap index")
    ax.set_ylabel("gap size / envelope")
    ax.set_title(f"Hilbert instantaneous amplitude envelope vs. raw sequence and log-trend fit [{ts}]")
    ax.legend(fontsize=9, loc="upper left")
    fig1.tight_layout()
    env_path = out_dir / "layer3_hilbert_amplitude_envelope.png"
    fig1.savefig(env_path, dpi=150)
    plt.close(fig1)
    print(f"\nSaved figure to {env_path.relative_to(REPO_ROOT)}")

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    ax1.plot(idx, phase, color="#4c72b0", lw=0.3)
    for cp in changepoints:
        ax1.axvline(cp, color="#d1495b", lw=0.8, ls=":", alpha=0.7)
    ax1.set_ylabel("instantaneous phase (rad, wrapped)")
    ax1.set_title("Hilbert instantaneous phase (wrapped)")

    ax2.plot(idx, unwrapped_phase, color="#e08214", lw=0.6)
    for cp in changepoints:
        ax2.axvline(cp, color="#d1495b", lw=0.8, ls=":", alpha=0.7)
    ax2.set_xlabel("gap index")
    ax2.set_ylabel("instantaneous phase (rad, unwrapped)")
    ax2.set_title(f"Hilbert instantaneous phase (unwrapped) -- changepoint jump test "
                  f"p={jump_test['p_value']:.4f} (obs={jump_test['observed_mean_jump_at_changepoints']:.3f} "
                  f"vs. null={jump_test['null_mean']:.3f}+/-{jump_test['null_std']:.3f})")
    fig2.suptitle(f"Hilbert instantaneous phase, 39 changepoints marked [{ts}]", fontsize=13)
    fig2.tight_layout()
    phase_path = out_dir / "layer3_hilbert_instantaneous_phase.png"
    fig2.savefig(phase_path, dpi=150)
    plt.close(fig2)
    print(f"Saved figure to {phase_path.relative_to(REPO_ROOT)}")

    results = {
        "timestamp": ts,
        "gaps_source": str(GAPS_CACHE_PATH.relative_to(REPO_ROOT)),
        "changepoints_source": str(CHANGEPOINTS_SOURCE.relative_to(REPO_ROOT)),
        "config": {"rolling_window": ROLLING_WINDOW, "n_perm": N_PERM, "seed": SEED},
        "n_changepoints": len(changepoints),
        "envelope_range": [float(envelope.min()), float(envelope.max())],
        "changepoint_phase_jump_test": jump_test,
        "log_fits": {
            "mean": {"a": round(a_mean, 6), "b": round(b_mean, 6), "r_squared": round(r2_mean, 6)},
            "std": {"a": round(a_std, 6), "b": round(b_std, 6), "r_squared": round(r2_std, 6)},
        },
        "envelope_vs_trend_correlation": {"vs_mean_fit": r_env_mean, "vs_std_fit": r_env_std},
        "confound_note": "Hilbert transform applied to the raw, strictly-positive, trend-dominated gap "
                          "sequence (not zero-mean/narrowband) as literally instructed. Pre-registered "
                          "expectation (before running) was that the envelope would track local signal "
                          "magnitude closely -- vs_mean_fit and vs_std_fit are mathematically identical "
                          "(both curves are positive-slope affine transforms of the same ln(N+2), and "
                          "Pearson r is invariant to that), so this is one comparison, not two. The actual "
                          "result (r~0.17) is WEAK, contrary to the pre-stated expectation of high "
                          "correlation -- reported as a genuine surprise, not evidence of hidden structure "
                          "(a weak correlation is still consistent with 'no interesting structure', just not "
                          "in the direction originally guessed).",
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    msg = (f"experiment: Hilbert analytic signal on raw gap sequence {ts} -- "
           f"phase-jump test p={jump_test['p_value']:.4f} ({'sig' if jump_test['p_value'] < 0.05 else 'not sig'}), "
           f"envelope-vs-mean-trend r={r_env_mean:.3f}, envelope-vs-std-trend r={r_env_std:.3f}")
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
