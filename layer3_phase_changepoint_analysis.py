"""layer3_phase_changepoint_analysis.py

Stage 3 of this session: does the phase of the gap sequence's dominant
frequency components show anything unusual at the 39 confirmed gap-space
changepoints (output/prime/20260818_015045/results.json, mean peak kurtosis
6.94, positions 499 through 19395)?

**Two distinct things are reported, since a single global FFT phase can't
answer "phase discontinuity" on its own:**

1. **Literal global-FFT phase.** One FFT over the full 19,999-gap sequence
   (mean-subtracted first, so the DC bin doesn't dominate) gives one complex
   coefficient per frequency bin, hence one phase per frequency for the
   *entire* sequence -- not a per-position quantity. For the top 3 dominant
   frequency components (by magnitude, excluding DC), this script evaluates
   each component's own instantaneous phase function
   `phase(n) = (2*pi*f*n + phi_0) mod 2*pi` at every changepoint position and
   reports it. **This is reported for completeness and is not itself a
   discontinuity test** -- a single global sinusoid's phase advances
   perfectly linearly by construction; anything that looks like a "jump" here
   is the mod-2*pi wrap artifact, not a real discontinuity, and is called out
   as such rather than left to look like a finding.

2. **STFT-based local phase-jump test (the actual discontinuity test).** A
   sliding-window FFT (width=500, step=50) tracks, at each window position,
   the phase of the frequency bin nearest each of the same top-3 global
   frequencies. Each local phase series is unwrapped (np.unwrap, removing
   trivial 2*pi-wrap jumps) and its window-to-window |phase derivative| is
   the "local jump size" at that position. Whether the 39 changepoints sit at
   unusually large local jumps is tested against a permutation null of 39
   random positions (same design as layer2_magnitude_test.py, which is this
   repo's own template for "does a real quantity spike at these positions
   more than chance" -- picked deliberately over a fixed-radius proximity
   check, since two proximity checks in
   hypotheses/second_order_gap_structure.md were later found to have no
   discriminating power at all).

Run: python layer3_phase_changepoint_analysis.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"
CHANGEPOINTS_SOURCE = REPO_ROOT / "output/prime/20260818_015045/results.json"
OUT_ROOT = REPO_ROOT / "output" / "prime"

N_DOMINANT = 3
STFT_WIDTH = 500
STFT_STEP = 50
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


def dominant_components(x: np.ndarray, n_dominant: int, detrend: bool = False) -> list[dict]:
    """detrend=False: mean-subtract only (the literal 'FFT of the full gap
    sequence' request). detrend=True: also remove the series' own best-fit
    line first -- same prospective fix layer3_regime_characterization_20k.py
    applied to avoid the known regime-0 FFT artifact (an undetrended series
    with a strong trend dumps power into the lowest-frequency bins, which
    isn't real periodicity). Both are run and reported; see module
    docstring."""
    n = len(x)
    if detrend:
        lin_idx = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(lin_idx, x, 1)
        centered = x - (slope * lin_idx + intercept)
    else:
        centered = x - x.mean()
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n, d=1.0)
    mag = np.abs(spectrum)
    mag[0] = 0.0  # exclude DC
    top_idx = np.argsort(mag)[::-1][:n_dominant]
    components = []
    for idx in top_idx:
        components.append({
            "bin_index": int(idx), "frequency": float(freqs[idx]),
            "period": float(1.0 / freqs[idx]) if freqs[idx] > 0 else float("inf"),
            "magnitude": float(mag[idx]), "phase": float(np.angle(spectrum[idx])),
        })
    return components


def global_phase_at_positions(component: dict, positions: list[int]) -> list[float]:
    f, phi0 = component["frequency"], component["phase"]
    return [float(((2 * np.pi * f * p + phi0) + np.pi) % (2 * np.pi) - np.pi) for p in positions]


def is_resolvable(period: float, width: int) -> bool:
    """A window of `width` samples needs at least ~2 full cycles of the
    target period to give a meaningful local phase estimate; below that, the
    nearest-bin lookup degenerates toward the window's own DC/near-DC bin and
    the "phase" it returns is not a real instantaneous phase (see module
    docstring caveat, and the erratic-square-wave pattern this produces --
    caught by inspecting the first render of this plot before trusting it)."""
    return period <= width / 2.0


def stft_local_phase(x: np.ndarray, target_freq: float, width: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window FFT; at each window, take the phase of the bin nearest
    target_freq. Returns (window centers, unwrapped local phase series).
    Caller must check is_resolvable() before trusting the result -- see
    that function's docstring."""
    starts = np.arange(0, len(x) - width + 1, step)
    freqs = np.fft.rfftfreq(width, d=1.0)
    nearest_bin = int(np.argmin(np.abs(freqs - target_freq)))
    phases = np.empty(len(starts))
    for i, s in enumerate(starts):
        window = x[s:s + width] - x[s:s + width].mean()
        spectrum = np.fft.rfft(window)
        phases[i] = np.angle(spectrum[nearest_bin])
    unwrapped = np.unwrap(phases)
    centers = starts + width / 2
    return centers, unwrapped


def local_jump_series(unwrapped_phase: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(unwrapped_phase))


def changepoint_jump_test(centers: np.ndarray, jumps: np.ndarray, changepoints: list[int],
                           n_perm: int, rng: np.random.Generator) -> dict:
    """For each changepoint, the jump value at the nearest window-to-window
    step is looked up; the mean of those 39 values is compared to a
    permutation null of the mean jump at 39 random positions (same design as
    layer2_magnitude_test.py's percentile-rank approach)."""
    step_positions = (centers[:-1] + centers[1:]) / 2
    cp_jumps = []
    for cp in changepoints:
        nearest_i = int(np.argmin(np.abs(step_positions - cp)))
        cp_jumps.append(jumps[nearest_i])
    observed_mean = float(np.mean(cp_jumps))

    n_steps = len(jumps)
    null_means = np.empty(n_perm)
    for i in range(n_perm):
        rand_idx = rng.choice(n_steps, size=len(changepoints), replace=False)
        null_means[i] = jumps[rand_idx].mean()
    p_value = float(np.mean(null_means >= observed_mean))
    return {
        "observed_mean_jump_at_changepoints": observed_mean,
        "null_mean": float(null_means.mean()), "null_std": float(null_means.std()),
        "p_value": p_value, "per_changepoint_jump": [float(j) for j in cp_jumps],
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    full_gaps = load_full_gaps()
    changepoints = load_changepoints()
    print(f"Loaded {len(full_gaps)} raw gaps and {len(changepoints)} changepoints")

    variants = {}
    for variant_name, detrend in [("raw", False), ("detrended", True)]:
        components = dominant_components(full_gaps, N_DOMINANT, detrend=detrend)
        print(f"\nTop dominant frequency components ({variant_name}, DC excluded):")
        for c in components:
            print(f"  bin={c['bin_index']}, freq={c['frequency']:.6f}, period={c['period']:.2f}, "
                  f"mag={c['magnitude']:.2f}, global phase={c['phase']:.4f} rad")
            c["global_phase_at_changepoints"] = global_phase_at_positions(c, changepoints)

        print(f"Running STFT local phase-jump test at each {variant_name} dominant frequency...")
        stft_results = []
        for c in components:
            resolvable = is_resolvable(c["period"], STFT_WIDTH)
            centers, unwrapped = stft_local_phase(full_gaps, c["frequency"], STFT_WIDTH, STFT_STEP)
            jumps = local_jump_series(unwrapped)
            test = changepoint_jump_test(centers, jumps, changepoints, N_PERM, rng)
            if not resolvable:
                print(f"  freq={c['frequency']:.6f} (period={c['period']:.2f}): NOT RESOLVABLE by "
                      f"width={STFT_WIDTH} STFT (period > width/2) -- a window this short can't contain "
                      f"a full cycle, so the 'phase' extracted is not a real instantaneous phase; "
                      f"p={test['p_value']:.4f} is reported but NOT interpretable as evidence either way.")
            else:
                sig = "SIGNIFICANT (p<0.05)" if test["p_value"] < 0.05 else "not significant"
                print(f"  freq={c['frequency']:.6f} (period={c['period']:.2f}): observed mean jump="
                      f"{test['observed_mean_jump_at_changepoints']:.4f}, null(mean={test['null_mean']:.4f}, "
                      f"std={test['null_std']:.4f}), p={test['p_value']:.4f} -> {sig}")
            stft_results.append({"component": c, "centers": centers, "unwrapped": unwrapped,
                                  "jumps": jumps, "test": test, "resolvable": resolvable})
        variants[variant_name] = {"components": components, "stft_results": stft_results}

    print("\nNote: the 'raw' variant's dominant frequencies are period~N, N/2, N/3 (the lowest 3 bins) -- "
          "the known FFT trend-leakage artifact from the sequence's strong PNT growth trend, same failure "
          "mode already diagnosed as the 'regime 0 FFT artifact' elsewhere in this repo, not real "
          "periodicity. The 'detrended' variant (own best-fit line removed first) is the scientifically "
          "meaningful one; 'raw' is reported only because the session prompt literally asked for "
          "'the FFT of the full gap sequence'.")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(N_DOMINANT, 2, figsize=(20, 4 * N_DOMINANT), sharex=True)
    for col, variant_name in enumerate(["raw", "detrended"]):
        stft_results = variants[variant_name]["stft_results"]
        for row, res in enumerate(stft_results):
            ax = axes[row, col] if N_DOMINANT > 1 else axes[col]
            c, centers, unwrapped, test = res["component"], res["centers"], res["unwrapped"], res["test"]
            ax.plot(centers, unwrapped, color="#4c72b0", lw=1.0)
            for cp in changepoints:
                ax.axvline(cp, color="#d1495b", lw=0.8, ls=":", alpha=0.7)
            ax.set_ylabel("unwrapped local phase (rad)")
            if res["resolvable"]:
                title = (f"[{variant_name}] period={c['period']:.1f} -- jump test p={test['p_value']:.4f} "
                         f"(obs={test['observed_mean_jump_at_changepoints']:.3f} vs. "
                         f"null={test['null_mean']:.3f}+/-{test['null_std']:.3f})")
            else:
                title = (f"[{variant_name}] period={c['period']:.1f} -- NOT RESOLVABLE by "
                         f"width={STFT_WIDTH} STFT (period > width/2); p-value not interpretable")
            ax.set_title(title, fontsize=9, color="#111111" if res["resolvable"] else "#b91c1c")
        (axes[-1, col] if N_DOMINANT > 1 else axes[col]).set_xlabel("gap index")
    fig.suptitle(f"Phase analysis at 39 changepoints -- top {N_DOMINANT} dominant frequencies, "
                 f"raw vs. detrended [{ts}]", fontsize=13)
    fig.tight_layout()
    png_path = out_dir / "layer3_phase_changepoint_analysis.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {png_path.relative_to(REPO_ROOT)}")

    def variant_json(variant_name: str) -> dict:
        components = variants[variant_name]["components"]
        stft_results = variants[variant_name]["stft_results"]
        return {
            "dominant_components": [
                {"bin_index": c["bin_index"], "frequency": round(c["frequency"], 8), "period": round(c["period"], 3),
                 "magnitude": round(c["magnitude"], 3), "global_phase": round(c["phase"], 6),
                 "global_phase_at_changepoints": [round(p, 6) for p in c["global_phase_at_changepoints"]]}
                for c in components
            ],
            "stft_phase_jump_tests": [
                {"period": round(res["component"]["period"], 3), "frequency": round(res["component"]["frequency"], 8),
                 "resolvable_by_stft_window": res["resolvable"], **res["test"]}
                for res in stft_results
            ],
        }

    results = {
        "timestamp": ts,
        "gaps_source": str(GAPS_CACHE_PATH.relative_to(REPO_ROOT)),
        "changepoints_source": str(CHANGEPOINTS_SOURCE.relative_to(REPO_ROOT)),
        "config": {"n_dominant": N_DOMINANT, "stft_width": STFT_WIDTH, "stft_step": STFT_STEP,
                   "n_perm": N_PERM, "seed": SEED},
        "n_changepoints": len(changepoints),
        "raw": variant_json("raw"),
        "detrended": variant_json("detrended"),
        "caveat": "The 'global_phase_at_changepoints' values are evaluated from a single whole-sequence FFT "
                  "phase and frequency -- they advance perfectly linearly (mod 2*pi) by construction and are "
                  "NOT a discontinuity test on their own; any apparent jump in that series is a wrap artifact. "
                  "The stft_phase_jump_tests (sliding-window FFT + permutation null vs. random positions) are "
                  "the actual discontinuity test. The 'raw' variant's dominant frequencies are the lowest 3 "
                  "FFT bins (period~N, N/2, N/3) -- known trend-leakage from the sequence's PNT growth trend, "
                  "not real periodicity (same failure mode as the repo's already-diagnosed 'regime 0 FFT "
                  "artifact'). 'detrended' (own best-fit line removed first) is the scientifically meaningful "
                  "variant; 'raw' is included only because the session prompt literally asked for 'the FFT of "
                  "the full gap sequence'.",
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {json_path.relative_to(REPO_ROOT)}")

    n_resolvable_raw = sum(1 for res in variants["raw"]["stft_results"] if res["resolvable"])
    n_resolvable_detrended = sum(1 for res in variants["detrended"]["stft_results"] if res["resolvable"])
    n_sig_raw = sum(1 for res in variants["raw"]["stft_results"] if res["resolvable"] and res["test"]["p_value"] < 0.05)
    n_sig_detrended = sum(1 for res in variants["detrended"]["stft_results"]
                           if res["resolvable"] and res["test"]["p_value"] < 0.05)
    msg = (f"experiment: phase analysis at 39 changepoints {ts} -- "
           f"of resolvable components (raw {n_resolvable_raw}/{N_DOMINANT}, detrended {n_resolvable_detrended}/{N_DOMINANT}), "
           f"significant: raw {n_sig_raw}, detrended {n_sig_detrended}")
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
