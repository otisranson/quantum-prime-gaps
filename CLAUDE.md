# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup:
```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Lint (whole repo, one ruff config in `pyproject.toml`, also runs in CI via `.github/workflows/lint.yml`):
```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/ruff check .
```

There is no test suite (no pytest/unittest, nothing in `requirements-dev.txt` but ruff). Correctness
is checked with inline runtime assertions inside the scripts themselves — e.g.
`quantum_prime_gaps/quantum_prime_gaps.py` re-derives the 50 hardcoded primes with a sieve and
cross-checks its Qiskit circuit against an independent numpy statevector implementation to
floating-point precision before anything is plotted (`verify_extrapolation_roundtrip`,
`verify_quantum_fft_matches_padded_numpy`). New quantitative scripts in this repo generally follow
that pattern: assert the thing that must hold, don't add a separate test file for it.

Running a script: everything is a standalone `python script.py` (some accept flags, e.g.
`--qubits`, `--hardware`, `--backend`, `--predict-steps`, `--top-k` on
`quantum_prime_gaps/quantum_prime_gaps.py` — see README for the full flag list). Hardware runs need
`export QISKIT_IBM_TOKEN="..."` and hit real IBM Quantum devices (queue time, real cost in job
quota) — don't run `*_hw.py` or `--hardware` scripts without the user asking for that explicitly.
Most scripts auto-commit and push their own output on completion; be aware a script run can push to
`origin/main` on its own, separately from any git commands you run.

## Architecture

**Two independent quantum pathways, easy to conflate:**
1. **Recurrent predictor** (`prime_predictor.py` / `prime_predictor_hw.py`) — the "v3" architecture
   described in `PROJECT_STATUS.md`. A 4-qubit circuit (Bell pair → per-qubit `RY` gap encoding,
   locally normalized to each window's max gap → approximated inverse QFT) slides across the 49
   known prime gaps in windows of 4, with each window's measured bitstring feeding forward as an
   angle offset into the next window. 46 sequential windows; the last window's distribution is the
   prediction. This pathway only uses `RY` and QFT-family gates — no `RZ`, no golden ratio or other
   constants baked into the circuit.
2. **Amplitude/spectral pathway** (`quantum_prime_gaps/quantum_prime_gaps.py`) — a *different*
   encoding of the same 49 gaps: data re-uploading for the "amplitude landscape" analysis (lossy,
   non-invertible) versus direct zero-padded statevector loading for prediction (a literal,
   invertible QFT via `quantum_fft`, sign-convention-corrected against `np.fft.fft`). Forward
   prediction beyond the known window is a separate, explicitly classical extrapolation step
   (`quantum_fourier_extrapolate` / `fourier_extrapolate`), not something the QFT itself does.

**Terrain visualization** (`terrain_visualizer.py`, `terrain_1000primes.py`, `terrain_5000primes.py`)
renders per-window probability distributions as topographic maps and writes the underlying
per-window data (MI, mode, gaps) to `output/prime/{timestamp}/results_*.json`.

**`data/`** (added 2026-08-17, built by `build_prime_cache.py`) holds the standalone primes+gaps
cache for classical analysis: `primes_5000.json` (independently sieved, verified byte-identical to
the gap sequence reconstructed from `results_5000primes.json`) and `primes_20000.json`. Every
`layer2_*` / `layer3_*` / `smoothed_*` / `gap_derivative_*` script at the repo root reads gaps from
one of these cache files now, rather than reconstructing them from a quantum-run JSON each time.
`regime_fit_5k.py` is the one exception — it needs quantum-measured MI, not just gaps, so it still
reads `results_5000primes.json` directly. `mi_landscape_25groups.py` also bypasses the cache by
design: it regenerates only its own first 100 gaps inline (with an independent trial-division
check baked in) for a real-quantum-circuit purpose unrelated to this cache.

**Two changepoint sets exist for the gap sequence, at different scales and from different signals
— don't conflate them.** The original 3 (windows 1529, 2501, 4211) come from `regime_fit_5k.py`
running binary-segmentation changepoint detection on quantum-measured MI (5000-prime run, K=100
rolling mean) and are what `hypotheses/regime_internal_wave_structure.md` means by "the known
changepoints" through its Kurtosis Robustness Check and Changepoint Character Comparison sections.
A second set of 39 comes from `layer3_20k_scaleup.py`, which runs the same least-squares
mean-shift cost (vectorized, verified equivalent to the original's O(n²) loop) directly on the
*raw gap sequence's* rolling mean at 20k-prime scale — no 20k-prime quantum run exists — with a
data-driven permutation-null stopping rule instead of a fixed count of 3. Saved to
`output/prime/20260818_015045/results.json` and reused (not re-detected) by
`layer3_regime_characterization_20k.py` and the `layer3_regime_wave_gallery*.py` scripts. Check
which changepoint set (and which timestamped run) a script points at before assuming it covers the
latest data or the same detection method as another script.

**`qubit_hierarchy_core.py`** is a data-agnostic statistical core (per-qubit marginals, pairwise MI
correlation matrix, hierarchical bisection with Miller-Madow bias correction) shared between this
repo's `quantum_prime_gaps/qubit_hierarchy_analysis.py` and the sibling `quantum_radio` project in
`QuantumResearch` — duplicated deliberately (not imported cross-repo) so this repo has no external
dependency. If you change the math here, the sibling copy will drift unless updated too.

**`output/prime/`** holds timestamped run directories (PNG + JSON), each written by whichever script
produced it; not gitignored, tracked normally. **`archive/{YYYY-MM-DD}/`** holds superseded
scripts and one-off historical results, moved there with `git mv` to preserve history rather than
deleted — when in doubt about whether something is still active, check `README.md`'s "Contents"
list and `PROJECT_STATUS.md`'s file map before archiving it.

**`hypotheses/`** is a research-log convention (started 2026-08-16, no older precedent, so infer
the pattern from the existing files rather than assuming it's an established format). **Before
starting any work in this directory — reading a task, continuing an investigation, adding a new
hypothesis or empirical check — read every file in `hypotheses/` first.** Each file builds on
claims, corrections, and open questions from the others; acting on one without reading the rest
risks repeating an already-refuted test or contradicting a correction that's already on record.
First-person
markdown notes, each with `## Observation` / `## Hypothesis` / `## Prediction` / `## Status`
sections (and `## Critical test`, `## Empirical Check — <name>`, `## Correction` as follow-ups
arrive). The working discipline here is real and intentional, not decorative: a prediction is
committed to git *before* the verification run happens, so the commit timestamp proves it wasn't
fitted to the answer after seeing it; empirical checks and corrections are **appended as new
sections**, never rewritten over the original claim, so the full reasoning trail — including wrong
turns — stays visible in one file's history. Commit messages follow a `<kind>: <summary>, <date>`
convention (`hypothesis: ...`, `analysis: ...`, `prediction: ...`, `empirical check: ...`,
`correction: ...`). If you're asked to add to a hypothesis file, follow this pattern: don't delete
or silently rewrite a prior claim that turned out wrong — append a correction section instead. Also
watch for look-ahead bias when running the actual analysis: several rounds in this project's history
were tests that turned out to have no statistical discriminating power (e.g. a fixed-radius
proximity check against a signal so dense it always passes) — a base rate against a null
distribution should be computed and reported *before* a result is called a confirmation.

**`quantum_evolve/`** runs [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)
(an LLM-driven evolutionary search, config in `quantum_evolve/config.yaml`) against one specific
function, `reconstruct` in `initial_program.py` — the frequency-truncation/inverse-DFT step shared
by both the classical and quantum spectral prediction pathways above. It costs real LLM API calls
per generation; don't run it without the user asking.

## Session Log: 2026-08-18 Morning — Reciprocal-Prime & Log-Polar Exploration

A separate, later session on the same calendar date as the "Next Session" handoff below (see that
section's own note on system-clock date labeling in this repo) — this was an exploratory detour
into two new angles on the existing data, not a continuation of that handoff's planned work, which
remains unstarted (see the updated note at the top of "Next Session" below).

**`exploration/reciprocal_prime_curve.py`** — new exploration track looking at 1/p_n (prime
reciprocals) instead of raw gaps, built up across four figures in one file: index vs. 1/p (linear +
log y-axis), cumulative sum Σ1/p (log x-axis), and |Δ(1/p)| rate-of-change; rolling variance of that
rate of change at K=100/K=500, normalized by the PNT-predicted envelope `(1/(n·ln n))²`; a scipy
`curve_fit` of `a/(n·ln n)` to 1/p (converged **a=0.5271**), inverted to prime-position residuals —
the normalized residual flattens to **≈ −0.76** rather than to zero, consistent with `n·ln n` missing
the standard `ln(ln n)` Rosser correction term rather than evidence of hidden structure; and a
gap-prediction figure comparing actual gaps to the derivative `(n+1)ln(n+1) − n·ln(n)`, whose
prediction-error autocorrelation is flat past lag 0 out to lag 100 — no memory, consistent with every
other null/no-structure result already on record in `hypotheses/`.

**`exploration/log_polar_changepoint_remap.py`** — remaps the full 20k-gap sequence into log-polar
coordinates (`radius=log(n)`, `angle=2π·gap/local_max_gap_in_window`, K=201) and overlays the 39
confirmed changepoints (`output/prime/20260818_015045/results.json`). Found the 39 changepoints
**49% tighter** (mean nearest-neighbor distance, z-scored axes) in log-polar space than in flat
(position, gap) space. Follow-up `exploration/log_polar_cluster_test.py` isolated the visually
densest sub-cluster (5 points, angle 30–50°/large radius) and removed it — the tightening persisted
almost unchanged (49.0% → 49.6%), refuting the initial guess that the cluster alone explained it. A
permutation test (1,000 draws of 34 random gap-indices) puts the remaining set at the **15.9th
percentile** of a random null — suggestive but not below the conventional 5% significance threshold.
Full detail and caveats in `hypotheses/regime_internal_wave_structure.md`'s two newest sections
("Log-Polar Changepoint Remap" and "Log-Polar Cluster Exclusion Test").

**New output directories this session** (all under `output/prime/`):
- `20260818_063107/` — first `reciprocal_prime_curve.py` run, linear+log 1/p only (superseded by later runs in the same output file below)
- `20260818_064228/` — `reciprocal_prime_analysis.png`, adds cumulative sum + rate-of-change panels
- `20260818_070228/` — adds `reciprocal_prime_variance.png` (rolling variance + PNT-envelope-normalized residual)
- `20260818_071458/` — adds `reciprocal_prime_residuals.png` (PNT curve fit + position residuals)
- `20260818_072405/` — adds `reciprocal_prime_gap_prediction.png` (gap prediction + error autocorrelation)
- `20260818_074756/` — `log_polar_changepoint_remap.png` (first log-polar remap + changepoint overlay)
- `20260818_080542/` — `log_polar_cluster_exclusion_test.png` (cluster-exclusion + permutation test)

## Next Session: Log-Detrended Residual Analysis

**Status update (2026-08-18 morning session, above): still outstanding, unstarted.** No script (e.g.
a `layer3_log_detrend_residual.py`) implementing the plan below has been written yet — the morning
session was an independent exploratory detour into 1/p and log-polar coordinate reframings, not a
replacement for this objective. This remains the next session's primary objective.

**Context:** Tonight's session (2026-08-18) confirmed via the 40-regime characterization that mean
gap and variance both show real, significant log-scale trends across the full 20k sequence
(r=0.85, r=0.88, both p<0.0001), consistent with PNT's ln(N) growth. Every other tested property —
skew, kurtosis, cross-regime self-similarity, boundary-kurtosis ordering — came back statistically
null (n=39-40, permutation-tested). The full-sequence overview plot (`layer3_full_sequence_overview.py`
output) visually confirms the rolling std climbs on a log-like curve and never fully flattens,
oscillating in large humps out to prime 20,000 with no sign of settling into a steady state.

**Open hypothesis for next session:** All prior wave/shape analysis (regime characterization,
cross-regime overlay, kurtosis scans) was run on raw or per-regime-normalized data, which means any
real wave/periodic structure could be getting masked or contaminated by the confirmed log growth
trend itself — this is the same issue previously identified as the regime-0 FFT trend-artifact
problem, just now suspected to matter at full-sequence scale too.

**Next step:** Fit a log function (`std(N) ≈ a·ln(N) + b`, and separately `mean(N) ≈ a·ln(N) + b`)
to the full 20,000-point rolling mean and rolling std, subtract to produce a detrended residual
series, then rerun the existing spectral/kurtosis/self-similarity tools (FFT, autocorrelation,
kurtosis scan) on the residual instead of raw or per-regime data. Rationale: if real periodic or
self-similar wave structure exists, it should be more visible once log growth is no longer
dominating the signal. If the residual also comes back flat, that's a cleaner, more decisive null
than anything found tonight, since it removes the PNT confound entirely rather than working around
it per-regime.

**Do not start this work now** — record only, as the next session's starting point.
