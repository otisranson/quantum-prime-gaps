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
