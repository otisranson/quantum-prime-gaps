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
per-window data (MI, mode, gaps) to `output/prime/{timestamp}/results_*.json`. The regime-change /
gap-derivative analysis scripts at the repo root (`regime_fit_5k.py`,
`gap_derivative_zero_crossings.py`, `smoothed_gap_derivative_zero_crossings.py`,
`layer2_magnitude_test.py`, `smoothed_derivative_wave.py`) all read directly from one of those
existing `results_*.json` files rather than re-running the circuit — check which timestamped run a
script points at before assuming it covers the latest data.

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
