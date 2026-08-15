# Quantum prime gap prediction -- dynamical decoupling hardware run report

Generated automatically by `quantum_prime_gaps.py --hardware --dynamical-decoupling`
when the DD run completes. Overwritten on every DD run -- prior results live in git
history. "First hardware run" below is job `d9tso90u5hac73agdrk0`
(documented in `7QUBIT_HW_RESULTS.md`), re-fetched fresh from IBM rather than re-run,
so dynamical decoupling is the only variable that changed between the two.

- **Timestamp:** 2026-08-11T21:58:05
- **Backend:** ibm_kingston
- **DD job ID:** d9tt68l35hes73fj54ng
- **Shots used:** 4096
- **Dynamical decoupling sequence:** XpXm
- **Transpiled circuit depth:** 908 (exceeds the 50-gate warning threshold) -- NOTE: dynamical decoupling is a
  server-side scheduling pass applied to idle windows in the already-transpiled circuit, so this number is
  expected to match the first run's exactly; that is not evidence DD had no effect, it's the wrong metric
  to look at for DD's effect (see the amplitude/MAE/candidate-zone comparisons below instead).
- **Readout error mitigation:** Yes (measurement twirling, same as the first run)

## Three-way amplitude landscape comparison

| Comparison | MAE |
|---|---:|
| Simulated vs. first hardware run (no DD) | 0.00989 |
| Simulated vs. DD hardware run | 0.00880 |
| First hardware run vs. DD hardware run | 0.00346 |

## Candidate zones: classical vs. first-run hybrid vs. DD-run hybrid

Same magnitude-from-hardware/phase-from-simulator hybrid methodology as
`7QUBIT_HW_RESULTS.md` (NOT a full hardware reconstruction -- phase is unmeasurable
from a single Sampler setting; see that report for the full explanation), computed
identically for both hardware runs so the comparison isolates DD.

### Candidate zones -- classical (numpy FFT, noiseless, for reference)

| Frequencies kept | Power fraction | Nearest primes | Mean distance |
|---:|---:|---|---:|
| 1 | 71.6% | 233, 239, 241, 251, 251, 257, 263, 269, 271, 277 | 1.46 |
| 2 | 74.3% | 233, 239, 241, 251, 251, 257, 263, 263, 271, 277 | 1.26 |
| 3 | 76.9% | 233, 239, 239, 241, 251, 257, 263, 263, 271, 271 | 1.70 |
| 4 | 79.3% | 229, 233, 239, 241, 251, 257, 263, 263, 271, 271 | 1.68 |
| 5 | 81.7% | 229, 233, 239, 241, 251, 257, 257, 263, 269, 271 | 1.35 |

### Candidate zones -- first hardware run hybrid (no DD)

| Frequencies kept | Power fraction | Nearest primes | Mean distance |
|---:|---:|---|---:|
| 1 | 5.3% | 227, 227, 227, 227, 229, 229, 229, 229, 227, 227 | 0.44 |
| 2 | 10.3% | 227, 227, 223, 227, 227, 229, 229, 229, 227, 227 | 0.82 |
| 3 | 14.1% | 227, 223, 223, 223, 223, 227, 229, 229, 227, 223 | 0.66 |
| 4 | 17.3% | 227, 223, 223, 223, 227, 229, 229, 229, 229, 227 | 0.83 |
| 5 | 20.2% | 227, 223, 223, 227, 227, 229, 229, 229, 229, 227 | 0.96 |

### Candidate zones -- DD hardware run hybrid

| Frequencies kept | Power fraction | Nearest primes | Mean distance |
|---:|---:|---|---:|
| 1 | 6.0% | 229, 229, 229, 229, 233, 233, 233, 233, 233, 233 | 1.22 |
| 2 | 9.7% | 229, 229, 229, 229, 229, 227, 227, 227, 227, 227 | 0.56 |
| 3 | 13.4% | 229, 229, 229, 233, 233, 233, 233, 233, 233, 233 | 1.29 |
| 4 | 16.1% | 229, 233, 233, 233, 233, 233, 233, 233, 229, 229 | 1.24 |
| 5 | 18.5% | 229, 233, 233, 233, 233, 233, 233, 233, 233, 229 | 1.13 |

## Verdict

MAE vs. simulated: first run 0.00989, DD run 0.00880 (improved -- lower means closer to the simulated distribution). Hybrid candidate-zone mean nearest-prime distance (max frequencies kept, magnitude-from-hardware/phase-from-simulator, same caveats as `7QUBIT_HW_RESULTS.md`): first run 0.96, DD run 1.13 (did not improve -- classical baseline is 1.35 for reference). The two signals disagree (one improved, one didn't) -- treat this as inconclusive rather than a clean win or loss; the underlying amplitude landscape comparison plot is the more direct evidence than either single summary number.

## Outputs from this run

| PNG file | Pathway that produced it |
|---|---|
| `hardware_dd_amplitude_landscape.png` | prediction circuit, DD hardware run only |
| `hardware_dd_vs_first_run_comparison.png` | prediction circuit, simulated + first hardware run + DD hardware run |
| `hardware_dd_frequency_portrait.png` | prediction circuit, simulated + first hardware run + DD hardware run overlay |

## Console warnings / notable events during this run

- --top-k not given: using top-5 for the paired backward-verification comparison below so classical and quantum are compared at the same truncation level (classical's own default elsewhere in this run is still "keep everything")
- Quantum vs classical MAE differ by 0.0093 at matched top_k=5. This is NOT a sign-convention or rescaling bug -- that's separately proven to floating-point precision (~1e-13) by the "quantum_fft matches np.fft.fft on the zero-padded array" hard check above, which compares the two pathways' spectra bin-for-bin on the identical (padded) array. This MAE gap instead reflects that amplitude encoding's mandatory zero-padding (39 known gaps into a 128-slot register) dilutes spectral power across many more bins, so "top-5" selects a genuinely coarser, different set of frequencies than the classical path's native unpadded spectrum -- an inherent cost of quantum amplitude encoding onto a power-of-2 register, not a computational error.
- quantum path: --top-k not given, defaulting to top-5 frequency components (a full-spectrum reconstruction of the zero-padded register would just reproduce the padding as "predicted" gaps)
- --shots not given: queue depth 0 <= 5 threshold, using 4096 shots
- WARNING: transpiled depth 590 exceeds the 50-gate threshold -- deep circuits accumulate noise fast, expect the hardware distribution to diverge more from the simulated one.
- Readout error mitigation: measurement twirling enabled (no added circuit depth).
- --hardware set: the prediction circuit is also submitted to hardware, but its measured output is Born-rule probabilities only (no phase) -- the prediction MAE/candidates reported above still come from the noiseless simulator regardless; recovering phase from hardware would need full state tomography, out of scope.
- --shots not given: queue depth 0 <= 5 threshold, using 4096 shots
- WARNING: transpiled depth 913 exceeds the 50-gate threshold -- deep circuits accumulate noise fast, expect the hardware distribution to diverge more from the simulated one.
- Readout error mitigation: measurement twirling enabled (no added circuit depth).
- Warning: /home/oranson/Projects/QuantumResearch/.venv/lib/python3.14/site-packages/stevedore/extension.py:338: DeprecationWarning: Since backends now support running jobs that contain both fractional gates and dynamic circuit, IBMFractionalTranslationPlugin is deprecated as of qiskit-ibm-runtime 0.42.0 and will be removed no sooner than 3 months after the release date. Use IBMDynamicFractionalTranslationPlugin instead.
  obj = plugin(*invoke_args, **invoke_kwds)
- Warning: /home/oranson/Projects/QuantumResearch/.venv/lib/python3.14/site-packages/stevedore/extension.py:338: DeprecationWarning: Since backends now support running jobs that contain both fractional gates and dynamic circuit, IBMFractionalTranslationPlugin is deprecated as of qiskit-ibm-runtime 0.42.0 and will be removed no sooner than 3 months after the release date. Use IBMDynamicFractionalTranslationPlugin instead.
  obj = plugin(*invoke_args, **invoke_kwds)
- Warning: /home/oranson/Projects/QuantumResearch/.venv/lib/python3.14/site-packages/stevedore/extension.py:338: DeprecationWarning: Since backends now support running jobs that contain both fractional gates and dynamic circuit, IBMFractionalTranslationPlugin is deprecated as of qiskit-ibm-runtime 0.42.0 and will be removed no sooner than 3 months after the release date. Use IBMDynamicFractionalTranslationPlugin instead.
  obj = plugin(*invoke_args, **invoke_kwds)
- Warning: /home/oranson/Projects/QuantumResearch/.venv/lib/python3.14/site-packages/stevedore/extension.py:338: DeprecationWarning: Since backends now support running jobs that contain both fractional gates and dynamic circuit, IBMFractionalTranslationPlugin is deprecated as of qiskit-ibm-runtime 0.42.0 and will be removed no sooner than 3 months after the release date. Use IBMDynamicFractionalTranslationPlugin instead.
  obj = plugin(*invoke_args, **invoke_kwds)
- --shots not given: queue depth 0 <= 5 threshold, using 4096 shots
- WARNING: transpiled depth 908 exceeds the 50-gate threshold -- deep circuits accumulate noise fast, expect the hardware distribution to diverge more from the simulated one.
- Readout error mitigation: measurement twirling enabled (no added circuit depth).
- Dynamical decoupling enabled (sequence: XpXm) -- a server-side scheduling pass on idle windows, applied on top of the already-transpiled circuit; it will not change the locally measured transpiled depth printed above.
- Warning: /home/oranson/Projects/QuantumResearch/.venv/lib/python3.14/site-packages/stevedore/extension.py:338: DeprecationWarning: Since backends now support running jobs that contain both fractional gates and dynamic circuit, IBMFractionalTranslationPlugin is deprecated as of qiskit-ibm-runtime 0.42.0 and will be removed no sooner than 3 months after the release date. Use IBMDynamicFractionalTranslationPlugin instead.
  obj = plugin(*invoke_args, **invoke_kwds)
- Warning: /home/oranson/Projects/QuantumResearch/.venv/lib/python3.14/site-packages/stevedore/extension.py:338: DeprecationWarning: Since backends now support running jobs that contain both fractional gates and dynamic circuit, IBMFractionalTranslationPlugin is deprecated as of qiskit-ibm-runtime 0.42.0 and will be removed no sooner than 3 months after the release date. Use IBMDynamicFractionalTranslationPlugin instead.
  obj = plugin(*invoke_args, **invoke_kwds)
