# Quantum prime gap prediction -- real hardware run report

Generated automatically by `quantum_prime_gaps.py --hardware` when the prediction
circuit's hardware job completes. Overwritten on every hardware run -- prior results
live in git history, not accumulated here.

- **Timestamp:** 2026-08-11T21:57:49
- **Backend:** ibm_kingston (selected dynamically via `least_busy`, not hardcoded)
- **Job ID:** d9tt65k98n5s73922oog
- **Queue depth at selection:** 0 pending jobs
- **Shots used:** 4096
- **Transpiled circuit depth:** 913 (exceeds the 50-gate warning threshold)
- **Transpiled gate count:** 1465
- **Readout error mitigation applied:** Yes (measurement twirling)
- **Queue wait time:** 1.1s

## Hardware vs. simulated amplitude landscape

**Mean absolute difference (hardware vs. simulated probability per basis state):**
0.00962 -- this is the noise floor: on a noiseless simulator this would be
exactly 0, so this number *is* the measurable effect of real hardware noise on the
prediction circuit's readout.

## Candidate zones under hardware noise

The rigorous comparison (simulated quantum vs. classical, both noiseless) is in `7QUBIT_QUANTUM_PREDICTION.md`. For the hardware-noise-specific question: full hardware candidate zones cannot be computed from a single measurement setting -- Sampler destroys phase, and `_dft_reconstruct` needs it; recovering it would need full state tomography, out of scope for this run. The magnitude-from-hardware/phase-from-simulator hybrid below answers only the narrower question of whether *readout noise alone* moves the zones -- at max frequencies kept, the hybrid's mean nearest-prime distance is 1.02 vs. classical's 1.35 (closer to actual primes than classical under this magnitude-only noise slice -- NOT a statement about the full hardware-noise question, which needs tomography to answer properly).

### Candidate zones -- classical (numpy FFT, noiseless, for reference)

| Frequencies kept | Power fraction | Nearest primes | Mean distance |
|---:|---:|---|---:|
| 1 | 71.6% | 233, 239, 241, 251, 251, 257, 263, 269, 271, 277 | 1.46 |
| 2 | 74.3% | 233, 239, 241, 251, 251, 257, 263, 263, 271, 277 | 1.26 |
| 3 | 76.9% | 233, 239, 239, 241, 251, 257, 263, 263, 271, 271 | 1.70 |
| 4 | 79.3% | 229, 233, 239, 241, 251, 257, 263, 263, 271, 271 | 1.68 |
| 5 | 81.7% | 229, 233, 239, 241, 251, 257, 257, 263, 269, 271 | 1.35 |

### Candidate zones -- quantum circuit (noiseless simulator, for reference)

| Frequencies kept | Power fraction | Nearest primes | Mean distance |
|---:|---:|---|---:|
| 1 | 35.0% | 229, 233, 233, 233, 233, 233, 233, 239, 239, 239 | 1.79 |
| 2 | 62.4% | 233, 233, 239, 241, 241, 241, 251, 251, 251, 257 | 1.90 |
| 3 | 69.1% | 233, 233, 239, 239, 239, 241, 241, 241, 241, 241 | 1.81 |
| 4 | 71.2% | 233, 233, 233, 239, 239, 239, 239, 241, 241, 241 | 1.03 |
| 5 | 72.4% | 233, 233, 233, 239, 239, 239, 239, 239, 241, 241 | 1.13 |

### Candidate zones -- magnitude-from-hardware / phase-from-simulator HYBRID (NOT a full hardware reconstruction)

| Frequencies kept | Power fraction | Nearest primes | Mean distance |
|---:|---:|---|---:|
| 1 | 4.1% | 229, 229, 229, 229, 229, 229, 229, 229, 229, 229 | 0.76 |
| 2 | 7.6% | 229, 233, 229, 229, 233, 233, 233, 233, 233, 233 | 1.26 |
| 3 | 10.3% | 229, 233, 233, 229, 233, 233, 233, 233, 233, 233 | 1.13 |
| 4 | 12.7% | 233, 233, 233, 229, 233, 233, 233, 233, 233, 233 | 0.90 |
| 5 | 15.2% | 229, 233, 229, 229, 229, 229, 229, 227, 227, 229 | 1.02 |

## Outputs from this run

| PNG file | Pathway that produced it |
|---|---|
| `hardware_amplitude_landscape.png` | prediction circuit, hardware only |
| `hardware_vs_sim_comparison.png` | prediction circuit, simulated + hardware side by side |
| `hardware_frequency_portrait.png` | prediction circuit, simulated + hardware overlay |

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
