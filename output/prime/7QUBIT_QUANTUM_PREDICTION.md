# Quantum prime gap prediction -- run report

Generated automatically by `quantum_prime_gaps.py` at the end of every run -- this
file is overwritten each time, it does not accumulate history (see `NOTES.md` and
`6QUBIT_RESULTS.md` in this directory for hand-written point-in-time snapshots).

- **Timestamp:** 2026-08-12T11:33:15
- **Qubits (`--qubits`):** 7 (dim = 128, shared by the landscape/portrait and prediction pathways)
- **Effective top-k for the paired backward-verification comparison:** 5
- **`--hardware` used this run:** No

## Classical vs. quantum backward verification

Predicting gaps 40-49 (i.e. primes 41-50) from only the first 40 primes' 39 known
gaps, both pathways run at the SAME `top_k` above so the comparison isolates the
pathway itself (numpy FFT vs. an actual Qiskit `QFTGate`), not a differing truncation
assumption.

| Pathway | MAE | Baseline (mean gap) | Baseline (repeat last) | Beats both baselines? |
|---|---:|---:|---:|:---:|
| Classical (numpy FFT) | 4.7128 | 3.2769 | 3.6000 | False |
| Quantum (Qiskit circuit) | 4.7220 | 3.2769 | 3.6000 | False |

**Absolute MAE difference:** 0.009262

**Verdict:** Quantum vs classical MAE differ by 0.0093 at matched top_k=5. This is NOT a sign-convention or rescaling bug -- that's separately proven to floating-point precision (~1e-13) by the "quantum_fft matches np.fft.fft on the zero-padded array" hard check above, which compares the two pathways' spectra bin-for-bin on the identical (padded) array. This MAE gap instead reflects that amplitude encoding's mandatory zero-padding (39 known gaps into a 128-slot register) dilutes spectral power across many more bins, so "top-5" selects a genuinely coarser, different set of frequencies than the classical path's native unpadded spectrum -- an inherent cost of quantum amplitude encoding onto a power-of-2 register, not a computational error.

## Forward prediction (past gap 49, candidate primes from 229)

Quantum-circuit forecast (primary, per this run's request) alongside the classical
forecast for comparison:

| Step | Classical predicted gap | Classical candidate | Quantum predicted gap | Quantum candidate |
|---:|---:|---:|---:|---:|
| 1 | 1.00 | 230.0 | 2.06 | 231.1 |
| 2 | 2.00 | 232.0 | 2.63 | 233.7 |
| 3 | 2.00 | 234.0 | 1.62 | 235.3 |
| 4 | 4.00 | 238.0 | 1.49 | 236.8 |
| 5 | 2.00 | 240.0 | 1.45 | 238.2 |
| 6 | 4.00 | 244.0 | 0.45 | 238.7 |
| 7 | 2.00 | 246.0 | 1.14 | 239.8 |
| 8 | 4.00 | 250.0 | -0.08 | 239.7 |
| 9 | 6.00 | 256.0 | 0.51 | 240.3 |
| 10 | 2.00 | 258.0 | -0.05 | 240.2 |

### Candidate zones -- classical (numpy FFT)

| Frequencies kept | Power fraction | Nearest primes | Mean distance |
|---:|---:|---|---:|
| 1 | 71.6% | 233, 239, 241, 251, 251, 257, 263, 269, 271, 277 | 1.46 |
| 2 | 74.3% | 233, 239, 241, 251, 251, 257, 263, 263, 271, 277 | 1.26 |
| 3 | 76.9% | 233, 239, 239, 241, 251, 257, 263, 263, 271, 271 | 1.70 |
| 4 | 79.3% | 229, 233, 239, 241, 251, 257, 263, 263, 271, 271 | 1.68 |
| 5 | 81.7% | 229, 233, 239, 241, 251, 257, 257, 263, 269, 271 | 1.35 |

### Candidate zones -- quantum circuit

| Frequencies kept | Power fraction | Nearest primes | Mean distance |
|---:|---:|---|---:|
| 1 | 35.0% | 229, 233, 233, 233, 233, 233, 233, 239, 239, 239 | 1.79 |
| 2 | 62.4% | 233, 233, 239, 241, 241, 241, 251, 251, 251, 257 | 1.90 |
| 3 | 69.1% | 233, 233, 239, 239, 239, 241, 241, 241, 241, 241 | 1.81 |
| 4 | 71.2% | 233, 233, 233, 239, 239, 239, 239, 241, 241, 241 | 1.03 |
| 5 | 72.4% | 233, 233, 233, 239, 239, 239, 239, 239, 241, 241 | 1.13 |

## Outputs from this run

| PNG file | Pathway that produced it |
|---|---|
| `gap_sequence.png` | raw data (neither pathway) |
| `amplitude_landscape_sim.png` | landscape re-upload circuit (Qiskit simulator) |
| `frequency_portrait_sim.png` | landscape re-upload circuit (Qiskit simulator) |
| `extended_wave_predicted.png` | both -- classical and quantum series shown together, clearly labeled |

## Console warnings / notable events during this run

- --top-k not given: using top-5 for the paired backward-verification comparison below so classical and quantum are compared at the same truncation level (classical's own default elsewhere in this run is still "keep everything")
- Quantum vs classical MAE differ by 0.0093 at matched top_k=5. This is NOT a sign-convention or rescaling bug -- that's separately proven to floating-point precision (~1e-13) by the "quantum_fft matches np.fft.fft on the zero-padded array" hard check above, which compares the two pathways' spectra bin-for-bin on the identical (padded) array. This MAE gap instead reflects that amplitude encoding's mandatory zero-padding (39 known gaps into a 128-slot register) dilutes spectral power across many more bins, so "top-5" selects a genuinely coarser, different set of frequencies than the classical path's native unpadded spectrum -- an inherent cost of quantum amplitude encoding onto a power-of-2 register, not a computational error.
- quantum path: --top-k not given, defaulting to top-5 frequency components (a full-spectrum reconstruction of the zero-padded register would just reproduce the padding as "predicted" gaps)
- --hardware not set: all pathways ran on the noiseless statevector simulator only.
