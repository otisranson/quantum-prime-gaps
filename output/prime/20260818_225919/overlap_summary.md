# Entropy / expanded-changepoint overlap, with permutation-null baseline -- 20260818_225919

## Where the original MI changepoints come from, and why they stop at N~4999

`regime_fit_5k.py` runs binary-segmentation changepoint detection on quantum-measured mutual information (`per_window` MI from `output/prime/20260816_010716/terrain_5000primes/results_5000primes.json`), produced by an actual quantum circuit run (`terrain_5000primes.py`) over the 5000-prime sequence. The script has no hardcoded N-limit and no compute-cost guard of its own -- it simply reads whatever `per_window` data exists in that one file, which covers ~4996 windows because that is the scope of the only quantum terrain run ever executed in this repo. No 20,000-prime terrain run exists, and MI is a quantum measurement, not a function of the gap sequence alone, so it cannot be recovered from `data/primes_20000.json` (primes and gaps only).

**Decision on how this was handled (per explicit instruction):** rather than running a new, large 20k-prime quantum terrain circuit, this reuses the existing 20k-scale changepoint set already produced classically by `layer3_20k_scaleup.py` -- the *same* binary-segmentation algorithm, unmodified core logic, applied to the raw-gap rolling mean instead of MI (because no 20k MI data exists). This is a different signal from the original MI-based detection, not a true extension of it -- named and documented as gap-space throughout, per CLAUDE.md's explicit warning against conflating the two changepoint sets.

## Changepoint count: original vs. expanded

- Original (MI-based, quantum-measured, valid only for gap-index < 4999): **3** -- [1529, 2501, 4211]
- Expanded (gap-space, raw-gap rolling mean, classical, full 20k range): **39**

## Overlap table (all changepoints, nearest flagged window, distance, null p-value)

p-value = fraction of 100,000 uniformly random points in [0, 19999] whose nearest-flagged-window distance is <= this changepoint's observed distance (lower = more surprising under the null of no real relationship).

| changepoint | nearest flagged window | distance | contained | null p-value |
|---|---|---|---|---|
| 499 | [100, 200) (center 150.0) | 349.0 | no | 0.72723 |
| 1271 | [1300, 1400) (center 1350.0) | 79.0 | no | 0.24251 |
| 1890 | [2100, 2200) (center 2150.0) | 260.0 | no | 0.61031 |
| 2290 | [2175, 2275) (center 2225.0) | 65.0 | no | 0.20533 |
| 2728 | [3150, 3250) (center 3200.0) | 472.0 | no | 0.84390 |
| 3314 | [3150, 3250) (center 3200.0) | 114.0 | no | 0.32765 |
| 3715 | [3975, 4075) (center 4025.0) | 310.0 | no | 0.68380 |
| 4120 | [3975, 4075) (center 4025.0) | 95.0 | no | 0.28174 |
| 4521 | [4475, 4575) (center 4525.0) | 4.0 | yes | 0.01746 |
| 4921 | [4500, 4600) (center 4550.0) | 371.0 | no | 0.75071 |
| 5683 | [6075, 6175) (center 6125.0) | 442.0 | no | 0.81801 |
| 6084 | [6075, 6175) (center 6125.0) | 41.0 | yes | 0.13897 |
| 6545 | [6300, 6400) (center 6350.0) | 195.0 | no | 0.50035 |
| 7004 | [6300, 6400) (center 6350.0) | 654.0 | no | 0.95249 |
| 7472 | [7650, 7750) (center 7700.0) | 228.0 | no | 0.55612 |
| 8118 | [8475, 8575) (center 8525.0) | 407.0 | no | 0.78834 |
| 8666 | [8525, 8625) (center 8575.0) | 91.0 | no | 0.27230 |
| 9282 | [9175, 9275) (center 9225.0) | 57.0 | no | 0.18352 |
| 9865 | [9825, 9925) (center 9875.0) | 10.0 | yes | 0.04127 |
| 10335 | [10350, 10450) (center 10400.0) | 65.0 | no | 0.20533 |
| 10810 | [10725, 10825) (center 10775.0) | 35.0 | yes | 0.12101 |
| 11350 | [11350, 11450) (center 11400.0) | 50.0 | yes | 0.16426 |
| 11750 | [11350, 11450) (center 11400.0) | 350.0 | no | 0.72848 |
| 12226 | [12625, 12725) (center 12675.0) | 449.0 | no | 0.82418 |
| 12641 | [12625, 12725) (center 12675.0) | 34.0 | yes | 0.11797 |
| 13251 | [13700, 13800) (center 13750.0) | 499.0 | no | 0.86533 |
| 13722 | [13700, 13800) (center 13750.0) | 28.0 | yes | 0.09912 |
| 14123 | [13700, 13800) (center 13750.0) | 373.0 | no | 0.75297 |
| 14657 | [14775, 14875) (center 14825.0) | 168.0 | no | 0.44748 |
| 15087 | [14775, 14875) (center 14825.0) | 262.0 | no | 0.61364 |
| 15488 | [15475, 15575) (center 15525.0) | 37.0 | yes | 0.12764 |
| 16006 | [15975, 16075) (center 16025.0) | 19.0 | yes | 0.07151 |
| 16407 | [16375, 16475) (center 16425.0) | 18.0 | yes | 0.06811 |
| 16807 | [16375, 16475) (center 16425.0) | 382.0 | no | 0.76194 |
| 17516 | [18075, 18175) (center 18125.0) | 609.0 | no | 0.93354 |
| 18130 | [18075, 18175) (center 18125.0) | 5.0 | yes | 0.02133 |
| 18594 | [18725, 18825) (center 18775.0) | 181.0 | no | 0.47401 |
| 18995 | [19050, 19150) (center 19100.0) | 105.0 | no | 0.30606 |
| 19395 | [19050, 19150) (center 19100.0) | 295.0 | no | 0.66239 |

## Aggregate hit rate vs. null baseline

- Observed: **11 / 39** changepoints fall inside a flagged window.
- Null-expected containment: **6.39 / 39** (closed-form domain coverage by flagged windows = 0.1638, sampled null containment rate = 0.1634 -- these agree, confirming the null simulation is well-calibrated).
- Changepoints individually significant at p < 0.05 (uncorrected): **3 / 39** vs. **2.0** expected by chance alone with no real relationship (5% of 39, since 39 independent tests at alpha=0.05 will produce that many false positives on average even under a true null -- no multiple-comparison correction was applied, so this count needs to clear that bar meaningfully, not just be nonzero, before being read as a real signal).

## Honest interpretation

Observed containment (11/39) is meaningfully above the null-expected rate (6.4/39), and 3 changepoints clear p<0.05 individually vs. 2.0 expected by chance. This is suggestive of a real relationship between gap-space regime boundaries and entropy-deviation windows -- worth a closer look, though still not a fully corrected-for-multiple-comparisons confirmation.
