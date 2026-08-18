# Entropy velocity and residual-amplitude decay -- 20260818_235249

## Data-availability note

`output/prime/20260818_224457/results.json` only stored the 39 flagged windows, not the full 796-window H(N) series. Recomputed deterministically (same cache, same window grid, same binning) and verified to exactly reproduce the stored flagged-window H values and trend fit (a=0.160767) before use.

## Velocity anomalies

- Velocity-residual std (empirical dH/dN minus analytic a/(N+2), a=0.160767): 0.002089
- Flagged (|residual| > 2.0 sigma): **40 of 796 windows**

| window start | center N | empirical dH/dN | analytic dH/dN | residual |
|---|---|---|---|---|
| 1500 | 1550.0 | 0.005236 | 0.000104 | +0.005132 |
| 1675 | 1725.0 | -0.004203 | 0.000093 | -0.004297 |
| 2150 | 2200.0 | -0.005451 | 0.000073 | -0.005524 |
| 3125 | 3175.0 | 0.006377 | 0.000051 | +0.006327 |
| 4425 | 4475.0 | 0.004399 | 0.000036 | +0.004364 |
| 4450 | 4500.0 | 0.005346 | 0.000036 | +0.005311 |
| 5275 | 5325.0 | 0.004764 | 0.000030 | +0.004734 |
| 6275 | 6325.0 | -0.005765 | 0.000025 | -0.005790 |
| 6325 | 6375.0 | 0.005359 | 0.000025 | +0.005334 |
| 6350 | 6400.0 | 0.004284 | 0.000025 | +0.004259 |
| 6900 | 6950.0 | -0.004530 | 0.000023 | -0.004554 |
| 8175 | 8225.0 | -0.005165 | 0.000020 | -0.005185 |
| 8550 | 8600.0 | -0.006548 | 0.000019 | -0.006567 |
| 8575 | 8625.0 | -0.005034 | 0.000019 | -0.005053 |
| 8875 | 8925.0 | -0.006473 | 0.000018 | -0.006491 |
| 8925 | 8975.0 | 0.005729 | 0.000018 | +0.005711 |
| 9225 | 9275.0 | 0.004549 | 0.000017 | +0.004531 |
| 9850 | 9900.0 | -0.005220 | 0.000016 | -0.005236 |
| 10150 | 10200.0 | 0.004331 | 0.000016 | +0.004315 |
| 10325 | 10375.0 | -0.004341 | 0.000015 | -0.004357 |
| 10375 | 10425.0 | 0.005033 | 0.000015 | +0.005018 |
| 10700 | 10750.0 | 0.004284 | 0.000015 | +0.004269 |
| 10750 | 10800.0 | -0.004688 | 0.000015 | -0.004702 |
| 13675 | 13725.0 | -0.005123 | 0.000012 | -0.005135 |
| 14800 | 14850.0 | -0.005960 | 0.000011 | -0.005971 |
| 15000 | 15050.0 | 0.004820 | 0.000011 | +0.004809 |
| 15500 | 15550.0 | 0.005279 | 0.000010 | +0.005269 |
| 15950 | 16000.0 | -0.006098 | 0.000010 | -0.006108 |
| 16000 | 16050.0 | 0.004592 | 0.000010 | +0.004582 |
| 16025 | 16075.0 | 0.004608 | 0.000010 | +0.004598 |
| 16300 | 16350.0 | 0.005635 | 0.000010 | +0.005625 |
| 16325 | 16375.0 | 0.005450 | 0.000010 | +0.005440 |
| 16400 | 16450.0 | -0.005481 | 0.000010 | -0.005491 |
| 16425 | 16475.0 | -0.005448 | 0.000010 | -0.005458 |
| 17550 | 17600.0 | -0.004200 | 0.000009 | -0.004209 |
| 18025 | 18075.0 | -0.004244 | 0.000009 | -0.004253 |
| 18900 | 18950.0 | 0.004377 | 0.000008 | +0.004369 |
| 18975 | 19025.0 | -0.005596 | 0.000008 | -0.005604 |
| 19300 | 19350.0 | -0.004216 | 0.000008 | -0.004225 |
| 19350 | 19400.0 | 0.004202 | 0.000008 | +0.004194 |

## Residual amplitude decay

Rolling window: 40 H(N) samples (span=1000 gap-index units). Chosen over the raw 100-gap-index scale because that maps to only 4 H(N) samples per window -- too few for a stable std estimate; this window balances estimator stability against resolution (see script output for the exact relative-std-error figure).

| model | R^2 | notes |
|---|---|---|
| flat/constant | 0.0000 | 0 by construction (the mean-baseline R^2 is defined against) |
| 1/sqrt(N) decay | 0.0103 | slope=-0.1123, intercept=0.0535 |
| exponential decay | -0.0046 | log-slope=0.000000 |

**Winner: inv_sqrt** (R^2=0.0103), permutation-null p=0.0060 (n_perm=2000).

## Value-anomaly / velocity-anomaly overlap

Both flagging passes run on the identical window grid, so overlap is exact window-index equality, not a distance-based comparison.

- Value-flagged: 39, velocity-flagged: 40, population: 796
- Observed overlap: **0** (expected under independence: 1.96)
- Exact hypergeometric p-value (P[overlap >= observed]): **1.0000**

This is at or below what independence would predict. Not statistically distinguishable from chance at the p<0.05 level -- reported as an honest descriptive comparison, not a confirmed relationship.

## Damped-oscillation hypothesis

**Inconclusive.** The inv_sqrt model nominally beats the flat baseline (R^2=0.0103), but the permutation null (p=0.0060) does not clear the p<0.05 bar used here, or the R^2 itself is below the 0.1 threshold -- the apparent improvement over flat is not clearly distinguishable from what an unstructured series would produce by chance at this sample size.

