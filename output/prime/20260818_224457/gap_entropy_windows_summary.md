# Gap-sequence sliding-window entropy -- 20260818_224457

Source: `cache:20000` (19999 gaps). Window size=100, step=25, 796 windows total.

## Binning methodology

Each distinct integer gap value observed within a window is its own histogram bin (a categorical count over observed values, not fixed-width numeric binning) -- applied identically across every window. Shannon entropy H(N) is reported in bits (log base 2); Boltzmann-style entropy S(N)=ln(Omega) is reported in nats (natural log) -- these use different units by the definitions requested and should not be compared directly without accounting for that.

## Entropy growth rate

Log-linear fit: **H(N) ~ 0.160767 * ln(N+2) + 2.108679** (R^2=0.7488)
Residual std: 0.0894; windows flagged where |residual| > 2.0 sigma.
Pearson r(H, S) across all windows: 0.8798 (both track window value-diversity by construction, so a high correlation here is an expected consistency check, not an independent finding).

**Confound note:** average gap size is already known to grow ~ln(N) for this dataset (hypotheses/regime_internal_wave_structure.md, 40-Regime Characterization: mean gap r=0.85, p<0.0001). A wider typical gap gives each fixed-size window more distinct integer values to draw from, so upward H(N)/S(N) trend with N is expected under a structurally boring gap distribution -- the log-linear fit above is intended to characterize and remove exactly that expected growth before anything is flagged as unusual.

## Flagged windows (candidate regime boundaries, n=39)

Flagged because H(N) deviates from the fitted trend by more than 2.0 residual standard deviations. These are candidates, not confirmed regime boundaries.

| window start | window end | center | H(N) | fit | residual |
|---|---|---|---|---|---|
| 0 | 100 | 50.0 | 2.5181 | 2.7439 | -0.2258 |
| 25 | 125 | 75.0 | 2.5519 | 2.8070 | -0.2551 |
| 50 | 150 | 100.0 | 2.6513 | 2.8522 | -0.2009 |
| 100 | 200 | 150.0 | 2.7348 | 2.9164 | -0.1816 |
| 1300 | 1400 | 1350.0 | 3.4478 | 3.2677 | +0.1801 |
| 1475 | 1575 | 1525.0 | 3.0736 | 3.2873 | -0.2137 |
| 2100 | 2200 | 2150.0 | 3.5392 | 3.3424 | +0.1968 |
| 2175 | 2275 | 2225.0 | 3.1668 | 3.3479 | -0.1811 |
| 3150 | 3250 | 3200.0 | 3.5918 | 3.4063 | +0.1855 |
| 3975 | 4075 | 4025.0 | 3.6341 | 3.4432 | +0.1910 |
| 4400 | 4500 | 4450.0 | 3.2525 | 3.4593 | -0.2068 |
| 4475 | 4575 | 4525.0 | 3.6765 | 3.4620 | +0.2145 |
| 4500 | 4600 | 4550.0 | 3.6686 | 3.4629 | +0.2057 |
| 6075 | 6175 | 6125.0 | 3.7017 | 3.5106 | +0.1911 |
| 6300 | 6400 | 6350.0 | 3.2624 | 3.5164 | -0.2540 |
| 7650 | 7750 | 7700.0 | 3.7523 | 3.5474 | +0.2049 |
| 8475 | 8575 | 8525.0 | 3.7643 | 3.5638 | +0.2005 |
| 8500 | 8600 | 8550.0 | 3.7918 | 3.5643 | +0.2276 |
| 8525 | 8625 | 8575.0 | 3.7495 | 3.5647 | +0.1848 |
| 8900 | 9000 | 8950.0 | 3.3092 | 3.5716 | -0.2624 |
| 9175 | 9275 | 9225.0 | 3.3928 | 3.5765 | -0.1836 |
| 9800 | 9900 | 9850.0 | 3.7965 | 3.5870 | +0.2095 |
| 9825 | 9925 | 9875.0 | 3.8225 | 3.5874 | +0.2351 |
| 10350 | 10450 | 10400.0 | 3.4140 | 3.5957 | -0.1817 |
| 10725 | 10825 | 10775.0 | 3.8520 | 3.6014 | +0.2506 |
| 11350 | 11450 | 11400.0 | 3.4117 | 3.6105 | -0.1988 |
| 12625 | 12725 | 12675.0 | 3.4079 | 3.6275 | -0.2197 |
| 13700 | 13800 | 13750.0 | 3.4358 | 3.6406 | -0.2049 |
| 14775 | 14875 | 14825.0 | 3.8489 | 3.6527 | +0.1962 |
| 15300 | 15400 | 15350.0 | 3.8454 | 3.6583 | +0.1871 |
| 15475 | 15575 | 15525.0 | 3.4351 | 3.6601 | -0.2250 |
| 15875 | 15975 | 15925.0 | 3.8493 | 3.6642 | +0.1851 |
| 15975 | 16075 | 16025.0 | 3.3754 | 3.6652 | -0.2898 |
| 16225 | 16325 | 16275.0 | 3.4454 | 3.6677 | -0.2223 |
| 16250 | 16350 | 16300.0 | 3.4890 | 3.6680 | -0.1790 |
| 16375 | 16475 | 16425.0 | 3.8632 | 3.6692 | +0.1940 |
| 18075 | 18175 | 18125.0 | 3.4964 | 3.6850 | -0.1886 |
| 18725 | 18825 | 18775.0 | 3.4620 | 3.6907 | -0.2287 |
| 19050 | 19150 | 19100.0 | 3.4844 | 3.6934 | -0.2091 |

## Overlap with known MI-drift regime markers

Comparison scope: only the 3 quantum-MI-based changepoints from `regime_fit_5k.py` (windows [1529, 2501, 4211]), valid only within gap-index < 4999 (the 5000-prime run this detector actually ran on). This is deliberately not compared against the separate 39-point gap-space changepoint set from `layer3_20k_scaleup.py` -- CLAUDE.md documents those as a different signal (raw-gap rolling mean, not MI) and warns against conflating the two sets.

13 of the 39 flagged windows fall within the valid comparison range.

| MI changepoint | inside a flagged window? | nearest flagged window center | distance |
|---|---|---|---|
| 1529 | yes | 1525.0 | 4.0 |
| 2501 | no | 2225.0 | 276.0 |
| 4211 | no | 4025.0 | 186.0 |

**Result, stated plainly:** 1 of 3 known MI-drift changepoints fall inside a flagged entropy-deviation window. Reported as overlap/non-overlap only -- no null distribution was computed for how often a random window would contain a given changepoint by chance, so this should not be read as a significance claim either way, only as a descriptive comparison.

