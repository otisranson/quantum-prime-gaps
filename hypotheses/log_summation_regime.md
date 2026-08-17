# Logarithmic Summation Hypothesis — Regime Changes in Quantum Prime Gap Terrain

**Date:** August 16, 2026

## Observation

At the 5000-prime scale, I'm seeing the quantum terrain visualizer show approximately three regime changes — step-ups in the MI rolling mean — spaced with increasing distance between them. The earlier regimes are shorter. Later regimes appear to extend longer before the next transition.

## Hypothesis

I think the regime change threshold follows a logarithmic function of accumulated prime gaps. Since the sum of all prime gaps up to N equals the position of the Nth prime on the number line, and the Nth prime grows approximately as N·ln(N) by the Prime Number Theorem, I expect the regime boundaries to follow:

R(N) ~ ln(N)

If this holds, it would mean the quantum circuit is recovering the logarithmic signature of prime distribution as an emergent output of measurement statistics — not as a programmed input.

## Prediction

**Predicted regime-change 4 window: 4585 (predicted August 16, 2026).**

Method: binary-segmentation changepoint detection on the MI rolling mean (K=100, stable across K=30–300) found regime-change windows at 1529, 2501, and 4211. Fitting window = a·ln(k) + b to these three points (k = regime-change index) gives window = 2329.4·ln(k) + 1355.8, extrapolated to k=4. See `regime_fit_5k.py` and `output/prime/analysis/regime_fit_5k.png`.

Two caveats logged before verification, not after:

1. The three detected transitions are **not** all step-ups — the sequence is up, down, up. The "three step-ups" framing in the Observation above does not hold under an unbiased changepoint detector. The fit was still run on all three points as directed, but this is a real discrepancy with the original observation, not a confirmation of it.
2. The predicted window (4585) falls *inside* the existing 5000-prime dataset (windows 0–4995), so it is partially checkable now without running the circuit further. A 4-breakpoint rescan of the existing data does **not** surface a breakpoint near 4585 — the nearest candidate is window 4211 (already one of the three known breakpoints), 374 windows away. This is evidence against the prediction landing where the fit says it should, using data already in hand.

This needs to be verified by running the circuit beyond 5000 primes to see whether a genuine new regime change (not previously in the fitted data) appears near window 4585.

## Status

Unverified. Prediction locked 2026-08-16, ahead of the beyond-5000-primes run. Existing-data check (caveat 2 above) is not favorable to the prediction as stated.
