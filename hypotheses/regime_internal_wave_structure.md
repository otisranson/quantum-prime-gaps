# Internal Wave Structure — Regimes Rhyme

**Date:** August 16, 2026

## Observation

If regime changes are governed by zero-crossings of the prime gap derivative, then each regime contains an internal wave — the local oscillation of gap size between two transitions. The question is what governs the character of that internal wave.

## Hypothesis

I think the internal waves of each regime are self-affine — they rhyme rather than repeat. Not identical, not periodic, but proportionally similar across scales. The wavelength stretches logarithmically as N grows, but the structural gesture — the ratio of peaks to valleys, the slope character, the build and release — repeats transposed to a new scale.

### The three layer structure

- **Layer 1 — Envelope:** Logarithmic function governs regime spacing. Regimes grow longer as N increases.
- **Layer 2 — Transitions:** Derivative of gap sequence governs regime changes. Zero-crossings mark entry and exit from prime deserts.
- **Layer 3 — Internal wave:** Self-affine structure governs the wave shape within each regime. Each regime rhymes with the others at a different scale.

## Implication

I think the primes are neither random nor periodic. They are self-affine — rhyming across scales. The quantum terrain may be the instrument sensitive enough to detect that rhyme directly from measurement statistics, without it being programmed as an input.

## Next steps

Extract the internal wave of each identified regime from the 5000-prime dataset. Compare wave shapes across regimes after normalizing for scale. Test for self-affinity.

## Status

Unverified. Extraction not yet run.

## Empirical Check — MI Landscape, 25 Independent Groups

**Date checked:** August 17, 2026 (note: system clock reports August 18, 2026 for the run timestamp/commit below; treating the two dates as the same session).

**Important scope note, stated up front:** this is *not* the "extract the raw gap sequence within each of the 3 regimes, normalize, overlay" self-affinity test described in Next steps above — that test is still unrun. This is a different, narrower probe: a real quantum measurement (not a classical reconstruction) of mutual information across the *first 100 prime gaps only*, split into 25 independent non-overlapping 4-qubit groups (gaps 0–3, 4–7, ..., 96–99), each run as its own Bell-pair + RY-gap-encoding + approximated-iQFT circuit (the same v3 architecture as `prime_predictor.py`, degree-1 iQFT, local per-group normalization, no feedback between groups since they're independent rather than recurrent). MI between each group's [q0,q1] and [q2,q3] halves computed via `qubit_hierarchy_core`'s Miller-Madow-corrected estimator, same method used elsewhere in this repo. See `mi_landscape_25groups.py` and `output/prime/20260818_000736/{mi_landscape.png,results.json}`.

**Method for locating the changepoints in this 25-group space:** the three known regime changepoints (windows 1529, 2501, 4211) were found in the 5000-prime run's 4996-window overlapping-MI space (`output/prime/20260816_010716/terrain_5000primes/results_5000primes.json`, `config.n_windows`). Mapped proportionally: `group_position = cp / n_windows * 25`. This is a cross-scale comparison — the first 100 gaps (primes up to ~547) versus proportional positions in a run spanning primes up to 48611 — and the two window schemes differ (25 non-overlapping groups of 4 here vs. 4996 overlapping windows of 4 there). Flagging that up front since it means "alignment" here is a much looser claim than a same-scale test would support.

**Results:**

| changepoint window (5000-run) | group position | nearest group | group MI | percentile of 25 groups |
|---|---|---|---|---|
| 1529 | 7.651 | 8 | 0.0391 | 28.0 |
| 2501 | 12.515 | 13 | 0.0198 | **4.0 (minimum of all 25 groups)** |
| 4211 | 21.072 | 21 | 0.2534 | 92.0 |

MI landscape overall: mean=0.1256, std=0.1030, min=0.0198 (group 13), max=0.4080 (group 18).

**Result, stated clearly: no consistent pattern — INCONCLUSIVE, leaning negative.** The three changepoint-equivalent positions land at wildly different points in the MI distribution: one near the bottom decile (2501 → 4th percentile, literally the single lowest-MI group of all 25), one below the median (1529 → 28th percentile), and one near the top decile (4211 → 92nd percentile). If regime changes corresponded to a consistent MI feature (a peak, a trough, an inflection), the three changepoints should land at *similar* percentiles, not spread across nearly the full range. This does not support the internal-wave hypothesis as tested here.

**Caveats on top of the inconclusive result itself:**
1. n=25 groups is a small sample; percentile granularity is coarse (4-point steps), and no null distribution was computed for "how often would 3 points land this spread out by chance" — unlike the Layer 2 magnitude test's discipline, this check stops at descriptive percentiles rather than a proper randomization test. A follow-up should build that null before treating "spread out" as meaningfully different from a real anti-correlation.
2. This probes only the first 100 gaps, a regime far short of where any of the three changepoints actually occur in gap-index space — it's testing whether the *shape* of an early, unrelated MI landscape happens to echo proportionally-mapped positions from a much later part of the sequence, which is a weaker and more indirect test than same-scale self-affinity.
3. The originally-planned Layer 3 test (raw gap sequence extracted from *within* each of the 3 real regimes, normalized, overlaid) is still the more direct test of "do the regimes rhyme" and remains unrun.

**What held:** the run itself is clean — sieve-derived primes cross-checked by independent trial division, 25/25 groups completed, MI values are a real, non-trivial spread (not degenerate at 0 or a constant), consistent with a working circuit. The negative/inconclusive result is about the hypothesis, not the measurement.
