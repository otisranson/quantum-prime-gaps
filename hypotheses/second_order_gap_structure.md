# Second Order Prime Gap Structure — The Derivative Dimension

**Date:** August 16, 2026

## Observation

The quantum terrain visualizer shows regime changes that do not follow a simple step-up pattern — the algorithmic breakpoints show an up-down-up signature. This suggests regime changes are not responses to gap size alone, but to the *change* in gap size — the derivative of the gap sequence.

## Hypothesis

I think prime gap structure has two dimensions:

- **Dimension 1:** Gap size — the distance to the next prime. This is what the terrain currently visualizes as elevation.
- **Dimension 2:** Gap acceleration — whether gaps are growing or shrinking at each window. This is the derivative of Dimension 1.

Regime changes correspond to zero-crossings of Dimension 2 — transitions into and out of prime deserts. A large isolated prime gap is not itself the regime change. The regime change is the transition into and out of that desert. The up-down-up signature in the MI rolling mean may be encoding exactly this — the circuit detecting the derivative signal, not the gap size directly.

## The "primes of primes" framing

Just as primes are the irreducible structure of the integers, I think the zero-crossings of the gap derivative are the irreducible structural events of the prime gap sequence — the points where the wave changes character.

## Next steps

Compute Dimension 2 explicitly from the existing 5000-prime gap sequence. Plot zero-crossings against observed regime change windows. Test whether alignment is better than chance.

## Empirical Check — Layer 2

**Date checked:** August 16, 2026

**Method:** Reconstructed the full 4999-point prime gap sequence from the 5000-prime dataset, computed the first difference (Dimension 2), and found every sign-change (zero-crossing). Checked proximity of the three known regime changepoints (windows 1529, 2501, 4211) to the nearest zero-crossing, within a 50-window radius. See `gap_derivative_zero_crossings.py` and `output/prime/analysis/gap_derivative_zero_crossings.png`.

**Raw result:** All three changepoints have a zero-crossing within 2 windows (distances: 2, 0, 1).

**Result, stated clearly: zero-crossings do NOT meaningfully align with the changepoints — the raw result above is not evidence for the hypothesis.** The gap derivative changes sign at 65% of all indices (3264 zero-crossings out of 4998). At that density, every single index in the entire 4998-length sequence — not just the 3 changepoints, literally all of them — is already within 50 windows of *some* zero-crossing (base rate = 100%, computed directly, not assumed). Any 3 points I could have picked, changepoints or not, would have "aligned" by this test. The proximity check as specified has no discriminating power over this data; it cannot distinguish the hypothesis from chance.

This does not rule out a real relationship between gap derivative structure and regime changes — it means this specific test (nearest zero-crossing within a fixed radius) cannot detect one, because zero-crossings are too dense. A real test would need to look at something zero-crossings don't do almost everywhere: e.g. the *magnitude* or *run-length* of the derivative around changepoints vs. elsewhere, not just crossing presence/absence.

## Empirical Check — Layer 2 Revised

**Date checked:** August 16, 2026

**Method:** Smoothed the prime gap sequence with the same K=100 rolling mean used to find the three known changepoints, took the first difference of that smoothed sequence, and found its zero-crossings (local peaks/troughs of the smoothed gap curve). See `smoothed_gap_derivative_zero_crossings.py` and `output/prime/analysis/smoothed_gap_derivative_zero_crossings.png`.

**Density:** 2314 zero-crossings out of 4899 indices — **density = 0.472**. This is *not* meaningfully sparser than the raw-derivative density (0.653) from the first Layer 2 test. Smoothing the gap sequence at K=100 did not suppress the zero-crossing rate the way the request's Step 3 expected.

**Base rate (computed before the proximity test was evaluated):** at this density, **100% of all 4899 indices** are within ±50 windows of some zero-crossing — same as the raw-derivative test. P(a randomly chosen point passes the proximity test) = 1.000.

**Proximity result:** all three changepoints pass, with the nearest zero-crossing at distance 0, 1, and 0 windows respectively (windows 1529→1529, 2501→2502, 4211→4211).

**State this precisely:** the 0/1/0 distances look like an exact hit, but they are not informative. At 47% zero-crossing density, the *expected* distance from any randomly chosen point to its nearest zero-crossing is close to 1 anyway — landing within 0–1 windows is the typical outcome for an arbitrary point, not a rare one. Since the base rate is 100%, this test cannot fail, so it cannot confirm anything either.

**State whether this confirms, refutes, or is inconclusive: INCONCLUSIVE.** Same failure mode as the raw-derivative test — smoothing didn't sparsify the zero-crossings enough to give the proximity check any discriminating power. This is a second null result for Layer 2, not a second piece of confirming evidence, despite the surface appearance of the raw distances.

**What a real test would need:** a feature that isn't already present at ~50% of all indices — e.g. the size of the smoothed-derivative's peak/trough excursion around a changepoint vs. a typical excursion elsewhere, or run-length of sign persistence, compared against a proper null distribution (e.g. changepoint positions randomized many times and the proximity-pass rate recomputed) rather than a single fixed-radius proximity check.

## Empirical Check — Layer 2 Magnitude Test

**Date checked:** August 16, 2026

**Method:** Same K=100-smoothed gap derivative as the revised proximity test, but instead of checking whether a zero-crossing exists nearby (a test with no discriminating power, per above), this measures the derivative's absolute *magnitude* at each of the three known changepoint windows and compares it to a null distribution: 10,000 windows sampled at random (seed=42) from the same 4899-point derivative sequence. See `layer2_magnitude_test.py` and `output/prime/analysis/layer2_magnitude_test.png`.

**Null distribution:** mean=0.076, std=0.073, 90th percentile=0.180.

**Percentile ranks, stated explicitly:**

| changepoint window | \|deriv\| | percentile in null distribution |
|---|---|---|
| 1529 | 0.0400 | **38.3** |
| 2501 | 0.1200 | **79.9** |
| 4211 | 0.1000 | **74.6** |

**Conclusion, stated clearly: Layer 2 magnitude hypothesis is REFUTED.** None of the three changepoints exceed the 90th-percentile threshold set in advance (window 1529 is below the *median* of the null distribution, at the 38th percentile). This is not a base-rate/no-power failure like the two proximity tests above — magnitude is continuously distributed and this test could have come out either way. It came out negative: regime changes in the MI rolling mean are not reliably accompanied by unusually large swings in the smoothed gap derivative.

Taken together with the two inconclusive proximity tests, Layer 2 (as stated — "regime changes correspond to zero-crossings of Dimension 2") now has one clean negative result and no positive evidence. The "primes of primes" framing is not supported by any of the three checks run.

## Status

Layer 2 magnitude test: **refuted** (2026-08-16). Both proximity tests: inconclusive (no discriminating power). Net: no supporting evidence found for Layer 2 across three independent empirical checks; one direct refutation.
