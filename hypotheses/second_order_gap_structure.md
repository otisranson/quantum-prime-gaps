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

## Status

Unverified. Layer 2's specific proximity test returned a null result (no discriminating power, not "confirmed" and not "ruled out"). The "primes of primes" framing above is not supported by this check.
