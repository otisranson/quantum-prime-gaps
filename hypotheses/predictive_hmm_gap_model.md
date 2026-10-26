# Predictive Framing — Hidden Markov Model Under the Log Channel

**Date:** 2026-10-26

## Observation

Across everything tested so far the only real structure in the 20k gap sequence is the log-growth
trend (PNT: local mean gap ~ ln p). Every wave/regime/self-similarity test came back null once that
trend was accounted for. Those were all *descriptive* tests of structure. This switches to a
*predictive* one: does any model beat the trend alone at forecasting the next gap on held-out primes?

## Hypothesis

I think there is a "force" under the log channel, i.e. a slowly varying hidden state that modulates
the local gap scale around the ln p drift, and that a Hidden Markov Model over the normalized gap can
recover it and use it to predict the next gap better than the drift alone.

## Setup (committed before any model was fit or scored)

`gap_hmm_predictive.py`. Gaps k = gap/2 (skip the odd 2->3 gap). Every model shares one emission
family, geometric on k with mean m*ln(p)/2, so log-likelihoods are exactly comparable.

- **B0**: m = 1 fixed (Cramer + PNT drift, 0 params).
- **B0f**: m fitted (1 param).
- **B1**: m depends on previous gap class k_prev in {1,2,3,4,>=5}. Observed Markov dependence; the
  known Lemke Oliver-Soundararajan-style effect should live here.
- **HMM**: K hidden states in {1,2,3,4,6}, each its own m_s, full transition matrix, Baum-Welch with
  6 restarts, best restart by train likelihood.

Split: first 80% train, last 20% test (primes ~ 180k-225k). Score = one-step-ahead predictive
log-likelihood on test with frozen parameters (HMM via the forward filter, past data only). Paired
per-gap differences give a z-score.

## Prediction (written before running)

1. B0f > B0 slightly (m should fit near 1). B1 > B0f on held-out data: real but tiny, order 1e-3
   nats/gap, because the prev-gap dependence is a known small effect.
2. **HMM does not beat B1 on held-out data.** Concretely, "HMM wins" requires BOTH
   improvement over B1 >= 0.002 nats/gap AND z >= 3. I predict this fails for every K.
3. For K >= 2 the fitted m_s will be close together (spread < ~0.15) or the states will be doing the
   job B1 already does (short-lived, tied to gap size), not a slow regime. If a slow regime existed
   the fitted self-transition probabilities would be near 1 with well-separated m_s; I predict not.
4. Train BIC will prefer K=1 or at most K=2.

Reasoning: consecutive prime gaps look nearly independent given ln p apart from small residue-class
correlations, and the hypotheses/ log is uniformly null on slow regime structure.

## Status

Prediction locked 2026-10-26, ahead of the first run.

## Empirical Check — First Run

**Date checked:** 2026-10-26. `output/prime/20261026_161334/gap_hmm_predictive/results.json`.
Train 15,998 gaps, test 4,000 (held-out tail). Held-out nats/gap, difference vs B1 (paired z):

| model | params | test nats/gap | vs B0 | vs B1 (z) |
|---|---|---|---|---|
| B0 (force alone) | 0 | -2.71816 | 0 | -0.00393 (-3.95) |
| B0f | 1 | -2.71819 | -0.00003 | -0.00396 (-3.98) |
| **B1 (prev-gap class)** | 5 | **-2.71423** | **+0.00393** | 0 |
| HMM K=1 | 1 | -2.71819 | -0.00003 | -0.00396 (-3.98) |
| HMM K=2 | 4 | -2.71833 | -0.00017 | -0.00410 (-4.10) |
| HMM K=3 | 9 | -2.71827 | -0.00011 | -0.00404 (-4.06) |
| HMM K=4 | 16 | -2.71826 | -0.00010 | -0.00403 (-4.05) |
| HMM K=6 | 36 | -2.71824 | -0.00008 | -0.00401 (-4.03) |

**Against the locked prediction:**
1. Held. B0f ~ B0 (fitted m = 1.004). B1 beats B0f on held-out data at z ~ 4; the size (0.004
   nats/gap) is larger than the 1e-3 I guessed, but still tiny. B1's multipliers: after a gap of 2 the
   next gap runs ~15% above the force (m=1.147), after a gap >=10 it runs ~5% below (0.946). Small
   repulsion between consecutive gaps, the Lemke Oliver-Soundararajan direction.
2. Held. No HMM beats B1; every HMM is slightly *worse* than B0 on held-out data (K>=2 overfit).
3. Held. Fitted m_s spread is <= 0.07 (all states within ~1.00-1.07), and self-transitions are
   0.82-0.93, i.e. short-lived, not a slow regime. For K>=3 the extra states are duplicates at
   m=1.00; one state carries a mild +5% bump.
4. Held. Train BIC is lowest for B1, then K=1; it worsens monotonically with K.

**Conclusion: REFUTED for this model class.** Under a geometric-mean-scale emission, a hidden
slowly-varying "force" modulating the local gap scale around ln p does not exist in any way a
held-out forecast can use. What the data supports is the log channel (deterministic, in B0) plus a
~0.004 nats/gap (~0.15%) gain from knowing the previous gap.

**Caveats, not yet tested:**
- The geometric emission family is misspecified: real gaps are not geometric, since gaps divisible by
  6 are favoured (singular series / mod-3 structure). A hidden state could in principle be needed
  only to absorb that misfit; here it wasn't, but a richer emission (categorical on k, or with
  explicit mod-6 factors) is the fairer test of whether any latent state exists.
- Only the scale (mean) is state-dependent. A state could modulate shape/variance instead.
- Single split (last 20%), single seed set. Not yet checked across rolling-origin folds.

## Round 2 — Singular-Series Emission (prediction locked before run, 2026-10-26)

Addresses the first caveat above. `gap_hmm_singular_series.py`. Emission over k = gap/2, k = 1..150:
`P(k | m, c, p) ∝ S(2k)^c * exp(-k * 2 / (m ln p))`, normalised numerically, with
S(g) = prod over odd primes q | g of (q-1)/(q-2) (Hardy-Littlewood singular series). ln p is quantised
to 0.02 so normalisers can be tabulated; this is applied identically to every model. Each HMM state
has its own (m_s, c_s), so states may modulate shape (c) as well as scale (m).

Models: S0f (m fitted, c=1), S1 (m by previous-gap class), S1c (m and c by previous-gap class, the
strongest non-latent baseline, 10 params), HMM K = 1,2,3,4. Same 80/20 split and scoring as round 1.

**Predictions:**
1. S0f beats round-1 B0f by a clear margin (>= 0.01 nats/gap): the geometric misfit was real.
2. S1c beats S0f, but by less than the round-1 B1 gain (< 0.004): part of the "previous gap" effect in
   round 1 was the mod-3/5 structure leaking through, not true gap repulsion.
3. **No HMM beats S1c.** "HMM wins" needs BOTH >= 0.002 nats/gap AND z >= 3 over S1c. I predict this
   fails for every K, and that K>=2 fitted (m_s, c_s) collapse to near-duplicates or short-lived
   states (self-transition < 0.95), as in round 1.
4. Train BIC picks K=1 among the HMMs.

## Empirical Check — Round 2 (Singular-Series Emission)

**Date checked:** 2026-10-26. `output/prime/20261026_170802/gap_hmm_singular_series/results.json`.
Same 80/20 split (15,998 train, 4,000 test). Held-out nats/gap:

| model | params | train BIC/gap | test nats/gap | vs S0f | vs S1c (z) |
|---|---|---|---|---|---|
| S0f (force + singular series) | 1 | 5.09552 | -2.66326 | 0 | -0.00515 (-3.36) |
| S1 (m by prev-gap class) | 5 | 5.09232 | -2.65892 | +0.00434 | -0.00081 (-0.71) |
| **S1c (m, c by prev-gap class)** | 10 | 5.08962 | **-2.65811** | **+0.00515** | 0 |
| HMM K=1 | 2 | 5.09566 | -2.66299 | +0.00027 | -0.00488 (-3.22) |
| HMM K=2 | 6 | 5.09809 | -2.66299 | +0.00027 | -0.00488 (-3.22) |
| HMM K=3 | 12 | 5.10172 | -2.66299 | +0.00027 | -0.00489 (-3.22) |
| HMM K=4 | 20 | 5.10656 | -2.66299 | +0.00027 | -0.00489 (-3.22) |

(Bug caught pre-run: the first singular-series routine skipped the factor of 2 and gave S(6)=1; fixed
and checked against S(6)=2, S(10)=4/3, S(30)=8/3 before anything was fit.)

**Against the locked prediction:**
1. **Held.** S0f is -2.66326 vs round-1 B0f -2.71819: +0.055 nats/gap. The geometric misfit was large;
   the singular series is a far better "force" than ln p alone.
2. **FAILED.** I predicted the previous-gap gain would shrink below the round-1 0.004 once mod-3/5/7
   structure was explicit. It grew: S1c beats S0f by 0.00515 (z ~ 3.4), S1 by 0.00434. So the
   previous-gap effect is not leakage of the residue structure; it survives, and is slightly larger,
   with the singular series already in the base model. The extra shape freedom (c, S1c over S1) adds
   only 0.0008 and is not distinguishable from zero (z = -0.71 for S1 vs S1c), so it is the *scale*
   change by previous gap that matters. Fitted m by previous gap class: 0.965 / 0.852 / 0.849 / 0.902 /
   0.780 for k_prev = 1, 2, 3, 4, >=5 (monotone-ish: a big gap is followed by a shorter-scale one, a
   twin-prime gap by a longer-scale one).
3. **Held.** No HMM beats S1c (all about -0.0049, z ~ -3.2). K = 2, 3, 4 all score *identically* to
   K = 1 (-2.66299): the fitted states are copies of one another (m = 0.826-0.831, c = 1.05-1.10 in
   every state). Self-transition 0.88-0.91 is meaningless between indistinguishable states. The only
   thing the latent model learns is a global shape tweak, c ~ 1.07, worth +0.00027 nats/gap.
4. **Held.** Train BIC rises monotonically with K among the HMMs (K=1 lowest); S1c has the lowest of all.

**Conclusion: the latent-state hypothesis is refuted a second time, now under a properly specified
emission.** With the mod-3/5/7 structure explicit, there is no residual hidden variable for a state to
absorb: a K-state HMM with free scale AND free shape per state collapses to a single state.
The only usable structure beyond the deterministic force is the observed previous gap.

**Still untested:** rolling-origin folds (single split used); gap-to-gap dependence beyond lag 1 (e.g.
two previous gaps, or the residue of the prime itself, p mod 30); a *non-stationary* latent that
changes with position in ways a stationary-transition HMM can't represent.

## Round 3 — Wheel Constraint, Longer Memory, Rolling Folds (prediction locked before run, 2026-10-26)

`gap_hmm_wheel.py`. Motivation: round 2 found a previous-gap effect that *grew* once the singular series
was explicit. One candidate explanation I had not tested: given p, the next prime p+g cannot be divisible
by 3, 5 (or 7), so conditioning on p mod 30 (or 210) removes impossible gaps outright. That is a hard
constraint, a second-level "force", and the Lemke Oliver-Soundararajan bias is a statement about
consecutive-prime *residues*, so a previous-gap context that is capped (k_prev in 1..5+) may be a poor
proxy for it.

Emission: round 2's `S(2k)^c * exp(-k*2/(m ln p))` over k = 1..150, additionally masked to gaps where
p+2k is not divisible by the wheel primes {3,5} (W=30) or {3,5,7} (W=210); normalised over the allowed
k. c = 1 fixed for all context models. Data restricted to p >= 11 so the mask is valid (otherwise p+2k
may equal a wheel prime). Same 80/20 primary split; plus 5 expanding-window folds (train first
f in {0.5..0.9}, test next 0.1) for the non-latent models.

Models (W=30 unless stated): M0 (one m), M0_210, M1 (m by capped previous gap, 5 classes), M1r (m by
previous k mod 15, i.e. previous gap mod 30, 15 classes), M2 (m by capped previous two gaps, 25
classes), HMM K=2 on the M0 emission.

**Predictions:**
1. M0 beats round-2 S0f by >= 0.05 nats/gap. M0_210 beats M0 by a further >= 0.01.
2. The capped previous-gap gain (M1 over M0) is smaller than round 2's 0.0043, since the wheel mask
   absorbs part of what the previous gap was proxying.
3. M1r (residue-based previous gap) beats M1 by >= 0.001: the repulsion lives in residue structure.
4. M2 does NOT beat the best lag-1 model by >= 0.001 at z >= 2: no useful memory past one gap.
5. HMM K=2 collapses to K=1 again (held-out within 0.0005 of M0).
6. Rolling folds: sign of the best lag-1 model minus M0 is positive in >= 4 of 5 folds.

## Empirical Check — Round 3 (Wheel Constraint, Longer Memory, Folds)

**Date checked:** 2026-10-26. `output/prime/20261026_173636/gap_hmm_wheel/results.json`.
19,995 gaps (p >= 11), train 15,996 / test 3,999. Held-out nats/gap:

| model | params | train BIC/gap | test nats/gap | vs round-2 S0f | vs M0 (z) |
|---|---|---|---|---|---|
| S0f (round 2, no wheel) | 1 | 5.0961 | -2.66350 | 0 | -0.60623 |
| M0 (W=30) | 1 | 3.8566 | -2.05727 | +0.60623 | 0 |
| M0_210 (W=210) | 1 | 3.4818 | -1.87269 | +0.79081 | +0.18457 (z 75) |
| M1 (capped prev gap, 5 cls) | 5 | 3.8558 | -2.05468 | +0.60882 | +0.00258 (z 2.87) |
| M1r (prev k mod 15) | 15 | 3.8609 | -2.05569 | +0.60781 | +0.00158 (z 1.53) |
| M2 (capped prev two gaps, 25 cls) | 25 | 3.8661 | -2.05519 | +0.60831 | +0.00208 (z 1.82) |
| HMM K=2 (on M0 emission) | 4 | 3.8584 | -2.05727 | +0.60623 | 0.00000 |

**Against the locked prediction:**
1. **Held, hugely.** The wheel mask is worth +0.606 nats/gap over round 2's best force (vs the >= 0.05
   predicted) and W=210 adds a further +0.185 (vs >= 0.01). Caveat that matters: this is arithmetic,
   not discovered structure. "p+g cannot be divisible by 3, 5, 7" is a deterministic sieve constraint.
   Pushing the wheel up only moves toward exact knowledge of the next prime. It is evidence that the
   predictive scorecard rewards known number theory, not evidence about hidden structure.
2. **Held.** Capped previous-gap gain shrank from round 2's 0.0043 to 0.0026 (z 2.9): the wheel mask
   absorbed roughly 40% of it.
3. **FAILED.** The residue-based previous-gap context (M1r) did NOT beat the capped one: -0.0010 (z -1.46).
   The 15 classes are just noisier. The LOS effect I expected to find "in the residue of the previous
   gap" is not better captured that way, in this model.
4. **Held.** M2 - M1 = -0.0005 (z -0.75); M2 - M1r = +0.0005. No usable memory past one gap.
5. **Held.** HMM K=2 collapses: m = 0.756 / 0.759, held-out identical to M0 to five decimals. Third
   time a latent-state model has collapsed to one state.
6. **Held.** M1 beats M0 in 5/5 rolling folds, mean +0.00144 nats/gap. But the fold deltas are
   +0.00002, +0.00072, +0.00124, +0.00268, +0.00256: small and noisy early, larger later.

**Exploratory (not pre-registered), same day:** repeating the M1-vs-M0 fold test with W=210 and with
W=2310 (3,5,7,11) gives essentially the same picture (f=0.5: -0.00003 / -0.00014; f=0.8: +0.00296 /
+0.00264; f=0.9: +0.00212 / +0.00202; f=0.8 z = 2.3). The previous-gap effect does not shrink under a
bigger wheel, so it is not a mod-3/5 artifact. Whether the apparent growth with position (flat near
p ~ 80-110k, ~+0.003 near p ~ 180-225k) is real non-stationarity or noise cannot be told from five
2,000-gap folds (each fold SE ~ 0.001). Flagged for a proper test, not claimed.

**Net after three rounds:** (a) the force is ln p plus the singular series plus the sieve constraint;
(b) a hidden latent state is refuted in three forms (geometric, singular-series, wheel-masked);
(c) the one residual structure is a small previous-gap scale effect (~0.002-0.003 nats/gap, z ~ 2-3
per split, 5/5 folds same sign) that survives bigger wheels but has no support from lag 2.
