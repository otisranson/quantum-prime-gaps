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
