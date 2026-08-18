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

## Empirical Check — Regime Overlay (the originally-planned Layer 3 test)

**Date checked:** August 17, 2026

**Method:** This is the direct test flagged as still-unrun above — extract the raw gap sequence within each of the three regimes actually bounded by the confirmed changepoints (windows 1529, 2501, 4211), normalize each for length and amplitude, and check whether they rhyme. See `layer3_regime_overlay.py` and `output/prime/analysis/{layer3_regime_overlay.png,layer3_regime_overlay.json}`.

**Regime definition, stated explicitly:** three changepoints delimit exactly three segments that have *both* a start and end changepoint (using the sequence start as the first regime's implicit open end): `[0, 1529)` (length 1529), `[1529, 2501)` (length 972), `[2501, 4211)` (length 1710). The tail after the last changepoint, `[4211, 4999)` (length 788), is excluded — it has no closing changepoint, so by the same "bounded by the changepoints" framing used everywhere else in this file, it isn't one of the three regimes under test. This is one reasonable reading, not the only possible one — see caveats.

**Normalization:** each regime resampled (linear interpolation) to a common length of 500 points, then z-scored (mean 0, std 1) — so the overlay compares *shape*, not absolute gap magnitude or duration. Raw-normalized curves were overlaid first; then each was independently linear-detrended (its own best-fit line removed) and the residuals overlaid separately.

**Similarity score:** mean of the three pairwise Pearson correlations between regime curves.

- Raw normalized: **-0.0582** (pairwise: -0.0688, -0.0738, -0.0321)
- Detrended residuals: **-0.0570** (pairwise: -0.0700, -0.0689, -0.0321)

**Random-slice null:** 5,000 trials (seed=42), each drawing three random slices from the full 4999-gap sequence matching the real regimes' lengths (1529, 972, 1710) at independent random start positions, normalized and scored the same way.

- Raw null: mean=0.0019, std=0.0257 → observed similarity (-0.0582) sits at the **0.8th percentile**
- Detrended null: mean≈0.0000, std=0.0257 → observed similarity (-0.0570) sits at the **1.0th percentile**

**Result, stated clearly: REFUTED — and in the opposite direction than predicted.** The hypothesis predicted the three regimes should rhyme (similarity *higher* than random chance). Instead, the three regimes are *less* alike than 99%+ of random same-length triplets drawn from the same sequence, in both the raw-normalized and the detrended versions. The null's own mean sitting near zero is exactly what's expected if regimes were simply independent/unrelated (no relationship either way); the actual regimes land significantly *below* that zero-relationship baseline, in the bottom ~1% tail. That the raw and detrended results are nearly identical (-0.058 vs -0.057, both in the same extreme tail) indicates this isn't an artifact of differing linear trends across regimes being picked up as spurious correlation/anti-correlation — removing each regime's own trend didn't change the picture, so the divergence is in the shapes themselves, not just their slopes.

**Mismatch vs. regime index:** per-regime RMS distance of the detrended residual from the cross-regime mean residual curve: regime 0 = 0.8397, regime 1 = 0.8322, regime 2 = 0.8315. Correlation of this mismatch with regime index (0, 1, 2): **-0.9015**. **State this precisely: this number is not meaningful and should not be read as "mismatch decreases with regime index."** The three underlying values differ by about 1% of their own magnitude (0.8315–0.8397) — essentially flat — and any three points that are nearly flat with tiny monotonic noise will produce a large-magnitude Pearson r by construction, since n=3 gives the correlation almost no room to land anywhere but near ±1. This is reported because it was asked for, not because it's evidence of a real trend.

**Caveats:**
1. **Regime-definition choice.** Excluding the post-4211 tail (and using sequence-start as regime 0's open boundary) is one defensible reading of "the three regimes bounded by the changepoints," not the only one — a version that includes the tail as a fourth regime, or that uses the window-index space directly instead of mapping window index onto raw gap index 1:1 (an approximation used identically in every prior script in this file, not re-derived here), could give a different similarity score. Worth rerunning with an alternate regime definition before treating "refuted" as final.
2. **Only linear detrending was tried.** A regime with real curvature (quadratic or higher-order drift) would still carry that shape into the "residual." That the raw and detrended results agree so closely suggests linear trend isn't the main driver either way, but higher-order detrending wasn't tested.
3. **Null slices can overlap each other, the real regimes, or the excluded tail**, and are drawn independently rather than as a non-overlapping partition of the sequence — a reasonable and simple baseline for "how similar are three arbitrary same-length chunks of this data," but not the only possible null design.
4. **This result is surprising enough to flag rather than just accept at face value:** significant anti-similarity (not just "no similarity") between the three regimes could reflect something structurally real about how the changepoint-detection method itself defines boundaries (e.g., changepoints might systematically fall where the *character* of the local sequence is shifting away from its neighbor, which is close to a restatement of what a changepoint is — worth being careful not to treat this as an independent confirmation of anything). It could also reflect the specific detrending/normalization choices above. Not resolved here.

## Status

**Updated August 17, 2026, after both empirical checks above.** Internal-wave / self-affinity hypothesis ("regimes rhyme") has two independent negative results now: the MI-landscape probe (inconclusive, no consistent alignment) and the direct regime-overlay test (refuted, with the regimes significantly *less* alike than chance — bottom ~1st percentile against a random-slice null, both raw and detrended). No test run so far supports "regimes rhyme." The anti-similarity finding in the overlay test is unexplained and flagged as worth a closer look, not as confirmation of any alternative claim.

## Follow-up — Per-Regime Characterization (not a retest of "rhyme")

**Date:** August 17, 2026 (system clock reports August 18, 2026 UTC for the run timestamp/commit below — same session).

**Framing, stated up front:** the Regime Overlay check above refuted cross-regime shape matching — the three regimes are significantly *less* alike than random chance (bottom ~1st percentile, both raw and detrended). This follow-up does not retest that claim and does not attempt any cross-regime comparison, similarity score, or null test. Given the regimes don't rhyme, the natural next question is simply what each one actually looks like on its own terms. Each regime is treated as an independent object here. See `layer3_regime_characterization.py` and `output/prime/20260818_010500/{layer3_characterization_panels.png,layer3_characterization_table.png,results.json}`. Same regime definitions as the refutation test: `[0,1529)`, `[1529,2501)`, `[2501,4211)` in raw gap-index space, post-4211 tail excluded.

**Summary table (all metrics computed independently per regime, no shared normalization):**

| metric | regime 0 (n=1529) | regime 1 (n=972) | regime 2 (n=1710) |
|---|---|---|---|
| spike density (\|z\|>2, own mean/std) | 0.0517 (79 spikes) | 0.0453 (44 spikes) | 0.0427 (73 spikes) |
| volatility mean (rolling std, K=100) | 6.0242 | 7.2701 | 7.8669 |
| volatility trend (slope, own fractional position) | +2.6685 (increasing) | +0.6477 (increasing) | +1.2020 (increasing) |
| FFT top period (gap-steps/cycle) | 1529.0 (see note) | 2.298 | 5.917 |
| FFT 2nd/3rd period | 2.092 / 2.795 | 4.320 / 2.641 | 2.595 / 2.339 |
| skew | 1.4960 | 1.5719 | 1.8864 |
| excess kurtosis | 2.4216 | 3.2232 | 6.0411 |
| mean gap | 8.3891 | 9.7942 | 10.3801 |
| variance | 37.1677 | 53.7848 | 62.7304 |

**Observations, stated descriptively (no significance claims — none were tested here by design):**

- **Mean and variance climb monotonically with regime index** (8.39→9.79→10.38 mean; 37.2→53.8→62.7 variance). This tracks the Prime Number Theorem's expected average-gap growth with N and isn't itself a new finding — flagged for completeness, not as a discovery.
- **All three regimes are right-skewed with positive excess kurtosis** (heavier-tailed than normal), and both skew and kurtosis increase with regime index (kurtosis 2.42→3.22→6.04 — regime 2 is markedly heavier-tailed). Read alongside the next point, this says the *size* of the largest outliers is growing faster than their *count*.
- **Spike density (fixed at 2 own-sigma) is flat-to-slightly-declining across regimes** (0.052→0.045→0.043) even as kurtosis roughly triples. Since the z-score threshold is regime-relative, this isn't a contradiction — it means each regime has a similar *rate* of 2-sigma-plus events, but later regimes' extreme events are more extreme relative to their own spread, not more frequent.
- **Volatility (rolling std) increases within every regime** (all three trend slopes positive), but regime 0's internal ramp (+2.67) is roughly 2–4x steeper than regime 1 (+0.65) or regime 2 (+1.20) — regime 0 goes from calm to volatile faster, internally, than the later two.
- **FFT dominant period — caveat before the number:** the FFT here only removes each regime's mean, not its trend, before the periodogram. Regime 0's top peak landing at period=1529.0 — exactly its own length — is very likely an artifact of its own steep volatility/trend ramp (see above) dumping power into the lowest-frequency bin, not a genuine full-regime cycle; its 2nd and 3rd peaks (period ≈2.09, 2.80) are the more informative ones. Regimes 1 and 2 don't show this artifact — their top peaks are all short (2.3–5.9 gap-steps), consistent with the well-known short-range alternation/parity structure general to prime gap sequences (not specific to these regimes or this circuit). None of the three regimes show a long-period genuine oscillation once the trend artifact in regime 0 is discounted.

**Caveats:**
1. **Purely descriptive by design** — no null distributions, no significance tests, no p-values anywhere in this follow-up; none of the numbers above should be read as "significant" or "confirmed" in the sense the refutation test's percentiles were. If any of these differences (e.g. the volatility-trend-slope gap between regime 0 and the others) look interesting enough to chase, that would need its own dedicated null test, not an extension of this one.
2. **FFT trend artifact** (discussed above for regime 0) means "dominant period" as reported is not directly comparable across regimes without separately checking whether each regime's top peak is itself a trend artifact — only regime 0 showed this pattern here, but the check should be repeated if this table gets used for anything beyond description.
3. **Spike/volatility/skew/kurtosis are all computed on each regime's own raw values**, not on any shared or resampled scale — by construction they cannot be used to reproduce a cross-regime similarity claim; that's the point of this follow-up, but worth restating so a future session doesn't quietly repurpose this table as a comparison.
4. **Same regime-definition scope caveat as the refutation test applies here too** — the post-4211 tail is excluded, and window-index-to-gap-index mapping is the same 1:1 approximation used throughout this file.
