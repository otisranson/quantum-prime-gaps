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

## Kurtosis Robustness Check

**Date:** August 17, 2026 (system clock reports August 18, 2026 UTC for the run timestamp/commit below — same session).

**Framing:** the Characterization follow-up above found excess kurtosis climbing across the three regimes (2.42 → 3.22 → 6.04) while 2-sigma spike density stayed roughly flat-to-declining (0.0517 → 0.0453 → 0.0427) — outlier gaps aren't more frequent later, but they're more extreme when they occur. This check tests whether that pattern is robust or an artifact, along five axes, none of which retest cross-regime similarity (that stays refuted, see above). See `layer3_kurtosis_robustness.py` and `output/prime/20260818_012118/{layer3_kurtosis_robustness_overview.png,layer3_kurtosis_sliding_window.png,results.json}`. Same regime definitions throughout: `[0,1529)`, `[1529,2501)`, `[2501,4211)`.

**Verdict up front, since the five checks disagree with each other: MIXED.** Two axes support the original finding being real and not an artifact; one axis (the CIs) says the point-estimate ordering is statistically weaker than it first looked; one axis (sliding window) gives partial, not clean, support. Detail below.

**1. Bootstrap CIs (n=2000 resamples, seed=42, 95% percentile CI):**

| | kurtosis (point) | kurtosis 95% CI | spike density (point) | spike density 95% CI |
|---|---|---|---|---|
| regime 0 | 2.4216 | [1.8052, 3.0995] | 0.0517 | [0.0438, 0.0641] |
| regime 1 | 3.2232 | [1.7260, 4.9057] | 0.0453 | [0.0381, 0.0607] |
| regime 2 | 6.0411 | [3.0358, 9.2943] | 0.0427 | [0.0380, 0.0550] |

**All three regimes' CIs overlap pairwise, for both kurtosis and spike density** (0v1, 1v2, and 0v2 all overlap = True). **This is the one result that pushes back on the original finding**: given sampling uncertainty at these regime sizes (n=1529, 972, 1710), the point-estimate ordering (rising kurtosis, declining density) is not statistically well-separated — a single bootstrap resample could plausibly reorder any adjacent pair. The 0-vs-2 kurtosis comparison is the closest call (regime 0's upper bound 3.0995 vs. regime 2's lower bound 3.0358 — overlapping by only 0.0637, the narrowest margin of the three pairs), so the endpoints of the progression are the least implausible to be genuinely different, but "closest to non-overlapping" is not the same as non-overlapping.

**2. Threshold robustness (spike density at 2.0 / 2.5 / 3.0 sigma):**

| sigma | regime 0 | regime 1 | regime 2 |
|---|---|---|---|
| 2.0 | 0.0517 | 0.0453 | 0.0427 |
| 2.5 | 0.0360 | 0.0309 | 0.0193 |
| 3.0 | 0.0190 | 0.0154 | 0.0146 |

**Holds at all three thresholds, not just 2-sigma** — spike density is monotonically non-increasing from regime 0 to regime 2 at every threshold tested (and the decline actually gets *proportionally* sharper at higher thresholds, e.g. regime 2's density roughly halves going 2.0→2.5 sigma while regime 0's drops less). This is evidence *for* the original finding being real rather than a threshold-specific artifact.

**3. Sliding-window kurtosis (width=500, step=100, across the full [0, 4211) range, regime boundaries ignored):** 38 windows, kurtosis ranging [1.5762, 9.2746]. Step-to-step jump size at each changepoint's position, compared to the median absolute step size everywhere else in the same series (0.3601):

| changepoint | jump at this position | median \|step\| elsewhere | ratio |
|---|---|---|---|
| 1529 | 0.2015 | 0.3601 | 0.56x (**below** the typical step) |
| 2501 | 1.7623 | 0.3601 | 4.89x |
| 4211 | 1.5519 | 0.3601 | 4.31x |

**Mixed, not clean confirmation either way.** If kurtosis rose as one smooth continuous drift unrelated to the changepoints, none of the three positions should show an unusual jump. Two of three changepoints (2501, 4211) show jumps 4–5x larger than the typical local step — consistent with a real, localized, regime-specific effect at those two boundaries. But the first changepoint (1529) shows *no* unusual jump at all (actually below the median step size elsewhere) — inconsistent with "kurtosis jumps specifically at changepoints" as a universal rule. Stated plainly: this test supports a regime-specific effect at two of the three boundaries and gives no support at the third.

**4. Background-growth control (local log-fit detrend, `gap ~ a·ln(global_index+2)+b`, fit separately per regime, then kurtosis recomputed on the gap/trend ratio):**

| | fit a | fit b | original kurtosis | detrended kurtosis |
|---|---|---|---|---|
| regime 0 | 1.1239 | 1.2634 | 2.4216 | 2.1020 (−13.2%) |
| regime 1 | 0.6327 | 4.9860 | 3.2232 | 3.1916 (−1.0%) |
| regime 2 | 1.0011 | 2.2630 | 6.0411 | 5.9440 (−1.6%) |

**Survives detrending — the kurtosis rise is not explained by ordinary gap-size growth alone.** Dividing out each regime's own local logarithmic growth trend barely moves kurtosis for regimes 1 and 2 (~1–2% change) and only a modest 13% drop for regime 0; the 2.10 → 3.19 → 5.94 progression after detrending is essentially the same shape as the original 2.42 → 3.22 → 6.04. This is evidence *for* the original finding being a real distributional-shape effect, not an artifact of typical gap size simply growing with N.

**5/6. Max and mean gap size per regime, for growth context:**

| | max gap | mean gap | max/mean ratio |
|---|---|---|---|
| regime 0 | 36.0 | 8.3891 | 4.29x |
| regime 1 | 52.0 | 9.7942 | 5.31x |
| regime 2 | 72.0 | 10.3801 | 6.93x |

Max gap size grows 36 → 52 → 72 across regimes (close to, not exactly, the 35/50/~70 recalled in this session's prompt). The max/mean ratio itself climbs (4.29x → 5.31x → 6.93x) — another way of restating the kurtosis finding: the largest gaps are pulling further away from the typical gap, relative to the typical gap, in later regimes.

**Net read:** two of five checks (threshold robustness, background-growth control) support the original "rising kurtosis, flat-to-declining spike density" finding as real. One check (bootstrap CIs) says the point-estimate ordering isn't statistically well-separated at these sample sizes — treat "kurtosis rises with regime index" as a real descriptive pattern in this data, not (yet) a statistically confirmed one. One check (sliding window) gives partial support — localized jumps at 2 of 3 changepoints, no jump at the third. This is not a clean confirm or refute; both directions of caution belong in any future claim built on the original finding.

**Caveats:**
1. **Bootstrap here is a standard nonparametric bootstrap treating each regime's gap values as exchangeable/iid** — it resamples values, not blocks of the sequence, so it doesn't account for whatever serial correlation exists along the gap sequence. If gaps are autocorrelated within a regime, this bootstrap's CIs could be too narrow (understating uncertainty) or too wide, in either direction — not verified here.
2. **The sliding-window jump-ratio test uses a single nearest-step comparison per changepoint**, not a proper null distribution over where "unusual" jumps occur throughout the full 38-window series — a rigorous version would ask "how often does *any* randomly chosen position show a jump this large," not just eyeball the one ratio number at each of the three known positions. Not built here; flagged as the natural next step if this specific finding gets pursued further.
3. **The log-fit detrend's functional form (`a·ln(global_index+2)+b`) is one reasonable choice**, matching this file's own Layer 1 "regime spacing ~ ln(N)" framing, but wasn't compared against other plausible growth models (e.g. a power law) — the "survives detrending" conclusion is conditional on this specific detrending choice.
4. **Same regime-definition and scope caveats as every other check in this file** — post-4211 tail excluded, window-index-to-gap-index mapping is the same 1:1 approximation used throughout.

## Changepoint Character Comparison

**Date:** August 17, 2026 (system clock reports August 18, 2026 UTC for the run timestamp/commit below — same session).

**Framing:** the Kurtosis Robustness Check above found the sliding-window kurtosis test jumps sharply at changepoints 2501 and 4211 (4.9x and 4.3x the typical local step) but shows no unusual jump at 1529 (0.56x, below median). This asks what distinguishes 1529 from the other two: a weaker version of the same kind of transition, or a fundamentally different kind of changepoint. See `layer3_changepoint_1529_investigation.py` and `output/prime/20260818_013309/{layer3_changepoint_comparison.png,layer3_changepoint_breakdown_table.png,results.json}`.

**1. Original detection evidence** (re-running the exact binary-segmentation detector `regime_fit_5k.py` originally used — least-squares mean-shift cost on the MI rolling mean, K=100 — reproduced the same three changepoints `[1529, 2501, 4211]` exactly, confirmed by an in-script assertion):

| changepoint | binary-seg gain | MI level-shift magnitude | envelope-fit residual (window = a·ln(k)+b) |
|---|---|---|---|
| 1529 | 0.02716 (**smallest**) | 0.01085 | +173.2 |
| 2501 | 0.04529 | 0.00855 (**smallest**) | −469.4 (**largest magnitude**) |
| 4211 | 0.07179 (largest) | 0.01166 (largest) | +296.1 |

**Mixed — 1529 is not cleanly "the weak one" by the original evidence.** By binary-segmentation gain (the direct cost-reduction the detector itself optimizes), 1529 is indeed the weakest of the three, consistent with the sliding-window kurtosis result. But by the other two original-evidence measures, 2501 is the outlier instead: 2501 has the *smallest* MI level-shift magnitude (smaller than 1529's) and by far the worst envelope-fit residual (−469.4, roughly 1.6–2.7x larger in magnitude than the other two). So "1529 was always the weakest by the original criteria" is not accurate — it's weakest by one specific measure (gain) and not by two others.

**2. Level / scale / shape breakdown** (before vs. after, window=300 gaps each side, bootstrap 95% CI, n_boot=2000, seed=42):

| changepoint | level (mean) | scale (variance) | shape (kurtosis) |
|---|---|---|---|
| 1529 | 9.4067→9.5000, no sig. shift | 45.26→44.40, no sig. shift | 1.2617→2.1013, no sig. shift |
| 2501 | 9.7333→10.2067, no sig. shift | 55.00→51.94, no sig. shift | 4.9671→2.1609, no sig. shift |
| 4211 | 10.6667→10.3600, no sig. shift | 64.46→62.62, no sig. shift | 2.8336→6.0628, no sig. shift |

**No significant shift in level, scale, or shape at any of the three changepoints, at this local scale — including at 2501 and 4211**, which is worth flagging rather than quietly reconciling: it does **not** contradict the Kurtosis Robustness Check's finding of real jumps at 2501/4211, it's a power problem specific to this narrower test. A ±300-gap window is a much smaller sample than the ~970–1710-point full regimes the earlier bootstrap CIs were built on, and kurtosis in particular has high sampling variance at n=300 (e.g. 2501's "before" shape CI spans [1.59, 8.73] — wide enough to swallow almost any plausible after-value). This test, as built, cannot detect the effect the coarser regime-scale and sliding-window tests already found — it isn't evidence against that effect, just an underpowered instrument for it at this local scale.

**3. Fine-grained local kurtosis scan** (width=150, step=25, across ±300 gaps around each changepoint):

| changepoint | local kurtosis range | peak offset | peak kurtosis |
|---|---|---|---|
| 1529 | [0.573, 3.180] | **+0** (right at the changepoint) | 3.180 |
| 2501 | [1.031, 7.294] | −225 (before the changepoint) | 7.294 |
| 4211 | [1.456, 7.272] | +75 (just after the changepoint) | 7.272 |

**This is the most direct answer to the question this script asks.** 1529 is not "nothing" — it has a genuine local kurtosis peak, and that peak sits exactly at offset 0, right at the changepoint itself, same qualitative signature as the other two. What's different is magnitude: 1529's peak (3.18) is roughly **2.3x smaller** than 2501's (7.29) or 4211's (7.27), which are close to each other. There's no sign of a delayed or offset transition near 1529 that the coarse regime cut missed — the local structure that exists is centered right where the changepoint already is, just weaker.

**Answer, stated as directly as the evidence allows: 1529 looks like a weaker version of the same kind of transition, not a fundamentally different kind of changepoint.** It shows the same signature (a local kurtosis peak located at the changepoint) as 2501 and 4211, just proportionally smaller (~2.3x), and the "weakest of the three" framing from the sliding-window test only holds up cleanly on one of the three original-detection measures (gain) — on the other two (MI level-shift, envelope-fit residual), 2501 is actually the more unusual point. This is a real distinction in degree, not evidence of two categorically different mechanisms — but it's a moderate-confidence read, not a proven one; see caveats.

**Caveats:**
1. **The level/scale/shape breakdown (point 2) was underpowered by construction** at window=300 — it should not be read as "no shift exists at 2501/4211," only as "this specific local test at this specific window size can't detect the shift the other tests found." A proper resolution would sweep window size (e.g. 300, 500, 750, 1000) to find where the bootstrap CIs start separating, rather than picking one size and reporting a null.
2. **"Weakest by gain" being the one original criterion that agrees with the newer kurtosis result is itself worth treating cautiously** — gain is the binary-segmentation detector's own optimization target on the MI rolling mean, a different signal entirely from the raw-gap kurtosis tested later. That the two happen to agree on 1529 could be a real shared cause or could be coincidence at n=3; not resolved here.
3. **The fine local scan's "peak offset" is a single argmax over a fairly coarse fine-grained series** (19 points per changepoint, width=150/step=25) — a true peak location and its uncertainty weren't estimated with a CI; small perturbations to width/step could shift the reported offset by one or two steps (25–50 gaps) without changing the qualitative conclusion (1529's peak is near-zero-offset and smaller than the other two).
4. **Same regime-definition and scope caveats as every other check in this file apply** — window-index-to-gap-index mapping is the same 1:1 approximation used throughout, and no cross-regime similarity claim is made or retested here.

## 20k Scale-Up: Intensity vs. Position

**Date:** August 17, 2026 (system clock reports August 18, 2026 UTC for the run timestamp/commit below — same session).

**Framing:** the Changepoint Character Comparison above found peak kurtosis intensity climbing 3.18 → 7.29 → 7.27 across the three known changepoints (1529, 2501, 4211) — but n=3 has essentially no statistical power to say whether intensity really increases with position. This scales the same measurement up 4x (20,000 primes instead of 5,000) to get enough changepoints for a real test. See `layer3_20k_scaleup.py` and `output/prime/20260818_015045/{layer3_20k_intensity_vs_position.png,layer3_20k_changepoint_table.png,results.json}`.

**Part A prerequisite (data refactor, done first as instructed):** this repo had no standalone primes/gaps cache file before this session. `terrain_5000primes.py` generated primes inline via its own sieve each run, and eight classical analysis scripts each separately reconstructed the raw gap sequence from `output/prime/20260816_010716/terrain_5000primes/results_5000primes.json`'s per-window data via the same repeated boilerplate. `build_prime_cache.py` now generates `data/primes_5000.json` (independently sieved, verified byte-identical to that reconstruction via an in-script assertion) and the new `data/primes_20000.json` (20,000 primes, 19,999 gaps). All eight scripts (`smoothed_derivative_wave.py`, `layer2_magnitude_test.py`, `gap_derivative_zero_crossings.py`, `smoothed_gap_derivative_zero_crossings.py`, `layer3_regime_overlay.py`, `layer3_regime_characterization.py`, `layer3_kurtosis_robustness.py`, `layer3_changepoint_1529_investigation.py`) were refactored to read from the cache; the four that don't auto-commit were rerun post-refactor and printed numbers matched their previously recorded values exactly (e.g. layer2_magnitude_test.py's percentiles 38.3/79.9/74.6, gap_derivative_zero_crossings.py's distances 2/0/1). `mi_landscape_25groups.py` (independent first-100-gap circuit encoding, with its own trial-division correctness check already built in) was intentionally left alone — it isn't regenerating "the 5k sequence," just a small subset for a different, already-justified reason. Cache sizes: `data/primes_5000.json` — 5000 primes, 4999 gaps, 49.6 KB; `data/primes_20000.json` — 20000 primes, 19999 gaps, 213.9 KB.

**Method, Part B:**
1. **Changepoint detection at 20k scale.** No 20,000-prime quantum circuit run exists (that would require actually running the terrain-style circuit 20,000 primes deep, not built here) — so unlike the original `regime_fit_5k.py`, which ran binary segmentation on quantum-measured MI, this runs the *same* least-squares mean-shift binary-segmentation cost directly on the K=100 rolling mean of the raw gap sequence. This is stated as a real methodological difference, not a detail: the original 3 changepoints were found in MI-space; these 39 are found in gap-space. They are not necessarily measuring the same thing, even though the algorithm is identical.
2. **Vectorized re-implementation, verified equivalent.** The original nested-loop `best_single_split` is O(n²) — fine at ~4900 points, far too slow at ~19900. A cumulative-sum vectorized O(n) version was written and checked against the naive O(n²) reference on a synthetic mean-shift series before being trusted (`verify_vectorized_matches_naive`, confirmed identical gain and split position).
3. **Data-driven stopping rule, not a fixed count.** Rather than assuming a number of breakpoints, each candidate split must beat the 99th percentile of a 200-trial within-segment permutation null (shuffle the segment being split, rerun the split-finder, many times) before being accepted; the search stops at the first candidate that doesn't clear its own null. `min_size=400`, chosen to match the original's `min_size=100` as a fraction of series length (~2%).
4. **Fine-grained local kurtosis scan** (identical to the 1529/2501/4211 investigation: width=150, step=25, ±300 gaps) run at every accepted changepoint, extracting peak intensity and its offset.
5. **Correlation test**, permutation-based (5,000 shuffles of intensity against position) rather than an assumed-normal p-value.

**Result: 39 changepoints found** (raw gap-index positions 499 through 19395), well more than the original 3, as expected at this scale. Peak kurtosis intensity across all 39: min=1.97, max=15.87, mean=6.94, median=6.73 — a real, non-degenerate spread.

**Position vs. peak-intensity correlation: r = 0.112, permutation p = 0.493 (two-tailed, n_perm=5000).** **This refutes the ordering hypothesis, with real statistical power this time.** The null distribution of r (shuffling intensity against position) has mean≈0.000, std=0.162 — the observed r=0.112 sits comfortably inside that null, not anywhere near its tails. At n=39, this is not an underpowered null result the way the n=3 case was structurally incapable of failing — a real, strong trend (e.g. |r|>0.4) would have been clearly detectable at this sample size and was not found.

**What this means for the original n=3 observation:** the 3.18 → 7.29 → 7.27 climb across 1529/2501/4211 does not generalize. It's consistent with what a correlation of ~0.11 against noise of std 0.16 would produce by chance in a lucky 3-point subsample — three points drawn from a population with no real position-intensity relationship can easily look like they're "climbing" (especially since 7.29 and 7.27 are themselves nearly equal, so the appearance of a climb was really driven by 1529 alone being lower, not by a sustained trend across three points). This is a clean, informative negative result, not a failure to detect something real.

**Caveats:**
1. **The changepoint-detection *signal* differs between the original 3 and these 39** (quantum MI vs. raw-gap rolling mean, stated above) — the 39 changepoints found here are not on equal footing with 1529/2501/4211 as "the same kind of thing measured the same way." The refutation of the ordering hypothesis stands on its own terms (kurtosis intensity vs. position, tested directly on this dataset), but it is not a like-for-like extension of the original MI-based detection.
2. **The stopping rule (99th percentile of a within-segment permutation null) is a reasonable, principled choice but not a published/standard algorithm**, and it is not corrected for the multiple comparisons implicit in testing many candidate splits across the recursive search — some fraction of the 39 accepted changepoints could still be false positives at the individual-test level even though each one individually cleared its own null.
3. **`min_size=400` was chosen to match the original's proportional minimum segment size (~2% of series length)**, not re-derived or swept — a different `min_size` could change how many changepoints are found (more may appear with a smaller min_size, fewer with a larger one), though the position-vs-intensity null result is unlikely to be an artifact of this specific choice given how cleanly it lands inside the permutation null.
4. **The vectorized `best_single_split` was checked against the naive reference on one synthetic series, not the actual 20k data itself** (checking on the real 20k data at the naive O(n²) implementation's own scale was avoided for being too slow — the whole point of vectorizing). The synthetic check confirms the two formulas are algebraically equivalent; it doesn't independently verify the specific gains reported at 20k scale beyond that equivalence.

## 40-Regime Characterization

**Date:** August 17, 2026 (system clock reports August 18, 2026 UTC for the run timestamp/commit below — same session).

**Framing:** the original Per-Regime Characterization follow-up (above) described 3 regimes on their own terms — no statistical power to say whether any of its stats (mean, variance, skew, kurtosis) actually trend with position versus just looking like they do. This reuses the exact 39 changepoints from the 20k Scale-Up run (not re-detected) to carve the full 20,000-prime gap sequence into 40 regimes and runs the same characterization at real sample size, testing every trend with the same permutation-correlation method used for the position-vs-intensity result rather than eyeballing. See `layer3_regime_characterization_20k.py` and `output/prime/20260818_020038/{layer3_40regime_trends.png,layer3_40regime_table.png,results.json}`.

**Regime definition:** 39 changepoints (positions 499 through 19395, from `output/prime/20260818_015045/results.json`) split the full 19,999-gap sequence into 40 regimes — unlike the original 3-regime work, which excluded the tail after its last changepoint (that changepoint set only had a closing boundary for 3 of what would have been 4 segments). Here, with 39 changepoints, all 40 segments are used, including the leading segment before the first changepoint and the trailing segment after the last one. Regime lengths: min=400 (exactly `MIN_SIZE` from the detection run — expected, since the detector itself enforces that floor), max=772, mean=500.0.

**Methodology, same as the original 3-regime script, with one deliberate change:** the original found regime 0's top FFT peak sitting at period == its own length, diagnosed afterward as an artifact of only mean-subtracting (not detrending) before the FFT on a regime with a strong internal trend. This script removes each regime's own best-fit line (not just its mean) before every FFT, prospectively, rather than repeating that artifact 40 times over.

**Low-confidence flagging:** regimes below `MIN_CONFIDENT_SIZE=100` (a judgment-call threshold, not derived — 4th-moment statistics like kurtosis need a reasonable sample) would be flagged rather than silently included. **None of the 40 regimes were flagged** — the smallest is 400, well above the threshold, because the upstream changepoint detector's own `min_size=400` already enforces a floor larger than this script's confidence threshold. The flagging logic exists and was exercised (checked every regime), it simply found nothing to flag this run.

**Permutation correlation tests (position vs. stat, n=40, 5,000-trial permutation null each):**

| stat | observed r | null (mean, std) | p-value (two-tailed) | verdict |
|---|---|---|---|---|
| mean gap | **+0.8505** | (0.0002, 0.1617) | **<0.0001** | **SIGNIFICANT** |
| variance | **+0.8756** | (0.0016, 0.1633) | **<0.0001** | **SIGNIFICANT** |
| skew | +0.1406 | (−0.0002, 0.1610) | 0.3848 | not significant |
| excess kurtosis | +0.0936 | (0.0003, 0.1597) | 0.5714 | not significant |

**Result, stated clearly: mean and variance climb with position — real and strongly significant, but not a new finding (this is the Prime Number Theorem's expected average-gap growth, already noted descriptively in the 3-regime work and now confirmed with actual statistical power rather than 3 anecdotal points). Skew and kurtosis, the more interesting candidates for "does the internal-wave hypothesis show structure," do NOT trend with position at n=40 — both land well inside their null distributions, not anywhere near significance.**

This is consistent with, and reinforces, the 20k Scale-Up's finding immediately above: peak boundary-kurtosis intensity showed no relationship with position (r=0.112, p=0.493) there; here, each regime's *own internal* excess kurtosis (a related but distinct quantity — computed over the whole regime, not a local ±300-gap window at its boundary) also shows no relationship with position (r=0.094, p=0.571). Two different kurtosis measurements, tested two different ways, on two overlapping but not identical samples, both come back null. The original n=3 observation (kurtosis climbing 2.42 → 3.22 → 6.04 across the three 5000-prime-run regimes) does not generalize to the 40-regime, 20,000-prime scale-up by either measure.

**Caveats:**
1. **Mean and variance climbing with position is expected under the Prime Number Theorem** (average gap near N grows ~ln(N)) — its statistical significance here confirms the null hypothesis of "no relationship" is correctly rejectable by this test design (a useful sanity check that the permutation-test machinery works and has power), not a novel discovery about regime structure.
2. **"Boundary kurtosis" per regime (in the full table) is the peak intensity of the changepoint that *starts* that regime** (regime i+1 gets changepoint i's value; regime 0 has none) — one specific, stated convention for a many-to-one relationship (each regime has two possible bounding changepoints, start and end, except the first and last); an end-boundary convention was not also computed or compared.
3. **All 40 regimes happened to clear `MIN_CONFIDENT_SIZE=100` only because the upstream detector's `min_size=400` already exceeds it** — this doesn't validate 100 as a universally sufficient threshold for stable skew/kurtosis estimation in general, only that it wasn't binding here.
4. **Same cross-scale caveat as the 20k Scale-Up above**: these 40 regimes come from gap-space changepoint detection (rolling-mean binary segmentation), not the original quantum-MI-space detection used for the 3-regime work — comparing "3 regimes" numbers to "40 regimes" numbers directly is comparing across two different detection methods, not a strict apples-to-apples scale-up of the identical procedure.
