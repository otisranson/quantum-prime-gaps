# Sim-based MI overlap vs. gap-space proxy, 20k scale -- 20260818_234059

## Sim-based MI overlap rate vs. null baseline

- 37 MI-based changepoints (data-driven stopping, see `/home/oranson/Projects/quantum-prime-gaps/output/prime/20260818_234057/mi_changepoints_sim_20k.json`)
- Contained in a flagged entropy window: **10 / 37** (27.0%)
- Null-expected containment: **6.06 / 37** (domain coverage = 0.1638)
- Individually significant at p<0.05 (uncorrected): **3 / 37** vs. **1.85** expected by chance alone

## Sim-based MI overlap rate vs. gap-space proxy rate

| | n changepoints | contained | containment rate | significant (p<0.05) |
|---|---|---|---|---|
| **Sim-based MI (this run)** | 37 | 10 | 0.2703 | 3 |
| **Gap-space proxy (prior run)** | 39 | 11 | 0.2821 | 3 |

## Interpretation

**Sim-based MI shows comparable correlation with entropy regime boundaries relative to the classical gap-space proxy.** Sim-based MI containment (27.0%) and the gap-space proxy's (28.2%) are close enough (within 15% relative) that this reads as comparable, not a clear win for either signal. Neither comparison here applies a multiple-comparison correction across the two detector types being compared, and both overlap tests share the same underlying limitation (the flagged entropy windows themselves are candidates, not confirmed regime boundaries -- see experiments/gap_entropy_windows.py) -- read this as a relative comparison between two proxies for the same underlying question, not a confirmation of either one in isolation.
