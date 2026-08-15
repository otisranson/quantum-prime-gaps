# Classical Baseline Comparison

**Target:** gap after prime 229  |  **Ground truth:** gap=4, prime=233
**Date:** 2026-08-15  |  **Quantum HW run:** 20260815_204703 (ibm_kingston)

---

## Comparison Table

| Method | Raw E[gap] | Rounded gap | Predicted prime | Error | Correct? |
|--------|:----------:|:-----------:|:---------------:|------:|:--------:|
| **Quantum circuit (hardware)** | 3.9578 | **4** | **233** | 0.0422 | ✓ |
| Mean gap | 4.6327 | 5 | 234 | 0.6327 | ✗ |
| Last gap | 2.0000 | 2 | 231 | 2.0000 | ✗ |
| Moving avg (w=4) | 7.5000 | 8 | 237 | 3.5000 | ✗ |
| Median gap | 4.0000 | 4 | 233 | 0.0000 | ✓ |
| Linear regression | 6.3112 | 6 | 235 | 2.3112 | ✗ |
| FFT (top-3) | 3.7200 | 4 | 233 | 0.2800 | ✓ |

---

## Ranked by Error (lowest to highest)

| Rank | Method | Error | Rounds correct? |
|------|--------|------:|:---------------:|
| 1 | Median gap | 0.0000 | ✓ |
| 2 | Quantum circuit (hardware) | 0.0422 | ✓ |
| 3 | FFT (top-3) | 0.2800 | ✓ |
| 4 | Mean gap | 0.6327 | ✗ |
| 5 | Last gap | 2.0000 | ✗ |
| 6 | Linear regression | 2.3112 | ✗ |
| 7 | Moving avg (w=4) | 3.5000 | ✗ |

---

## Interpretation

The quantum circuit hardware result (E[gap]=3.9578, error=0.0422) ranks **2 of 7** by absolute error.

Classical methods with lower error: Median gap.

Methods that round to the correct gap=4: **Quantum circuit (hardware), Median gap, FFT (top-3)**.

**Important caveats:**

- This is a single-instance comparison on one gap value. A robust evaluation would use backward verification across many held-out gaps.
- The last gap (gap[48]=2) and moving average of the final four gaps (12, 12, 4, 2) reflect recent sequence history but not its spectral structure.
- The median of prime gaps in this range is 4 — the same as the true answer — so a median predictor is a strong baseline for this particular instance.
- The FFT predictor's performance depends heavily on `top_k`; a fuller comparison would sweep it.
- The quantum circuit's recurrent feedback loop introduces structure that pure statistical baselines cannot replicate, but the single-instance comparison cannot confirm whether that structure is causally responsible for the correct prediction.

---

## Baseline Notes

**Mean gap:** mean of all 49 gaps
**Last gap:** gap[48] = 2
**Moving avg (w=4):** mean of last 4 gaps: [12, 12, 4, 2]
**Median gap:** median of all 49 gaps
**Linear regression:** slope=0.0671, intercept=3.0212, R²=0.1058, p=0.023
**FFT (top-3):** DFT extrapolation at t=49, keeping 3 dominant components

*Generated: 20260815_231029*