# mod-5 last-digit transition experiment -- 20260818_223521

Source: `data/primes_5000.json` (5000-prime cache, not a fresh sieve). n=4997 primes after excluding 2, 3, 5. Train n=4797, test n=200.

## Transition matrix (train set only, P(next residue | current residue))

| current \ next | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **1** | 0.1401 | 0.3624 | 0.3557 | 0.1418 |
| **2** | 0.2623 | 0.1236 | 0.2924 | 0.3216 |
| **3** | 0.2049 | 0.2955 | 0.1169 | 0.3827 |
| **4** | 0.3876 | 0.2164 | 0.2508 | 0.1451 |

## Prediction accuracy

- Markov (transition-matrix) accuracy: **0.3550** (n=200 held-out transitions)
- Uniform-random baseline: 0.2500
- Majority-class baseline: 0.2150 (always predicting residue 3)
- Permutation null (n=5000 trials, shuffled actual labels vs. fixed predictions): mean=0.2620, std=0.0287
- Observed accuracy sits at the **99.9th percentile** of that null distribution

## Confusion matrix (rows=actual, cols=predicted)

| actual \ predicted | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **1** | 20 | 7 | 0 | 24 |
| **2** | 13 | 19 | 0 | 21 |
| **3** | 11 | 12 | 0 | 20 |
| **4** | 9 | 12 | 0 | 32 |

## Residue pairs with transition probability > 40%

None.

## Verdict

- Clears the fixed 35% pre-registered threshold: **True**
- Clears a 95th-percentile permutation-null significance bar: **True**

Accuracy clears both the fixed threshold and the permutation-null significance bar. Still worth flagging: Lemke Oliver & Soundararajan (2016) documented a real, published bias in exactly this kind of consecutive-prime last-digit statistic at small N -- primes measurably avoid repeating their own last digit, an effect explained by Hardy-Littlewood k-tuple heuristics and expected to shrink toward uniform as N grows. A positive result here is far more likely a rediscovery of that known, already-explained effect than new structure worth layering into the gap estimator.
