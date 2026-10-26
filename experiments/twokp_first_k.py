"""twokp_first_k.py

Exploratory (2026-10-26): for each odd prime p, the smallest k >= 1 with 2kp+1 prime,
and how well anything predicts it.  Standalone number-generator question, unrelated to
the gap/HMM tests.  Results are logged in CLAUDE.md ("Session Log: 2026-10-26").

Run: .venv-analysis/bin/python experiments/twokp_first_k.py
"""

from __future__ import annotations

import math

import numpy as np

P_MAX = 20000


def isprime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def first_k(p: int) -> int:
    k = 1
    while not isprime(2 * k * p + 1):
        k += 1
    return k


def main():
    primes = [p for p in range(5, P_MAX + 1) if isprime(p)]
    ks = np.array([first_k(p) for p in primes], dtype=float)
    pa = np.array(primes, dtype=float)

    # density predictor: 2kp+1 is odd and never divisible by p, so it is prime with
    # probability ~ 2*(p/(p-1))/ln(2kp+1); evaluated at k=3 (first k ~ geometric).
    pred = np.array([1 / (2 * (p / (p - 1)) / math.log(6 * p)) for p in primes])

    print(f"{len(primes)} primes, p=5..{primes[-1]}")
    print(f"mean first k {ks.mean():.3f}  median {np.median(ks):.0f}  max {ks.max():.0f}  "
          f"(density predictor mean {pred.mean():.3f})")
    for lo, hi in ((5, 100), (100, 1000), (1000, 5000), (5000, P_MAX + 1)):
        m = (pa >= lo) & (pa < hi)
        print(f"  p in [{lo},{hi}): mean first k {ks[m].mean():.2f}  (density pred {pred[m].mean():.2f})")
    print(f"lag-1 corr of consecutive first-k: {np.corrcoef(ks[:-1], ks[1:])[0, 1]:+.3f}")

    const = np.full(len(ks), ks.mean())
    prev = np.concatenate([[ks.mean()], ks[:-1]])
    print(f"MAE  constant mean {np.abs(ks - const).mean():.3f} | "
          f"density {np.abs(ks - pred).mean():.3f} | previous k {np.abs(ks - prev).mean():.3f}")


if __name__ == "__main__":
    main()
