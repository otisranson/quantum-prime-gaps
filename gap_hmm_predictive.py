"""gap_hmm_predictive.py

Predictive framing: does a hidden-state model beat the PNT "force" alone at
one-step-ahead prediction of the next prime gap, on held-out primes?

Emission family (shared by every model, so likelihoods are exactly comparable):
    k = gap/2 >= 1,  P(k | m, p) = q (1-q)^(k-1),  q = 2 / (m * ln p)
i.e. geometric with mean m * ln(p) / 2.  m = 1 is plain Cramer + PNT drift.

Models:
  B0   m = 1 fixed                      (the force alone, 0 params)
  B0f  m fitted                         (1 param)
  B1   m depends on previous gap k_prev (observed Markov; Lemke Oliver-
       Soundararajan style dependence; NB_PREV params)
  HMM  K hidden states, each with its own m_s, full transition matrix.

Scoring: one-step-ahead predictive log-likelihood on the last TEST_FRAC of the
sequence with parameters frozen from the training part (HMM uses the forward
filter, so it only ever sees past gaps).  Paired per-gap differences vs B1
give a standard error.

Run: .venv-analysis/bin/python gap_hmm_predictive.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).parent
TEST_FRAC = 0.2
NB_PREV = 5  # previous-gap classes for B1: k_prev = 1,2,3,4,>=5
K_RANGE = (1, 2, 3, 4, 6)
RESTARTS = 6
EM_ITERS = 80
SEED = 0


def load():
    d = json.loads((REPO_ROOT / "data/primes_20000.json").read_text())
    p = np.array(d["primes"], dtype=float)
    gaps = np.diff(p)[1:]          # drop the odd 2->3 gap
    L = np.log(p[1:-1])            # ln of the prime each gap starts at
    k = (gaps / 2).astype(int)
    return k, L


def logpmf(k, L, m):
    """log P(k | m, L) for arrays; m scalar or array broadcastable to k."""
    q = np.clip(2.0 / (m * L), 1e-6, 0.999)
    return np.log(q) + (k - 1) * np.log1p(-q)


def fit_m(k, L, w, lo=0.3, hi=3.0):
    r = minimize_scalar(lambda m: -(w * logpmf(k, L, m)).sum(),
                        bounds=(lo, hi), method="bounded")
    return r.x


# ---------------------------------------------------------------- baselines
def prev_class(k):
    c = np.minimum(k, NB_PREV) - 1
    return np.concatenate([[NB_PREV - 1], c[:-1]])  # first gap: arbitrary class


def fit_b1(k, L):
    c = prev_class(k)
    return np.array([fit_m(k[c == j], L[c == j], np.ones((c == j).sum()))
                     for j in range(NB_PREV)])


def ll_b1(k, L, k_all_prev_class, ms):
    return logpmf(k, L, ms[k_all_prev_class])


# --------------------------------------------------------------------- HMM
def emission_matrix(k, L, ms):
    return np.stack([logpmf(k, L, m) for m in ms], axis=1)  # T x K


def forward_filter(logB, A, pi):
    """Returns scaled alphas and per-step predictive log-lik log P(x_t|x_<t)."""
    T, K = logB.shape
    B = np.exp(logB)
    alpha = np.zeros((T, K))
    ll = np.zeros(T)
    pred = pi
    for t in range(T):
        a = pred * B[t]
        s = a.sum()
        ll[t] = np.log(s)
        alpha[t] = a / s
        pred = alpha[t] @ A
    return alpha, ll


def backward(logB, A, ll):
    T, K = logB.shape
    B = np.exp(logB)
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        b = A @ (B[t + 1] * beta[t + 1])
        beta[t] = b / np.exp(ll[t + 1])
    return beta


def fit_hmm(k, L, K, rng):
    A = np.full((K, K), 0.1 / max(K - 1, 1)) if K > 1 else np.ones((1, 1))
    if K > 1:
        np.fill_diagonal(A, 0.9)
    pi = np.full(K, 1.0 / K)
    ms = np.sort(rng.uniform(0.6, 1.5, K)) if K > 1 else np.array([1.0])
    prev = -np.inf
    for _ in range(EM_ITERS):
        logB = emission_matrix(k, L, ms)
        alpha, ll = forward_filter(logB, A, pi)
        total = ll.sum()
        if total - prev < 1e-3:
            break
        prev = total
        beta = backward(logB, A, ll)
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        if K > 1:
            B = np.exp(logB)
            xi = (alpha[:-1, :, None] * A[None] *
                  (B[1:] * beta[1:])[:, None, :] / np.exp(ll[1:])[:, None, None])
            A = xi.sum(0)
            A /= A.sum(1, keepdims=True)
        pi = gamma[0]
        ms = np.array([fit_m(k, L, gamma[:, s]) for s in range(K)])
    return dict(ms=ms, A=A, pi=pi, train_ll=total)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    k, L = load()
    T = len(k)
    n_tr = int(T * (1 - TEST_FRAC))
    ktr, Ltr, kte, Lte = k[:n_tr], L[:n_tr], k[n_tr:], L[n_tr:]
    print(f"{T} gaps; train {n_tr}, test {T - n_tr}")

    results, test_ll = {}, {}

    # B0: force alone
    test_ll["B0"] = logpmf(kte, Lte, 1.0)
    results["B0"] = dict(params=0, train_ll=float(logpmf(ktr, Ltr, 1.0).sum()))

    # B0f
    m0 = fit_m(ktr, Ltr, np.ones(n_tr))
    test_ll["B0f"] = logpmf(kte, Lte, m0)
    results["B0f"] = dict(params=1, m=m0, train_ll=float(logpmf(ktr, Ltr, m0).sum()))

    # B1 (prev-gap class)
    ms1 = fit_b1(ktr, Ltr)
    c_all = prev_class(k)
    test_ll["B1"] = ll_b1(kte, Lte, c_all[n_tr:], ms1)
    results["B1"] = dict(params=NB_PREV, ms=ms1.tolist(),
                         train_ll=float(ll_b1(ktr, Ltr, c_all[:n_tr], ms1).sum()))

    # HMMs
    for K in K_RANGE:
        best = None
        for _ in range(RESTARTS if K > 1 else 1):
            f = fit_hmm(ktr, Ltr, K, rng)
            if best is None or f["train_ll"] > best["train_ll"]:
                best = f
        # frozen-parameter forward filter over train+test; score only test part
        logB = emission_matrix(k, L, best["ms"])
        _, ll = forward_filter(logB, best["A"], best["pi"])
        test_ll[f"HMM{K}"] = ll[n_tr:]
        results[f"HMM{K}"] = dict(
            params=K + K * (K - 1), ms=best["ms"].tolist(),
            A_diag=np.diag(best["A"]).tolist(), train_ll=float(best["train_ll"]))
        print(f"  fitted HMM K={K}  ({time.time() - t0:.0f}s)")

    # report
    print(f"\n{'model':6s} {'params':>6s} {'train BIC/gap':>14s} "
          f"{'test nats/gap':>14s} {'vs B0':>9s} {'vs B1 (z)':>14s}")
    for name, r in results.items():
        bic = (-2 * r["train_ll"] + r["params"] * np.log(n_tr)) / n_tr
        te = test_ll[name]
        d0 = te - test_ll["B0"]
        d1 = te - test_ll["B1"]
        z = d1.mean() / (d1.std(ddof=1) / np.sqrt(len(d1))) if d1.std() > 0 else 0.0
        r.update(test_nats_per_gap=float(te.mean()), vs_B0=float(d0.mean()),
                 vs_B1=float(d1.mean()), z_vs_B1=float(z))
        print(f"{name:6s} {r['params']:6d} {bic:14.5f} {te.mean():14.5f} "
              f"{d0.mean():+9.5f} {d1.mean():+9.5f} ({z:+.2f})")

    print("\nfitted multipliers m_s (mean gap = m * ln p):")
    for name, r in results.items():
        if "ms" in r:
            print(f"  {name}: {np.round(r['ms'], 3).tolist()}")
        if "A_diag" in r and r["params"] > 1:
            print(f"      self-transition: {np.round(r['A_diag'], 3).tolist()}")

    out = REPO_ROOT / "output/prime" / time.strftime("%Y%m%d_%H%M%S") / "gap_hmm_predictive"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {out}/results.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
