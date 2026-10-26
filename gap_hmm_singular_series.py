"""gap_hmm_singular_series.py

Round 2 of the predictive-HMM test (see hypotheses/predictive_hmm_gap_model.md).
Same protocol as gap_hmm_predictive.py, but the emission is a proper categorical
over k = gap/2 in 1..KMAX with Hardy-Littlewood singular-series shape:

    P(k | m, c, p)  ∝  S(2k)^c * exp(-k * 2 / (m * ln p))
    S(g) = prod_{odd primes q | g} (q-1)/(q-2)

so the mod-3/5/7 favouritism is built into the "force" and a latent state only
has to explain what is left.  Each state/class has (m, c): scale and shape.
ln p is quantised (LBIN) for every model so normalisers can be tabulated.

Run: .venv-analysis/bin/python gap_hmm_singular_series.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from gap_hmm_predictive import backward, forward_filter, load

REPO_ROOT = Path(__file__).parent
TEST_FRAC = 0.2
KMAX = 150
LBIN = 0.02
NB_PREV = 5
K_RANGE = (1, 2, 3, 4)
RESTARTS = 4
EM_ITERS = 60
SEED = 0
BOUNDS = [(0.5, 2.0), (0.0, 3.0)]  # m, c


def singular_series_log(kmax: int) -> np.ndarray:
    out = np.zeros(kmax + 1)
    for k in range(1, kmax + 1):
        x = k
        while x % 2 == 0:
            x //= 2  # odd part of g = 2k
        q = 3
        s = 0.0
        while q * q <= x:
            if x % q == 0:
                s += np.log((q - 1) / (q - 2))
                while x % q == 0:
                    x //= q
            q += 2
        if x > 2 and x % 2 == 1:
            s += np.log((x - 1) / (x - 2))
        out[k] = s
    return out[1:]  # index 0 <-> k=1


LOGS = singular_series_log(KMAX)
KS = np.arange(1, KMAX + 1)


class Binned:
    """Quantised ln p, with tabulated per-bin log-normalisers for any (m, c)."""

    def __init__(self, L):
        idx = np.round(L / LBIN).astype(int)
        self.ub, self.inv = np.unique(idx, return_inverse=True)
        self.Lb = self.ub * LBIN

    def logp(self, k, m, c, rows=None):
        """log P(k_t | m, c, L_t) for every t (or t in `rows`)."""
        theta = 2.0 / (m * self.Lb)                        # per bin
        logw = c * LOGS[None, :] - theta[:, None] * KS[None, :]
        logZ = logsumexp(logw, axis=1)
        inv = self.inv if rows is None else self.inv[rows]
        kk = k if rows is None else k[rows]
        return logw[inv, kk - 1] - logZ[inv]


def fit_mc(bn, k, w, x0, rows=None, free_c=True):
    wr = w if rows is None else w[rows]
    if free_c:
        f = lambda x: -(wr * bn.logp(k, x[0], x[1], rows)).sum()  # noqa: E731
        r = minimize(f, x0, method="L-BFGS-B", bounds=BOUNDS)
        return r.x
    f = lambda x: -(wr * bn.logp(k, x[0], 1.0, rows)).sum()  # noqa: E731
    r = minimize(f, [x0[0]], method="L-BFGS-B", bounds=[BOUNDS[0]])
    return np.array([r.x[0], 1.0])


def prev_class(k):
    c = np.minimum(k, NB_PREV) - 1
    return np.concatenate([[NB_PREV - 1], c[:-1]])


def emission_matrix(bn, k, params):
    return np.stack([bn.logp(k, m, c) for m, c in params], axis=1)


def fit_hmm(bn, k, K, rng):
    A = np.ones((1, 1)) if K == 1 else np.full((K, K), 0.1 / (K - 1))
    if K > 1:
        np.fill_diagonal(A, 0.9)
    pi = np.full(K, 1.0 / K)
    params = np.array([[rng.uniform(0.8, 1.2), rng.uniform(0.5, 1.5)] for _ in range(K)])
    if K == 1:
        params[0] = [1.0, 1.0]
    prev = -np.inf
    total = -np.inf
    for _ in range(EM_ITERS):
        logB = emission_matrix(bn, k, params)
        alpha, ll = forward_filter(logB, A, pi)
        total = ll.sum()
        if total - prev < 1e-2:
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
        params = np.array([fit_mc(bn, k, gamma[:, s], params[s]) for s in range(K)])
    return dict(params=params, A=A, pi=pi, train_ll=total)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    k, L = load()
    k = np.minimum(k, KMAX)
    T = len(k)
    n_tr = int(T * (1 - TEST_FRAC))
    bn = Binned(L)
    tr = np.arange(n_tr)
    te = np.arange(n_tr, T)
    print(f"{T} gaps; train {n_tr}, test {T - n_tr}; {len(bn.ub)} ln p bins")

    results, test_ll = {}, {}
    ones = np.ones(T)

    # S0f: force + singular series, m fitted, c = 1
    x = fit_mc(bn, k, ones, [1.0, 1.0], rows=tr, free_c=False)
    lp = bn.logp(k, x[0], 1.0)
    test_ll["S0f"] = lp[te]
    results["S0f"] = dict(params=1, mc=x.tolist(), train_ll=float(lp[tr].sum()))

    # S1 / S1c: observed previous-gap class
    pc = prev_class(k)
    for name, free_c, npar in (("S1", False, NB_PREV), ("S1c", True, 2 * NB_PREV)):
        lp = np.zeros(T)
        mcs = []
        for j in range(NB_PREV):
            rows_tr = tr[pc[tr] == j]
            x = fit_mc(bn, k, ones, [1.0, 1.0], rows=rows_tr, free_c=free_c)
            mcs.append(x.tolist())
            sel = pc == j
            lp[sel] = bn.logp(k, x[0], x[1])[sel]
        test_ll[name] = lp[te]
        results[name] = dict(params=npar, mc=mcs, train_ll=float(lp[tr].sum()))
    print(f"  baselines done ({time.time() - t0:.0f}s)")

    # HMMs
    bn_tr = Binned(L[:n_tr])
    for K in K_RANGE:
        best = None
        for _ in range(RESTARTS if K > 1 else 1):
            f = fit_hmm(bn_tr, k[:n_tr], K, rng)
            if best is None or f["train_ll"] > best["train_ll"]:
                best = f
        logB = emission_matrix(bn, k, best["params"])
        _, ll = forward_filter(logB, best["A"], best["pi"])
        test_ll[f"HMM{K}"] = ll[n_tr:]
        results[f"HMM{K}"] = dict(
            params=2 * K + K * (K - 1), mc=best["params"].tolist(),
            A_diag=np.diag(best["A"]).tolist(), train_ll=float(best["train_ll"]))
        print(f"  fitted HMM K={K} ({time.time() - t0:.0f}s)")

    ref = "S1c"
    print(f"\n{'model':6s} {'params':>6s} {'train BIC/gap':>14s} {'test nats/gap':>14s} "
          f"{'vs S0f':>9s} {'vs ' + ref + ' (z)':>16s}")
    for name, r in results.items():
        bic = (-2 * r["train_ll"] + r["params"] * np.log(n_tr)) / n_tr
        te_ll = test_ll[name]
        d0 = te_ll - test_ll["S0f"]
        d1 = te_ll - test_ll[ref]
        z = d1.mean() / (d1.std(ddof=1) / np.sqrt(len(d1))) if d1.std() > 0 else 0.0
        r.update(test_nats_per_gap=float(te_ll.mean()), vs_S0f=float(d0.mean()),
                 vs_ref=float(d1.mean()), z_vs_ref=float(z))
        print(f"{name:6s} {r['params']:6d} {bic:14.5f} {te_ll.mean():14.5f} "
              f"{d0.mean():+9.5f} {d1.mean():+9.5f} ({z:+.2f})")

    print("\nfitted (m, c):")
    for name, r in results.items():
        print(f"  {name}: {np.round(r['mc'], 3).tolist()}")
        if "A_diag" in r and r["params"] > 2:
            print(f"      self-transition: {np.round(r['A_diag'], 3).tolist()}")

    out = REPO_ROOT / "output/prime" / time.strftime("%Y%m%d_%H%M%S") / "gap_hmm_singular_series"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {out}/results.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
