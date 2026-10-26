"""gap_hmm_wheel.py

Round 3 of the predictive test (hypotheses/predictive_hmm_gap_model.md).

Emission = round 2's singular-series categorical over k = gap/2 in 1..KMAX,
additionally masked to gaps for which p + 2k is not divisible by any wheel
prime (3,5 for W=30; 3,5,7 for W=210), renormalised over the allowed k:

    P(k | m, p, residue) ∝ mask(p mod W, k) * S(2k) * exp(-k * 2 / (m ln p))

Context models fit one m per observed-context class (c fixed at 1). Also: a
K=2 HMM on this emission, and 5 expanding-window folds for the observed models.

Run: .venv-analysis/bin/python gap_hmm_wheel.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp  # noqa: F401  (kept for parity with round 2)

from gap_hmm_predictive import backward, forward_filter
from gap_hmm_singular_series import KMAX, KS, LBIN, LOGS

REPO_ROOT = Path(__file__).parent
TEST_FRAC = 0.2
FOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
FOLD_LEN = 0.1
MIN_CLASS = 30           # classes with fewer train rows fall back to global m
EM_ITERS = 50
RESTARTS = 3
SEED = 0
M_BOUNDS = (0.4, 2.5)


def load_wheel_data():
    d = json.loads((REPO_ROOT / "data/primes_20000.json").read_text())
    p = np.array(d["primes"], dtype=np.int64)
    gaps = np.diff(p)
    start = np.flatnonzero(p[:-1] >= 11)[0]
    ps, g = p[start:-1], gaps[start:]
    assert np.all(g % 2 == 0)
    return ps, (g // 2).astype(int), np.log(ps.astype(float)), start


class Model:
    """Tabulated masked emission for one wheel."""

    def __init__(self, ps, L, wheel_primes):
        idx = np.round(L / LBIN).astype(int)
        self.ub, self.inv = np.unique(idx, return_inverse=True)
        self.Lb = self.ub * LBIN
        W = int(np.prod(wheel_primes))
        res = ps % W
        self.ures, self.rinv = np.unique(res, return_inverse=True)
        self.mask = np.array([[all((int(r) + 2 * int(k)) % q for q in wheel_primes)
                               for k in KS] for r in self.ures], dtype=float)

    def logp(self, k, m):
        theta = 2.0 / (m * self.Lb)
        logw = LOGS[None, :] - theta[:, None] * KS[None, :]
        mx = logw.max(1, keepdims=True)
        Z = np.exp(logw - mx) @ self.mask.T                       # nbins x nres
        logZ = np.log(Z) + mx
        return logw[self.inv, k - 1] - logZ[self.inv, self.rinv]


def fit_m(model, k, w):
    r = minimize_scalar(lambda m: -(w * model.logp(k, m)).sum(),
                        bounds=M_BOUNDS, method="bounded")
    return r.x


def fit_context(model, k, labels, tr_mask):
    """One m per class (fit on train rows); returns per-t log-prob for all t."""
    m_glob = fit_m(model, k, tr_mask.astype(float))
    lp = np.zeros(len(k))
    ms = {}
    for j in np.unique(labels):
        sel = labels == j
        n_tr = (sel & tr_mask).sum()
        m = fit_m(model, k, (sel & tr_mask).astype(float)) if n_tr >= MIN_CLASS else m_glob
        ms[int(j)] = float(m)
        lp[sel] = model.logp(k, m)[sel]
    return lp, ms


def contexts(k):
    k1 = np.concatenate([[1], k[:-1]])
    k2 = np.concatenate([[1, 1], k[:-2]])
    cap = lambda x: np.minimum(x, 5) - 1  # noqa: E731
    return {
        "M1": cap(k1),
        "M1r": k1 % 15,
        "M2": cap(k1) * 5 + cap(k2),
        "M0": np.zeros(len(k), dtype=int),
    }


def fit_hmm2(model, k, n_tr, rng):
    K = 2
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi = np.array([0.5, 0.5])
    ms = np.sort(rng.uniform(0.8, 1.3, K))
    kt = k[:n_tr]
    sub = Model.__new__(Model)
    sub.__dict__.update(model.__dict__)
    sub.inv, sub.rinv = model.inv[:n_tr], model.rinv[:n_tr]
    prev, total = -np.inf, -np.inf
    for _ in range(EM_ITERS):
        logB = np.stack([sub.logp(kt, m) for m in ms], axis=1)
        alpha, ll = forward_filter(logB, A, pi)
        total = ll.sum()
        if total - prev < 1e-2:
            break
        prev = total
        beta = backward(logB, A, ll)
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        B = np.exp(logB)
        xi = (alpha[:-1, :, None] * A[None] *
              (B[1:] * beta[1:])[:, None, :] / np.exp(ll[1:])[:, None, None])
        A = xi.sum(0)
        A /= A.sum(1, keepdims=True)
        pi = gamma[0]
        ms = np.array([fit_m(sub, kt, gamma[:, s]) for s in range(K)])
    return dict(ms=ms, A=A, pi=pi, train_ll=total)


def paired(a, b):
    d = a - b
    sd = d.std(ddof=1)
    return d.mean(), (d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    ps, k, L, start = load_wheel_data()
    k = np.minimum(k, KMAX)
    T = len(k)
    assert np.all(k >= 1)
    n_tr = int(T * (1 - TEST_FRAC))
    tr_mask = np.arange(T) < n_tr
    te = np.arange(n_tr, T)
    print(f"{T} gaps (p >= 11); train {n_tr}, test {T - n_tr}")

    m30 = Model(ps, L, (3, 5))
    m210 = Model(ps, L, (3, 5, 7))
    assert np.isfinite(m30.logp(k, 1.0)).all() and np.isfinite(m210.logp(k, 1.0)).all(), \
        "observed gap masked out: mask is wrong"
    ctx = contexts(k)

    test_ll, results = {}, {}
    n_cls = {"M0": 1, "M1": 5, "M1r": 15, "M2": 25}
    for name, lab in ctx.items():
        lp, ms = fit_context(m30, k, lab, tr_mask)
        test_ll[name] = lp[te]
        results[name] = dict(params=n_cls[name], train_ll=float(lp[:n_tr].sum()), ms=ms)
    lp, ms = fit_context(m210, k, ctx["M0"], tr_mask)
    test_ll["M0_210"] = lp[te]
    results["M0_210"] = dict(params=1, train_ll=float(lp[:n_tr].sum()), ms=ms)
    print(f"  observed models done ({time.time() - t0:.0f}s)")

    best = None
    for _ in range(RESTARTS):
        f = fit_hmm2(m30, k, n_tr, rng)
        if best is None or f["train_ll"] > best["train_ll"]:
            best = f
    logB = np.stack([m30.logp(k, m) for m in best["ms"]], axis=1)
    _, ll = forward_filter(logB, best["A"], best["pi"])
    test_ll["HMM2"] = ll[n_tr:]
    results["HMM2"] = dict(params=4, train_ll=float(best["train_ll"]), ms=best["ms"].tolist(),
                           A_diag=np.diag(best["A"]).tolist())
    print(f"  HMM2 done ({time.time() - t0:.0f}s)")

    # reference: round-2 S0f on the same rows (no wheel)
    from gap_hmm_singular_series import Binned, fit_mc
    bn = Binned(L)
    x = fit_mc(bn, k, tr_mask.astype(float), [1.0, 1.0], free_c=False)
    s0f = bn.logp(k, x[0], 1.0)[te]
    test_ll["S0f"] = s0f
    results["S0f"] = dict(params=1, train_ll=float(bn.logp(k, x[0], 1.0)[:n_tr].sum()), ms=[x[0]])

    print(f"\n{'model':7s} {'params':>6s} {'BIC/gap':>9s} {'test nats/gap':>14s} "
          f"{'vs S0f':>9s} {'vs M0 (z)':>15s}")
    for name, r in results.items():
        bic = (-2 * r["train_ll"] + r["params"] * np.log(n_tr)) / n_tr
        d0, _ = paired(test_ll[name], test_ll["S0f"])
        d1, z1 = paired(test_ll[name], test_ll["M0"])
        r.update(test_nats_per_gap=float(test_ll[name].mean()), vs_S0f=float(d0),
                 vs_M0=float(d1), z_vs_M0=float(z1))
        print(f"{name:7s} {r['params']:6d} {bic:9.4f} {test_ll[name].mean():14.5f} "
              f"{d0:+9.5f} {d1:+9.5f} ({z1:+.2f})")

    print("\nlag-1 vs lag-2:")
    for a, b in (("M1", "M0"), ("M1r", "M1"), ("M2", "M1"), ("M2", "M1r"), ("M0_210", "M0"),
                 ("HMM2", "M0")):
        d, z = paired(test_ll[a], test_ll[b])
        print(f"  {a} - {b}: {d:+.5f}  (z {z:+.2f})")
        results[f"diff_{a}_minus_{b}"] = dict(delta=float(d), z=float(z))
    print("\nfitted m by class:")
    for name in ("M0", "M1", "M1r"):
        print(f"  {name}: {[round(v, 3) for v in results[name]['ms'].values()]}")
    print(f"  HMM2: m={np.round(results['HMM2']['ms'], 3).tolist()}  "
          f"self-trans={np.round(results['HMM2']['A_diag'], 3).tolist()}")

    # rolling-origin folds (observed models only)
    print(f"\nrolling-origin folds ({len(FOLDS)}), test nats/gap relative to M0:")
    fold_res = []
    for f in FOLDS:
        a, b = int(T * f), int(T * (f + FOLD_LEN))
        trm = np.arange(T) < a
        tem = np.arange(a, b)
        row = {}
        for name in ("M0", "M1", "M1r", "M2"):
            lp, _ = fit_context(m30, k, ctx[name], trm)
            row[name] = lp[tem]
        out = {nm: float((row[nm] - row["M0"]).mean()) for nm in ("M1", "M1r", "M2")}
        out["M2_vs_M1"] = float((row["M2"] - row["M1"]).mean())
        out["M1r_vs_M1"] = float((row["M1r"] - row["M1"]).mean())
        fold_res.append(dict(train_frac=f, **out))
        print(f"  f={f:.1f}  M1 {out['M1']:+.5f}  M1r {out['M1r']:+.5f}  M2 {out['M2']:+.5f}"
              f"  | M1r-M1 {out['M1r_vs_M1']:+.5f}  M2-M1 {out['M2_vs_M1']:+.5f}")
    results["folds"] = fold_res
    for nm in ("M1", "M1r", "M2"):
        pos = sum(r[nm] > 0 for r in fold_res)
        print(f"  {nm} > M0 in {pos}/{len(fold_res)} folds, mean "
              f"{np.mean([r[nm] for r in fold_res]):+.5f}")

    out = REPO_ROOT / "output/prime" / time.strftime("%Y%m%d_%H%M%S") / "gap_hmm_wheel"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {out}/results.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
