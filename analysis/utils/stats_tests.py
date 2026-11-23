"""
Differential-abundance testing and PLS-DA validation for the cohort stages.

* moderated_two_group_test: limma-style empirical-Bayes moderated t-test
  (Smyth 2004). With 3 v 3 replicates a per-feature variance is estimated from
  4 degrees of freedom; borrowing the variance distribution across all features
  stabilises it. Also protects against the artificially tiny variances that
  min/2 imputation produces.
* plsda_cross_validated: PLS-DA with the scaler fitted INSIDE each
  leave-one-out fold, Q2 from out-of-fold predictions, and a permutation p-value
  computed on that same cross-validated statistic (with the +1 correction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma, polygamma


# --------------------------------------------------------------------------
# moderated t-test
# --------------------------------------------------------------------------
def _trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x for y by Newton's method (needed to fit the variance prior); limits handled for tiny/huge x."""
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = polygamma(1, y)
        dif = tri * (1.0 - tri / x) / polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return float(y)


def fit_variance_prior(s2: np.ndarray, df: float) -> tuple[float, float]:
    """Empirical-Bayes prior for feature variances: (prior df d0, prior variance s0^2).
    d0 = inf means the variances are all alike (full shrinkage)."""
    s2 = np.asarray(s2, dtype=float)
    s2 = s2[np.isfinite(s2)]
    if len(s2) < 3:
        return 0.0, float(np.nanmedian(s2)) if len(s2) else 1.0
    floor = 1e-5 * np.median(s2[s2 > 0]) if np.any(s2 > 0) else 1e-12
    z = np.log(np.maximum(s2, floor))
    e = z - digamma(df / 2) + np.log(df / 2)
    emean = e.mean()
    evar = e.var(ddof=1) - polygamma(1, df / 2)
    if evar > 0:
        d0 = 2.0 * _trigamma_inverse(evar)
        s0 = float(np.exp(emean + digamma(d0 / 2) - np.log(d0 / 2)))
    else:
        d0 = np.inf
        s0 = float(np.exp(emean))
    return float(d0), s0


def moderated_two_group_test(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """a: features x nA, b: features x nB (log scale, no NaN). Tests mean(a) != mean(b).
    Returns log2_fc (a - b), t, p_value, df_total, s2_post, and the fitted prior."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.shape[1], b.shape[1]
    d = na + nb - 2
    if d < 1:
        raise ValueError("need at least 3 samples across the two groups")
    diff = a.mean(axis=1) - b.mean(axis=1)
    ss = ((a - a.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) + \
         ((b - b.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    s2 = ss / d
    d0, s0 = fit_variance_prior(s2, d)
    if np.isinf(d0):
        s2_post = np.full_like(s2, s0)
        df_total = np.inf
    else:
        s2_post = (d0 * s0 + d * s2) / (d0 + d)
        df_total = d0 + d
    t = diff / np.sqrt(s2_post * (1.0 / na + 1.0 / nb))
    p = 2 * (stats.norm.sf(np.abs(t)) if np.isinf(df_total) else stats.t.sf(np.abs(t), df_total))
    out = pd.DataFrame({"log2_fc": diff, "t_stat": t, "p_value": p, "s2_post": s2_post})
    out.attrs["prior_df"] = d0
    out.attrs["prior_var"] = s0
    out.attrs["df_total"] = df_total
    return out


# --------------------------------------------------------------------------
# PLS-DA cross-validation
# --------------------------------------------------------------------------
def _one_hot(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dummy-code group labels: returns (samples x classes 0/1 matrix, sorted class names)."""
    classes = np.array(sorted(set(labels)))
    y = (labels[:, None] == classes[None, :]).astype(float)
    return y, classes


def _loo_predictions(x: np.ndarray, y: np.ndarray, n_components: int) -> np.ndarray:
    """Leave-one-out PLS predictions; scaling is fitted on each training fold only, so no information leaks from the held-out sample."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    n = len(x)
    pred = np.zeros_like(y)
    for i in range(n):
        train = np.arange(n) != i
        scaler = StandardScaler().fit(x[train])          # fitted on the training fold only
        k = min(n_components, int(train.sum()) - 1, x.shape[1])
        pls = PLSRegression(n_components=k, scale=False).fit(scaler.transform(x[train]), y[train])
        pred[i] = pls.predict(scaler.transform(x[i:i + 1]))[0]
    return pred


def _q2(y: np.ndarray, pred: np.ndarray) -> float:
    """Q2 = 1 - PRESS/TSS from out-of-fold predictions (1 = perfect, <= 0 = no predictive value); NaN if y has no variance."""
    press = ((y - pred) ** 2).sum()
    tss = ((y - y.mean(axis=0)) ** 2).sum()
    return float(1.0 - press / tss) if tss > 0 else float("nan")


def plsda_cross_validated(
    x: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    n_permutations: int = 999,
    seed: int | None = 0,
) -> dict:
    """x: samples x features (log scale). Returns Q2, LOO accuracy and a permutation
    p-value for Q2. Unlike the in-sample R2, Q2 is computed from out-of-fold
    predictions, so it can go negative when there is no real structure."""
    labels = np.asarray(labels)
    y, classes = _one_hot(labels)
    pred = _loo_predictions(x, y, n_components)
    q2 = _q2(y, pred)
    acc = float((classes[pred.argmax(axis=1)] == labels).mean())

    p_value = None
    perm_q2: list[float] = []
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_permutations):
            yp = y[rng.permutation(len(y))]
            perm_q2.append(_q2(yp, _loo_predictions(x, yp, n_components)))
        p_value = float((1 + np.sum(np.array(perm_q2) >= q2)) / (1 + n_permutations))
    return {"q2": q2, "loo_accuracy": acc, "permutation_p_value": p_value,
            "n_permutations": n_permutations, "perm_q2_mean": float(np.mean(perm_q2)) if perm_q2 else None}
