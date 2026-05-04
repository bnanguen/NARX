# Task 2a: pick (d, na, nb) by multi-proxy CV-NMSE majority vote.
# Five proxies cover different inductive biases (boosting, bagging,
# ExtraTrees, XGBoost, plus a Ridge on a sin/x^2/x^3/pairwise expansion).
# CV uses TimeSeriesSplit, series-parallel mode (phi from observed y).

import itertools
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import independent_matrix
from .metrics import nmse


class NARXFeatureRidge(BaseEstimator, RegressorMixin):
    "Ridge on [X, sin(X), X^2, X^3, pairwise products]. Smooth proxy."

    def __init__(self, alpha=1e-6):
        self.alpha = alpha

    def _expand(self, X):
        feats = [X, np.sin(X), X**2, X**3]
        n = X.shape[1]
        for i in range(n):
            for j in range(i, n):
                feats.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(feats)

    def fit(self, X, y):
        self.model_ = Pipeline(
            [("sc", StandardScaler()), ("rg", Ridge(alpha=self.alpha))]
        )
        self.model_.fit(self._expand(X), y)
        return self

    def predict(self, X):
        return self.model_.predict(self._expand(X))


PROXY_MODELS = {
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=0, n_jobs=-1
    ),
    "FeatureRidge": NARXFeatureRidge(alpha=1e-6),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    ),
}


def cv_nmse_nonlinear(Y, U, d, na, nb, proxy_model, n_splits=5):
    "mean per-output CV-NMSE for one (d, na, nb) under one proxy."
    try:
        X, T = independent_matrix(Y, U, d, na, nb)
        if len(X) < max(20, n_splits + 1):
            print(f"  too few samples ({len(X)}) for d={d}, na={na}, nb={nb}")
            return np.inf

        tscv = TimeSeriesSplit(n_splits=n_splits)
        sc = StandardScaler()
        scores = []
        for tr, val in tscv.split(X):
            Xs_tr = sc.fit_transform(X[tr])
            Xs_val = sc.transform(X[val])
            for j in range(T.shape[1]):
                m = clone(proxy_model)
                m.fit(Xs_tr, T[tr, j])
                yp = m.predict(Xs_val)
                # nmse expects 2D-ish arrays; (N,1) is fine here
                scores.append(nmse(T[val, j].reshape(-1, 1), yp.reshape(-1, 1)))
        return float(np.mean(scores))
    except Exception as e:
        print(f"  err for (d={d}, na={na}, nb={nb}): {e}")
        return np.inf


def grid_search(
    Y,
    U,
    d_range,
    na_range,
    nb_range,
    proxy_models=None,
    label="",
    n_splits=5,
    verbose=True,
):
    """Multi-proxy grid search over (d, na, nb).

    For every triple in the cartesian product of d_range x na_range x nb_range
    we compute cv_nmse_nonlinear under each proxy. Each proxy then votes
    for its argmin triple; the triple with the most votes wins. Ties are
    broken by the mean score across proxies (lower is better).

    Y, U          training arrays, shape (N, n_out) and (N, n_inp).
    d_range, na_range, nb_range
                  iterables of candidate values for the structural
                  hyperparameters of the regressor.
    proxy_models  dict {name: estimator}; defaults to PROXY_MODELS above.
    label         passed through into print() messages, helps when
                  running grids on several systems back-to-back.
    n_splits      number of TimeSeriesSplit folds.
    verbose       print per-proxy summaries as we go.

    returns (best_triple, all_results) where all_results[name] is the
    list of (d, na, nb, score) tuples for that proxy, sorted ascending.
    """
    if proxy_models is None:
        proxy_models = PROXY_MODELS

    cands = list(itertools.product(d_range, na_range, nb_range))
    total = len(cands)
    all_results = {}
    best_per_model = {}

    for name, model in proxy_models.items():
        if verbose:
            print(f"\n  [{label}] proxy: {name}")
        best_score, best_triple = np.inf, None
        rows = []
        for i, (d, na, nb) in enumerate(cands, 1):
            s = cv_nmse_nonlinear(Y, U, d, na, nb, proxy_model=model, n_splits=n_splits)
            rows.append((d, na, nb, s))
            if s < best_score:
                best_score, best_triple = s, (d, na, nb)
            if verbose and (i % 10 == 0 or i == total):
                print(
                    f"    {i}/{total}  best so far: {best_triple} NMSE={best_score:.3f}"
                )
        rows.sort(key=lambda r: r[3])
        all_results[name] = rows
        best_per_model[name] = best_triple
        if verbose:
            print(f"    -> {name}: {best_triple} (CV-NMSE={best_score:.3f})")

    # majority vote across proxies; ties broken by mean score
    votes = Counter(best_per_model.values())
    top_count = max(votes.values())
    top = [t for t, v in votes.items() if v == top_count]

    if len(top) == 1:
        best = top[0]
    else:

        def avg(t):
            xs = [
                next(s for dd, nn, mm, s in all_results[name] if (dd, nn, mm) == t)
                for name in proxy_models
            ]
            return np.mean(xs)

        best = min(top, key=avg)

    if verbose:
        print(
            f"\n  [{label}] majority-vote pick: d={best[0]}, na={best[1]}, nb={best[2]}"
        )
        print(f"  per-proxy: {dict(best_per_model)}")

    return best, all_results


def plot_proxy_comparison(all_results, label="", d_fixed=0):
    "side-by-side (na, nb) heat-maps for each proxy at the given d."
    models = list(all_results.keys())
    na_vals = sorted({na for _, na, _, _ in all_results[models[0]]})
    nb_vals = sorted({nb for _, _, nb, _ in all_results[models[0]]})

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    fig.suptitle(f"{label}  CV-NMSE per proxy (d={d_fixed})")

    for ax, name in zip(axes, models):
        mat = np.full((len(na_vals), len(nb_vals)), np.nan)
        lookup = {(d, na, nb): s for d, na, nb, s in all_results[name] if d == d_fixed}
        for i, na in enumerate(na_vals):
            for j, nb in enumerate(nb_vals):
                v = lookup.get((d_fixed, na, nb), np.nan)
                if np.isfinite(v):
                    mat[i, j] = v

        if np.isnan(mat).all():
            ax.set_title(f"{name} (no data at d={d_fixed})")
            continue

        vmin = np.nanmin(mat)
        vmax = min(np.nanmax(mat), 2 * vmin)
        im = ax.imshow(mat, cmap="YlOrRd_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(nb_vals)))
        ax.set_xticklabels(nb_vals)
        ax.set_yticks(range(len(na_vals)))
        ax.set_yticklabels(na_vals)
        ax.set_xlabel("nb")
        if name == models[0]:
            ax.set_ylabel("na")
        ax.set_title(name, fontsize=10)

        for i in range(len(na_vals)):
            for j in range(len(nb_vals)):
                if not np.isnan(mat[i, j]):
                    star = "*" if np.isclose(mat[i, j], vmin) else ""
                    color = "white" if mat[i, j] < (vmin + vmax) / 2 else "black"
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.3f}\n{star}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
        plt.colorbar(im, ax=ax, label="CV-NMSE")

    plt.tight_layout()
    return fig
