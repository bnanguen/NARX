"""
Visual / statistical diagnostics for an unknown NARX dataset.

These plots don't select the lag orders on their own (that's the job of
`lag_selection.py`), but they give us a quick sanity-check before running
the more expensive grid search. If the cross-correlation between u and y
peaks at lag L, the input-side `nb` should at least cover L. If the PACF
of y dies off after lag M, na around M is a sensible starting point.
"""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_acf_pacf(Y, nlags=20, title=""):
    """ACF and PACF for each output column of Y, side by side."""
    n = Y.shape[1]
    fig, axes = plt.subplots(n, 2, figsize=(11, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for j in range(n):
        plot_acf(Y[:, j], lags=nlags, ax=axes[j, 0])
        axes[j, 0].set_title(f"ACF y{j + 1}")
        plot_pacf(Y[:, j], lags=nlags, ax=axes[j, 1], method="ywm")
        axes[j, 1].set_title(f"PACF y{j + 1}")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def cross_correlation(u, y, max_lag=20):
    """
    corr(u(k - lag), y(k)) for lag in [-max_lag, max_lag].

    A positive lag with a strong correlation means past values of u carry
    information about the current y, i.e. the "pure delay" we are trying
    to estimate. A negative lag with strong correlation would suggest the
    arrow runs the other way (which would be unusual here).
    """
    u = np.asarray(u).ravel()
    y = np.asarray(y).ravel()

    u = (u - u.mean()) / (u.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)

    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        if lag >= 0:
            corr[i] = np.mean(u[: len(u) - lag] * y[lag:])
        else:
            corr[i] = np.mean(u[-lag:] * y[: len(u) + lag])
    return lags, corr


def plot_cross_correlation(u, y, max_lag=20, title=""):
    lags, corr = cross_correlation(u, y, max_lag=max_lag)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.stem(lags, corr, basefmt=" ")
    ax.axhline(0, lw=0.5, color="k")
    ax.set_xlabel("lag")
    ax.set_ylabel("corr(u(k - lag), y(k))")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return lags, corr, fig


def estimate_input_delay(lags, corr):
    """
    Heuristic: pick the*positive lag at which |corr(u, y)| is largest.
    Used as a quick prior on `d` (and to lower-bound `nb`).
    """
    pos = lags >= 0
    return int(lags[pos][np.argmax(np.abs(corr[pos]))])
