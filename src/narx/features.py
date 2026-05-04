"""
NARX regressor matrix construction.

Convention is the one used in eq. (1) of the project brief:

    target  = y(k+1)
    outputs = [ y_1(k-d), y_1(k-d-1), ..., y_1(k-d-na),
                y_2(k-d), ..., y_2(k-d-na) ]                 ->  n_out * (na+1) terms
    inputs  = [ u_1(k), u_1(k-1), ..., u_1(k-nb),
                u_2(k), ..., u_2(k-nb) ]                     ->  n_in * (nb+1) terms

So `na` and `nb` are the deepest lag, not the count, and the regressor at
time k contains `na+1` output terms per output and `nb+1` input terms per
input.
"""

import numpy as np


def independent_matrix(Y, U, d, na, nb, sigma_y=0.0, rng=None):
    """
    Build the (X, target) regressor matrices for the NARX learning problem.

    Parameters
    ----------
    Y : ndarray, shape (N, n_out)
        Output series.
    U : ndarray, shape (N, n_in)
        Input series. Must have the same N as Y.
    d, na, nb : int
        Delay, output deepest lag, input deepest lag (see module docstring).
    sigma_y : float, default 0.0
        If > 0 (and `rng` is given), Gaussian noise is injected into the
        observed-output regressors at training time. This is the
        "noise injection" trick used to harden the recursive predictor
        against feedback drift (Bengio et al. 2015 scheduled-sampling
        analogue for autoregressive regression).
    rng : numpy.random.Generator or None
        Required if `sigma_y > 0`.

    Returns
    -------
    X : ndarray, shape (N - 1 - max(d+na, nb), n_features)
        The regressor matrix.
    Yt : ndarray, shape (N - 1 - max(d+na, nb), n_out)
        The corresponding targets y(k+1).
    """
    N = len(Y)
    n_out = Y.shape[1]
    n_in = U.shape[1]

    warmup = max(d + na, nb)

    rows, targets = [], []
    for k in range(warmup, N - 1):
        phi = []

        # output-history regressors (with optional noise injection)
        for j in range(n_out):
            for lag in range(d, d + na + 1):
                v = Y[k - lag, j]
                if sigma_y > 0 and rng is not None:
                    v = v + rng.normal(0.0, sigma_y)
                phi.append(v)

        # input-history regressors (no noise: inputs are observed cleanly)
        for j in range(n_in):
            for lag in range(0, nb + 1):
                phi.append(U[k - lag, j])

        rows.append(phi)
        targets.append(Y[k + 1])

    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def feature_names(d, na, nb, n_out=2, n_in=2):
    """Column labels matching `independent_matrix`'s feature order."""
    names = []
    for j in range(n_out):
        for lag in range(d, d + na + 1):
            names.append(f"y{j + 1}(k-{lag})")
    for j in range(n_in):
        for lag in range(0, nb + 1):
            names.append(f"u{j + 1}(k-{lag})")
    return names


def n_features(d, na, nb, n_out=2, n_in=2):
    """How many features `independent_matrix` produces for the given orders."""
    return n_out * (na + 1) + n_in * (nb + 1)
