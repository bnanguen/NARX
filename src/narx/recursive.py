"""one-step and recursive (closed-loop) predictors.

`one_step_predict` feeds the true past y; `recursive_predict` feeds its
own predictions, which is what the brief grades. Both expect a fitted
sklearn-style `pipe` whose `.predict(phi_row)` returns shape (1, n_out).
First warmup steps left at 0 (matches y(<=0) = 0 in the brief).
"""

import numpy as np


def _build_phi(Y, U, k, d, na, nb, n_out, n_inp):
    phi = []
    for j in range(n_out):
        for lag in range(d, d + na + 1):
            idx = k - lag
            phi.append(Y[idx, j] if idx >= 0 else 0.0)
    for j in range(n_inp):
        for lag in range(0, nb + 1):
            idx = k - lag
            phi.append(U[idx, j] if idx >= 0 else 0.0)
    return np.asarray(phi, dtype=np.float64)


def recursive_predict(pipe, U_test, d, na, nb, y_mean, y_std, clip_sigma=4.0, n_out=2):
    """
    Closed-loop / free-run prediction with a fitted multi-output Pipeline.

    Output is clipped to `y_mean ± clip_sigma * y_std` per output, a soft
    guard against unbounded blow-up when the predictor drifts outside
    the training distribution (relevant for the rational form of NARX2).
    """
    N = len(U_test)
    n_inp = U_test.shape[1]
    warmup = max(d + na, nb)

    y_pred = np.zeros((N, n_out))
    lo = y_mean - clip_sigma * y_std
    hi = y_mean + clip_sigma * y_std

    for k in range(warmup, N - 1):
        phi = _build_phi(y_pred, U_test, k, d, na, nb, n_out, n_inp)
        yk = np.asarray(pipe.predict(phi.reshape(1, -1))).reshape(-1)
        y_pred[k + 1] = np.clip(yk, lo, hi)

    return y_pred


def one_step_predict(pipe, Y_test, U_test, d, na, nb, n_out=None):
    """
    One-step-ahead predictions: each phi(k) uses the true past outputs.

    Used for the in-fold CV inside the lag-selection grid (where we want
    to score the regressor itself, not the rollout) and as a sanity check
    against the recursive predictor (one-step error <= recursive error).
    """
    Y_test = np.asarray(Y_test)
    U_test = np.asarray(U_test)
    N = len(Y_test)
    if n_out is None:
        n_out = Y_test.shape[1]
    n_inp = U_test.shape[1]
    warmup = max(d + na, nb)

    Y_hat = np.zeros((N, n_out))

    for k in range(warmup, N - 1):
        phi = _build_phi(Y_test, U_test, k, d, na, nb, n_out, n_inp)
        yk = np.asarray(pipe.predict(phi.reshape(1, -1))).reshape(-1)
        Y_hat[k + 1] = yk

    return Y_hat
