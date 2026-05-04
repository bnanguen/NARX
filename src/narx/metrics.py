"""RMSE and NMSE per the formulas in the project brief."""

import numpy as np


def rmse(y_true, y_hat):
    """
    Root mean squared error, averaged over the two outputs.

        RMSE = sqrt( 1/(2*N_ts) * sum_k sum_j (y_j(k) - yhat_j(k))^2 )

    Matches eq. (4) of the brief.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))


def nmse(y_true, y_hat):
    """
    Normalised mean squared error, joint over the two outputs.

        NMSE = sum_k sum_j (y_j(k) - yhat_j(k))^2  /  sum_k sum_j (y_j(k) - ybar_j)^2

    where ybar_j is the per-output mean. Matches eq. (5) of the brief.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    num = np.sum((y_true - y_hat) ** 2)
    denom = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    return float(num / (denom + 1e-12))
