"""RMSE per the project brief."""

import numpy as np


def rmse(y_true, y_hat):
    """
    Root mean squared error, averaged over the two outputs.

        RMSE = sqrt( 1/(2*N_ts) * sum_k sum_j (y_j(k) - yhat_j(k))^2 )

    Matches eq. (4) of the brief.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_hat  = np.asarray(y_hat,  dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
