"""one-step and recursive (closed-loop) predictors + rollout eval.

`one_step_predict` feeds the true past y; `recursive_predict` feeds its
own predictions, which is what the brief grades. Both expect a fitted
sklearn-style `pipe` whose `.predict(phi_row)` returns shape (1, n_out).
First warmup steps left at 0 (matches y(<=0) = 0 in the brief).
"""

import numpy as np

from .metrics import nmse, rmse


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


def evaluate_rollout(Y_true, Y_hat, label=""):
    """
    Print RMSE / NMSE and produce a 3-panel diagnostic figure (true vs
    predicted for each output, plus the cumulative squared error over k).

    The cumulative-error panel is the most useful one: linear growth means
    the prediction noise is bounded; super-linear growth flags drift /
    instability in the closed-loop predictor.
    """
    import matplotlib.pyplot as plt

    r = rmse(Y_true, Y_hat)
    n = nmse(Y_true, Y_hat)
    print(f"  {label}  RMSE={r:.4f}  NMSE={n:.4f}")

    sq_err = ((Y_true - Y_hat) ** 2).mean(axis=1)
    cum_err = np.cumsum(sq_err)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{label}  |  RMSE={r:.4f}  NMSE={n:.4f}")

    for j in range(Y_true.shape[1]):
        axes[j].plot(Y_true[:, j], label="true", color="black", lw=0.9)
        axes[j].plot(
            Y_hat[:, j], label="predicted", color="steelblue", lw=0.9, alpha=0.85
        )
        axes[j].set_title(f"y{j + 1}")
        axes[j].set_xlabel("k")
        axes[j].legend(fontsize=8)

    axes[2].plot(cum_err, color="crimson", lw=1.2)
    axes[2].set_title(
        "Cumulative squared error\n(linear = bounded, super-linear = diverging)"
    )
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Cumulative MSE")

    plt.tight_layout()
    return r, n, fig
