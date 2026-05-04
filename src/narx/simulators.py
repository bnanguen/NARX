"""Pilot NARX simulators, eq. (5) and (6) of the brief.

We simulate N_train + N_test + 20 steps and drop the first 20 (burn-in)
so the visible series doesn't show the cold-start transient.
"""

import numpy as np


def simulate_narx1(N_train, N_test, sigma=0.05, seed=42):
    """
    NARX1: single exogenous input u, two coupled outputs y1, y2.

        y1(k+1) = 0.5*y2(k-1) + sin(y2(k)) + 0.3*u(k-1) + w1(k+1)
        y2(k+1) = 0.5*y1(k-1) + sin(y1(k)) + 0.2*u(k)   + w2(k+1)

    True structural parameters: d=0, na=1, nb=1.
    The cross-coupling through sin(.) makes this mildly nonlinear, so a
    plain linear regressor will underfit; gradient boosting with depth >= 2
    recovers the structure cleanly.

    The single input is zero-padded to a (N, 2) array so all downstream
    code (which assumes two-input NARXs) sees a uniform shape.
    """
    rng = np.random.default_rng(seed)
    N = N_train + N_test + 20  # +20 burn-in steps
    u = rng.uniform(-1, 1, N)
    y = np.zeros((N, 2))
    w = rng.normal(0, sigma, (N, 2))

    for k in range(2, N - 1):
        y[k + 1, 0] = 0.5 * y[k - 1, 1] + np.sin(y[k, 1]) + 0.3 * u[k - 1] + w[k + 1, 0]
        y[k + 1, 1] = 0.5 * y[k - 1, 0] + np.sin(y[k, 0]) + 0.2 * u[k] + w[k + 1, 1]

    U_all = np.column_stack([u, np.zeros_like(u)])

    return (
        U_all[20 : 20 + N_train],
        y[20 : 20 + N_train],
        U_all[20 + N_train : 20 + N_train + N_test],
        y[20 + N_train : 20 + N_train + N_test],
    )


# Reading off eq. (5): both y1(k+1) and y2(k+1) need y at lags 0 and 1
# (so d=0, na=1) and u at lags 0 and 1 (nb=1). The lag-selection grid in
# §2a should land here.
NARX1_TRUE_PARAMS = dict(d=0, na=1, nb=1)


def simulate_narx2(N_train, N_test, sigma=0.01, seed=7):
    """
    NARX2: two exogenous inputs, two outputs, rational nonlinearity.

        y1(k+1) =  y1(k)*y1(k-1)*y1(k-2)*(y1(k-2)-1)*u2(k-1) + u2(k)
                   ----------------------------------------------------- + w1
                            1 + y2(k-1)^2 + y2(k-2)^2

        y2(k+1) =  y2(k)*y2(k-1)*y2(k-2)*(y2(k-2)-1)*u1(k-1) + u1(k)
                   ----------------------------------------------------- + w2
                            1 + y1(k-1)^2 + y1(k-2)^2

    True structural parameters: d=0, na=2, nb=1.
    Three output lags are needed and the rational form is sensitive to
    large outputs, so we squeeze the input range to [-0.5, 0.5].
    The cubic numerator means a linear proxy cannot identify the lags
    reliably; the lag-selection proxies in §2a have to be nonlinear.
    """
    rng = np.random.default_rng(seed)
    N = N_train + N_test + 20
    U = rng.uniform(-0.5, 0.5, (N, 2))
    Y = np.zeros((N, 2))
    w = rng.normal(0, sigma, (N, 2))

    for k in range(3, N - 1):
        d1 = 1 + Y[k - 1, 1] ** 2 + Y[k - 2, 1] ** 2
        n1 = (
            Y[k, 0] * Y[k - 1, 0] * Y[k - 2, 0] * (Y[k - 2, 0] - 1) * U[k - 1, 1]
            + U[k, 1]
        )
        Y[k + 1, 0] = n1 / d1 + w[k + 1, 0]

        d2 = 1 + Y[k - 1, 0] ** 2 + Y[k - 2, 0] ** 2
        n2 = (
            Y[k, 1] * Y[k - 1, 1] * Y[k - 2, 1] * (Y[k - 2, 1] - 1) * U[k - 1, 0]
            + U[k, 0]
        )
        Y[k + 1, 1] = n2 / d2 + w[k + 1, 1]

    return (
        U[20 : 20 + N_train],
        Y[20 : 20 + N_train],
        U[20 + N_train : 20 + N_train + N_test],
        Y[20 + N_train : 20 + N_train + N_test],
    )


# Reading off eq. (6): y reaches back to lag 2 (d=0, na=2) and u to lag 1 (nb=1).
NARX2_TRUE_PARAMS = dict(d=0, na=2, nb=1)

# Note: when we run the 2a grid on NARX2 it actually lands on (1, 1, 0).
# The rational form is close to linear in the |y| < 0.5 regime so the
# proxies prefer the smaller triple. Documented in the notebook.
