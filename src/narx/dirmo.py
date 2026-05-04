# DIRMO multi-step strategy. Bontempi et al. 2013, sec. 4.3.
# Split horizon H into ceil(H/s) blocks of size s; one multi-output
# regressor per block predicts y(k+1)..y(k+s) from phi(k_b). Errors
# only propagate across block boundaries, not inside a block.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor


def build_dirmo_dataset(Y, U, d, na, nb, block_size, n_out=2):
    "phi(k) -> [y_1(k+1)..y_1(k+s), y_2(k+1)..y_2(k+s)] (output-major)"
    N = len(Y)
    n_inp = U.shape[1]
    warmup = max(d + na, nb)

    Xs, Ts = [], []
    for k in range(warmup, N - block_size):
        # phi: same NARX regressor as features.independent_matrix
        phi = [Y[k - lag, j] for j in range(n_out) for lag in range(d, d + na + 1)]
        phi += [U[k - lag, j] for j in range(n_inp) for lag in range(0, nb + 1)]
        # block target, output-major so MultiOutputRegressor can fit per scalar
        tgt = [Y[k + h, j] for j in range(n_out) for h in range(1, block_size + 1)]
        Xs.append(phi)
        Ts.append(tgt)

    return np.asarray(Xs, dtype=np.float64), np.asarray(Ts, dtype=np.float64)


def fit_dirmo(Y_tr, U_tr, d, na, nb, block_size, base_model, n_out=2):
    X, Yt = build_dirmo_dataset(Y_tr, U_tr, d, na, nb, block_size, n_out)
    sc = StandardScaler().fit(X)
    mor = MultiOutputRegressor(clone(base_model), n_jobs=-1)
    mor.fit(sc.transform(X), Yt)
    return mor, sc, Y_tr.mean(0), Y_tr.std(0)


def predict_dirmo(
    mor, scaler, U_test, d, na, nb, block_size, y_mean, y_std, clip_sigma=4.0, n_out=2
):
    "closed-loop rollout. y_hat[0..warmup] stays 0 (brief: y(<=0)=0)."
    Nts = len(U_test)
    n_inp = U_test.shape[1]
    warmup = max(d + na, nb)
    Y_hat = np.zeros((Nts, n_out))
    lo, hi = y_mean - clip_sigma * y_std, y_mean + clip_sigma * y_std

    k = warmup
    while k < Nts:
        phi = []
        for j in range(n_out):
            for lag in range(d, d + na + 1):
                idx = k - lag
                phi.append(Y_hat[idx, j] if idx >= 0 else 0.0)
        for j in range(n_inp):
            for lag in range(0, nb + 1):
                idx = k - lag
                phi.append(U_test[idx, j] if idx >= 0 else 0.0)

        block = mor.predict(scaler.transform(np.array(phi).reshape(1, -1)))[0]
        # block is (s*n_out,), output-major
        for h in range(1, block_size + 1):
            tk = k + h
            if tk >= Nts:
                break
            for j in range(n_out):
                Y_hat[tk, j] = np.clip(block[j * block_size + (h - 1)], lo[j], hi[j])
        k += block_size

    return Y_hat


def compare_block_sizes(
    Y_tr,
    U_tr,
    Y_ts,
    U_ts,
    d,
    na,
    nb,
    base_model,
    block_sizes=(1, 2, 5, 10, 20, 50),
    n_out=2,
    clip_sigma=4.0,
):
    """Sweep block_size, fit DIRMO at each s, score on the (Y_ts, U_ts) pair.

    For each s in `block_sizes` we (i) call fit_dirmo on the training pair,
    (ii) call predict_dirmo on the test inputs, (iii) score the rollout vs
    Y_ts with the project's RMSE / NMSE. Returned dict maps s to a sub-dict
    {'rmse', 'nmse', 'Y_hat'}; the table is printed as we go.

    Used both on the pilots (where Y_ts comes from a continuation of the
    same simulation) and on the deployment held-out tail of Ytr (where
    Y_ts is the last 20% of the real training trajectory). In both cases
    the smallest s with NMSE near 1 marks the boundary at which the block
    model can no longer predict s steps ahead from one phi.
    """
    from .metrics import rmse, nmse

    out = {}
    print(f"\n{'s':>4} {'RMSE':>10} {'NMSE':>10}  blocks")
    print("-" * 40)
    for s in block_sizes:
        mor, sc, ym, ys = fit_dirmo(
            Y_tr, U_tr, d, na, nb, block_size=s, base_model=base_model, n_out=n_out
        )
        Yh = predict_dirmo(
            mor,
            sc,
            U_ts,
            d,
            na,
            nb,
            block_size=s,
            y_mean=ym,
            y_std=ys,
            clip_sigma=clip_sigma,
            n_out=n_out,
        )
        r, n = rmse(Y_ts, Yh), nmse(Y_ts, Yh)
        nb_blocks = int(np.ceil(len(U_ts) / s))
        out[s] = {"rmse": r, "nmse": n, "Y_hat": Yh}
        print(f" s={s:>3}  RMSE={r:>7.4f}  NMSE={n:>7.4f}  {nb_blocks:>4}")
    return out


def plot_dirmo_results(results, Y_ts, block_sizes, label=""):
    "NMSE-vs-s curve + best/worst overlay panels."
    ss = [s for s in block_sizes if s in results]
    ns = [results[s]["nmse"] for s in ss]
    best, worst = ss[int(np.argmin(ns))], ss[int(np.argmax(ns))]

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[:, 0])
    ax.plot(ss, ns, "o-", color="steelblue", lw=1.5, ms=6)
    ax.axhline(1.0, ls="--", color="red", alpha=0.6, label="NMSE=1 (mean)")
    ax.axvline(best, ls=":", color="green", alpha=0.7, label=f"best s={best}")
    ax.set_xlabel("block size s")
    ax.set_ylabel("NMSE")
    ax.set_title(f"{label}\nNMSE vs DIRMO block size")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for j in (0, 1):
        a = fig.add_subplot(gs[j, 1])
        a.plot(Y_ts[:, j], color="black", lw=0.8, label="true")
        a.plot(
            results[best]["Y_hat"][:, j],
            color="steelblue",
            lw=0.8,
            alpha=0.85,
            label=f"s={best}",
        )
        a.set_title(f"y{j + 1} (best s={best}, NMSE={results[best]['nmse']:.3f})")
        a.set_xlabel("k")
        a.legend(fontsize=7)
        a.grid(alpha=0.25)

        a = fig.add_subplot(gs[j, 2])
        a.plot(Y_ts[:, j], color="black", lw=0.8, label="true")
        a.plot(
            results[worst]["Y_hat"][:, j],
            color="crimson",
            lw=0.8,
            alpha=0.85,
            label=f"s={worst}",
        )
        a.set_title(f"y{j + 1} (worst s={worst}, NMSE={results[worst]['nmse']:.3f})")
        a.set_xlabel("k")
        a.legend(fontsize=7)
        a.grid(alpha=0.25)

    plt.suptitle(f"DIRMO {label}")
    plt.tight_layout()
    return fig
