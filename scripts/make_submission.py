"""Reproducible submission script.

Regenerates `submission.npz` from `data/StudentdataNARX.npz` without
opening the notebook. Same model and lag selection as the notebook.

Run from the repo root:

    python scripts/make_submission.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def _add_src_to_path():
    # Script lives in repo_root/scripts/ so the parent is the repo root.
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    if not (repo_root / "src").is_dir() or not (repo_root / "data").is_dir():
        raise SystemExit(f"expected src/ and data/ under {repo_root}")
    sys.path.insert(0, str(repo_root / "src"))
    return repo_root


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    root = _add_src_to_path()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path,
                        default=root / "data" / "StudentdataNARX.npz")
    parser.add_argument("--out", type=pathlib.Path,
                        default=root / "submission.npz")
    parser.add_argument("--tail", type=int, default=200,
                        help="Held-out tail size (last N steps of Ytr) used "
                              "to pick the DIRMO block size before refit.")
    args = parser.parse_args()

    from narx.lag_selection import grid_search
    from narx.dirmo import compare_block_sizes, fit_dirmo, predict_dirmo

    print(f"loading {args.data}")
    data = np.load(args.data)
    Utr  = data["Utr"].astype(np.float64)
    Ytr  = data["Ytr"].astype(np.float64)
    Uts1 = data["Uts1"].astype(np.float64)
    Uts2 = data["Uts2"].astype(np.float64)

    print("step 1: lag selection by multi-proxy CV grid search")
    best_params, _ = grid_search(
        Ytr, Utr,
        d_range=[0, 1], na_range=[0, 1, 2, 3], nb_range=[0, 1, 2, 3],
        label="REAL", verbose=False, n_splits=3,
    )
    d, na, nb = best_params
    print(f"  -> d={d}, na={na}, nb={nb}")

    print(f"step 2: DIRMO block-size sweep on held-out tail (last {args.tail})")
    Ytr_part, Y_tail = Ytr[:-args.tail], Ytr[-args.tail:]
    Utr_part, U_tail = Utr[:-args.tail], Utr[-args.tail:]
    base_gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    block_sizes = (1, 2, 5, 10, 20, 40, 80)
    results = compare_block_sizes(
        Ytr_part, Utr_part, Y_tail, U_tail,
        d=d, na=na, nb=nb, base_model=base_gb,
        block_sizes=block_sizes,
    )
    best_s = min(results, key=lambda s: results[s]["nmse"])
    print(f"  -> best block size: s={best_s} "
           f"(NMSE={results[best_s]['nmse']:.3f})")

    print("step 3: refit on the full training set, predict Yhat1 and Yhat2")
    mor, sc, y_mean, y_std = fit_dirmo(
        Ytr, Utr, d, na, nb,
        block_size=best_s, base_model=base_gb, n_out=2,
    )
    Yhat1 = predict_dirmo(mor, sc, Uts1, d, na, nb,
                           block_size=best_s, y_mean=y_mean, y_std=y_std)
    Yhat2 = predict_dirmo(mor, sc, Uts2, d, na, nb,
                           block_size=best_s, y_mean=y_mean, y_std=y_std)

    print(f"step 4: writing {args.out}")
    np.savez(args.out, Yhat1=Yhat1, Yhat2=Yhat2)

    loaded = np.load(args.out)
    for key in ("Yhat1", "Yhat2"):
        a = loaded[key]
        print(f"  {key}: shape={a.shape}, dtype={a.dtype}, "
               f"finite={bool(np.isfinite(a).all())}")


if __name__ == "__main__":
    main()
