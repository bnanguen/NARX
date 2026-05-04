"""PyTorch MLP for task 2d."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

from .metrics import nmse


def _to_tensor(a, device):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NarxMLP(BaseEstimator, RegressorMixin):
    """
    Feed-forward MLP regressor

    Parameters
    ----------
    hidden : tuple of int
        Hidden-layer widths.
    dropout : float
        Dropout probability applied after every hidden layer.
    lr, weight_decay : Adam hyperparameters.
    batch_size, max_epochs : training schedule.
    val_frac : fraction of the (chronologically last) training data held
        out for early stopping. With time-series data we never shuffle.
    patience : how many epochs of no val-NMSE improvement before stopping.
    seed : reproducibility.
    verbose : print epoch / val NMSE.
    """

    def __init__(
        self,
        hidden=(64, 32),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=64,
        max_epochs=200,
        val_frac=0.2,
        patience=20,
        seed=0,
        verbose=False,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.val_frac = val_frac
        self.patience = patience
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_val = max(1, int(len(X) * self.val_frac))
        # Chronological split, never shuffle a time series.
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        device = torch.device("cpu")
        net = _MLP(X.shape[1], y.shape[1], hidden=self.hidden, dropout=self.dropout).to(
            device
        )
        opt = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        # Pre-load tensors once.
        Xt = _to_tensor(X_tr, device)
        yt = _to_tensor(y_tr, device)
        Xv = _to_tensor(X_val, device)
        yv_np = y_val

        best_nmse = float("inf")
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        epochs_no_improve = 0
        history = []

        n_train = len(Xt)
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                out = net(Xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()

            net.eval()
            with torch.no_grad():
                yv_pred = net(Xv).cpu().numpy()
            val_n = nmse(yv_np, yv_pred)
            history.append(val_n)

            improved = val_n < best_nmse - 1e-6
            if improved:
                best_nmse = val_n
                best_state = {
                    k: v.detach().clone() for k, v in net.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if self.verbose and (epoch < 5 or epoch % 10 == 0):
                print(
                    f"  epoch {epoch:3d}  val NMSE={val_n:.4f}"
                    f"{'  *' if improved else ''}"
                )

            if epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"  early stop at epoch {epoch} (best NMSE={best_nmse:.4f})")
                break

        net.load_state_dict(best_state)
        self.net_ = net
        self.in_dim_ = X.shape[1]
        self.out_dim_ = y.shape[1]
        self.best_val_nmse_ = best_nmse
        self.history_ = history
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.as_tensor(X)).cpu().numpy()
        if self.out_dim_ == 1:
            return out.ravel()
        return out


def make_mlp_pipeline(**mlp_kwargs):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", NarxMLP(**mlp_kwargs)),
        ]
    )
