"""Minimal usage example for CV-Hybrid and BO-Hybrid on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from the repository root without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepdecouple import BOHybrid, CVHybrid


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    rng = np.random.default_rng(0)
    n, p = 400, 5
    X = rng.normal(size=(n, p))

    beta = np.array([1.5, -1.0, 0.5, 0.0, 0.0])
    linear = X @ beta
    nonlinear = 0.8 * np.sin(X[:, 0]) + 0.5 * (X[:, 1] ** 2)
    noise = rng.normal(scale=0.3, size=n)
    y = linear + nonlinear + noise

    n_train, n_val = 240, 80
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    common = dict(
        hidden_layers=[(32, "ReLU"), (16, "ReLU")],
        epochs=80,
        batch_size=32,
        lr=1e-3,
        tau=1.0,
        patience_fraction=0.2,
        random_state=0,
        verbose=False,
    )

    cv_model = CVHybrid(freeze_nonlinear_epochs=20, **common)
    cv_model.fit(X_train, y_train, X_val, y_val)
    cv_pred = cv_model.predict(X_test)

    bo_model = BOHybrid(**common)
    bo_model.fit(X_train, y_train, X_val, y_val)
    bo_pred = bo_model.predict(X_test)

    print(f"CV-Hybrid test RMSE: {rmse(y_test, cv_pred):.4f}")
    print(f"BO-Hybrid test RMSE: {rmse(y_test, bo_pred):.4f}")
    print(f"CV mixture weights (w_l, w_n): {cv_model.mixture_weights()}")
    print(f"BO mixture weights (w_l, w_n): {bo_model.mixture_weights()}")


if __name__ == "__main__":
    main()
