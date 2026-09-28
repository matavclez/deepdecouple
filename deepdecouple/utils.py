"""Shared regression hybrid core (architecture + training).

Internal helpers used by CVHybrid and BOHybrid. Not part of the public API.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "None": nn.Identity,
}


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array; got shape {arr.shape}")
    return arr


def _as_1d_float(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    return arr


class _Standardizer:
    """Column-wise z-score fit on training data only."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "_Standardizer":
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted.")
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted.")
        return x * self.scale_ + self.mean_


def _hidden_to_specs(hidden_layers: Sequence[Tuple[int, str]]) -> Dict[str, Dict]:
    specs: Dict[str, Dict] = {}
    for i, (units, act) in enumerate(hidden_layers):
        specs[str(i)] = {"units": int(units), "activation": str(act)}
    return specs


def _build_mlp(
    in_features: int,
    mlp_layer_specs: Dict[str, Dict],
    dropout_rate: float,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_features

    def _key(k: str):
        return int(k) if str(k).isdigit() else k

    for k in sorted(mlp_layer_specs.keys(), key=_key):
        cfg = mlp_layer_specs[k]
        units = int(cfg["units"])
        act_name = cfg.get("activation", "ReLU")
        act_cls = _ACTIVATIONS.get(act_name, nn.ReLU)
        layers += [nn.Linear(prev, units), act_cls(), nn.Dropout(dropout_rate)]
        prev = units
    # Final neural output has no bias; a single global bias gamma is added outside.
    layers += [nn.Linear(prev, 1, bias=False)]
    return nn.Sequential(*layers)


class HybridNet(nn.Module):
    """Linear branch + MLP branch + Softmax mixing (+ optional BO orthogonalisation).

    Prediction:
        y_hat = gamma + w_l * y_l + w_n * y_n

    with (w_l, w_n) = softmax(u / tau), both positive and summing to 1.
    """

    def __init__(
        self,
        in_features: int,
        mlp_layer_specs: Dict[str, Dict],
        dropout_rate: float,
        tau: float,
        *,
        use_orthogonalization: bool,
    ):
        super().__init__()
        if tau == 0.0:
            tau = 1e-6
        self.tau = float(tau)
        self.use_orthogonalization = bool(use_orthogonalization)
        self._freeze_mode = False

        self.linear = nn.Linear(in_features, 1, bias=False)
        self.global_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.nonlinear = _build_mlp(in_features, mlp_layer_specs, dropout_rate)
        self.u = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

        # Global residualisation coefficients (training-set LS fit). Used for
        # validation / inference only; never recomputed from val/test/predict data.
        self.register_buffer("a0_global", torch.zeros(1, 1))
        self.register_buffer("a_global", torch.zeros(in_features, 1))
        self.projection_mode = "batch"

    def set_projection_mode(self, mode: str) -> None:
        if mode not in ("batch", "global"):
            raise ValueError(f"projection_mode must be 'batch' or 'global'; got {mode!r}")
        self.projection_mode = mode

    def set_global_residualizer(self, a0: torch.Tensor, a: torch.Tensor) -> None:
        self.a0_global = a0.detach()
        self.a_global = a.detach()

    def _orth_batchwise(self, x: torch.Tensor, y_n_raw: torch.Tensor) -> torch.Tensor:
        """Batch-wise QR orthogonalisation (training only).

        For mini-batch B with design A_B = [1_M, X_B] = Q_B R_B:
            y_n,B = z_B - Q_B Q_B^T z_B
        """
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        a_b = torch.cat([ones, x], dim=1)
        q, _ = torch.linalg.qr(a_b)
        y_n_proj = q @ (q.T @ y_n_raw)
        return y_n_raw - y_n_proj

    def _orth_global(self, x: torch.Tensor, y_n_raw: torch.Tensor) -> torch.Tensor:
        """Fixed global residualisation: y_n = f(X) - (a0 + X a)."""
        return y_n_raw - (self.a0_global + x @ self.a_global)

    def _apply_orthogonal_projection(self, x: torch.Tensor, y_n_raw: torch.Tensor) -> torch.Tensor:
        if not self.use_orthogonalization:
            return y_n_raw
        if self.projection_mode == "batch":
            return self._orth_batchwise(x, y_n_raw)
        return self._orth_global(x, y_n_raw)

    def weights_final(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w = F.softmax(self.u / self.tau, dim=0)
        return w[0], w[1]

    def components(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_l = self.linear(x)
        if self._freeze_mode:
            return y_l, torch.zeros_like(y_l)
        y_n_raw = self.nonlinear(x)
        y_n = self._apply_orthogonal_projection(x, y_n_raw)
        return y_l, y_n

    def combine(self, y_l: torch.Tensor, y_n: torch.Tensor) -> torch.Tensor:
        w_l, w_n = self.weights_final()
        return self.global_bias + w_l * y_l + w_n * y_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_l = self.linear(x)
        if self._freeze_mode:
            return y_l + self.global_bias
        y_n_raw = self.nonlinear(x)
        y_n = self._apply_orthogonal_projection(x, y_n_raw)
        return self.combine(y_l, y_n)


def fit_global_residualizer(
    model: HybridNet,
    x_train: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit of z_train ≈ a0 + X_train a on the full training set."""
    model.eval()
    with torch.no_grad():
        z_train = model.nonlinear(x_train)
        design = torch.cat(
            [torch.ones(x_train.size(0), 1, device=x_train.device, dtype=x_train.dtype), x_train],
            dim=1,
        )
        sol = torch.linalg.lstsq(design, z_train).solution
        a0 = sol[0:1, :]
        a = sol[1:, :]
    return a0, a


def _combine_pred(model: HybridNet, y_l: torch.Tensor, y_n: torch.Tensor, freeze_now: bool) -> torch.Tensor:
    if freeze_now:
        return y_l + model.global_bias
    return model.combine(y_l, y_n)


def train_hybrid(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    mlp_layer_specs: Dict[str, Dict],
    *,
    epochs: int,
    batch_size: int,
    dropout_rate: float,
    lr: float,
    patience_fraction: float,
    tau: float,
    freeze_nonlinear_epochs: int,
    use_orthogonalization: bool,
    random_state: int,
    verbose: bool,
) -> Tuple[HybridNet, Dict[str, List[float]]]:
    torch.manual_seed(random_state)
    device = x_train.device

    model = HybridNet(
        in_features=x_train.shape[1],
        mlp_layer_specs=mlp_layer_specs,
        dropout_rate=dropout_rate,
        tau=tau,
        use_orthogonalization=use_orthogonalization,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    patience = max(1, int(round(epochs * patience_fraction)))
    best_val = float("inf")
    trigger = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    hist_train: List[float] = []
    hist_val: List[float] = []

    for epoch in range(epochs):
        freeze_now = epoch < int(freeze_nonlinear_epochs)
        model._freeze_mode = freeze_now

        # Freeze / Wake-up: during freeze, only linear branch + gamma train.
        for name, param in model.named_parameters():
            if name.startswith("nonlinear") or name == "u":
                param.requires_grad = not freeze_now
            else:
                param.requires_grad = True

        model.set_projection_mode("batch")
        model.train()

        perm = torch.randperm(len(x_train), device=device)
        xe, ye = x_train[perm], y_train[perm]

        for i in range(0, len(xe), batch_size):
            xb = xe[i : i + batch_size]
            yb = ye[i : i + batch_size]
            optimizer.zero_grad()
            y_l, y_n = model.components(xb)
            y_hat = _combine_pred(model, y_l, y_n, freeze_now)
            loss = torch.mean((yb - y_hat) ** 2)
            loss.backward()
            optimizer.step()

        # After each epoch: fit global residualiser on FULL TRAIN only (BO),
        # then evaluate train/val under global residualisation.
        if use_orthogonalization and not freeze_now:
            a0_tmp, a_tmp = fit_global_residualizer(model, x_train)
            model.set_global_residualizer(a0_tmp, a_tmp)

        model.set_projection_mode("global")
        model.eval()
        with torch.no_grad():
            y_l_t, y_n_t = model.components(x_train)
            train_mse = float(torch.mean((_combine_pred(model, y_l_t, y_n_t, freeze_now) - y_train) ** 2).item())
            y_l_v, y_n_v = model.components(x_val)
            val_mse = float(torch.mean((_combine_pred(model, y_l_v, y_n_v, freeze_now) - y_val) ** 2).item())

        hist_train.append(train_mse)
        hist_val.append(val_mse)
        model.set_projection_mode("batch")

        if val_mse < best_val:
            best_val = val_mse
            trigger = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            trigger += 1
            if trigger >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model._freeze_mode = False
    if use_orthogonalization:
        a0_fin, a_fin = fit_global_residualizer(model, x_train)
        model.set_global_residualizer(a0_fin, a_fin)
    model.set_projection_mode("global")
    model.eval()

    history = {
        "epoch": list(range(1, len(hist_train) + 1)),
        "train_mse": hist_train,
        "val_mse": hist_val,
    }
    return model, history


class BaseHybridRegressor:
    """Shared fit / predict API for CV-Hybrid and BO-Hybrid."""

    def __init__(
        self,
        *,
        hidden_layers: Optional[Sequence[Tuple[int, str]]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        tau: float = 1.0,
        dropout: float = 0.0,
        freeze_nonlinear_epochs: int = 0,
        use_orthogonalization: bool = False,
        patience_fraction: float = 0.1,
        scale_features: bool = True,
        scale_target: bool = True,
        random_state: int = 0,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.hidden_layers = list(hidden_layers) if hidden_layers is not None else [(32, "ReLU"), (16, "ReLU")]
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.tau = float(tau)
        self.dropout = float(dropout)
        self.freeze_nonlinear_epochs = int(freeze_nonlinear_epochs)
        self.use_orthogonalization = bool(use_orthogonalization)
        self.patience_fraction = float(patience_fraction)
        self.scale_features = bool(scale_features)
        self.scale_target = bool(scale_target)
        self.random_state = int(random_state)
        self.device = device
        self.verbose = bool(verbose)

    def _resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "BaseHybridRegressor":
        Xt = _as_2d_float(X_train)
        Xv = _as_2d_float(X_val)
        yt = _as_1d_float(y_train)
        yv = _as_1d_float(y_val)

        if Xt.shape[0] != yt.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows.")
        if Xv.shape[0] != yv.shape[0]:
            raise ValueError("X_val and y_val must have the same number of rows.")
        if Xt.shape[1] != Xv.shape[1]:
            raise ValueError("X_train and X_val must have the same number of features.")

        self.n_features_in_ = Xt.shape[1]
        self.scaler_X_ = _Standardizer().fit(Xt) if self.scale_features else None
        self.scaler_y_ = _Standardizer().fit(yt.reshape(-1, 1)) if self.scale_target else None

        Xt_s = self.scaler_X_.transform(Xt) if self.scaler_X_ is not None else Xt
        Xv_s = self.scaler_X_.transform(Xv) if self.scaler_X_ is not None else Xv
        yt_s = self.scaler_y_.transform(yt.reshape(-1, 1)).ravel() if self.scaler_y_ is not None else yt
        yv_s = self.scaler_y_.transform(yv.reshape(-1, 1)).ravel() if self.scaler_y_ is not None else yv

        dev = self._resolve_device()
        x_tr = torch.tensor(Xt_s, dtype=torch.float32, device=dev)
        y_tr = torch.tensor(yt_s, dtype=torch.float32, device=dev).unsqueeze(1)
        x_va = torch.tensor(Xv_s, dtype=torch.float32, device=dev)
        y_va = torch.tensor(yv_s, dtype=torch.float32, device=dev).unsqueeze(1)

        model, history = train_hybrid(
            x_tr,
            y_tr,
            x_va,
            y_va,
            _hidden_to_specs(self.hidden_layers),
            epochs=self.epochs,
            batch_size=self.batch_size,
            dropout_rate=self.dropout,
            lr=self.lr,
            patience_fraction=self.patience_fraction,
            tau=self.tau,
            freeze_nonlinear_epochs=self.freeze_nonlinear_epochs,
            use_orthogonalization=self.use_orthogonalization,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.model_ = model
        self.device_ = dev
        self.train_history_ = history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before predict().")
        Xnp = _as_2d_float(X)
        if Xnp.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features; got {Xnp.shape[1]}."
            )
        Xs = self.scaler_X_.transform(Xnp) if self.scaler_X_ is not None else Xnp
        xt = torch.tensor(Xs, dtype=torch.float32, device=self.device_)

        # Inference always uses stored training residualiser coefficients (BO)
        # or identity (CV). Coefficients are never recomputed from X.
        self.model_.set_projection_mode("global")
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(xt).cpu().numpy().ravel()

        if self.scaler_y_ is not None:
            out = self.scaler_y_.inverse_transform(out.reshape(-1, 1)).ravel()
        return out

    def mixture_weights(self) -> Tuple[float, float]:
        """Return (w_l, w_n) after fitting."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before mixture_weights().")
        w_l, w_n = self.model_.weights_final()
        return float(w_l.detach().cpu()), float(w_n.detach().cpu())

    def residualizer_coefficients(self) -> Tuple[float, np.ndarray]:
        """Return stored global residualisation coefficients (a0, a).

        Available after fit when orthogonalisation is enabled (BO-Hybrid).
        These are estimated exclusively from the training set.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before residualizer_coefficients().")
        if not self.use_orthogonalization:
            raise RuntimeError("Global residualisation is only used by BO-Hybrid.")
        a0 = float(self.model_.a0_global.detach().cpu().numpy().ravel()[0])
        a = self.model_.a_global.detach().cpu().numpy().ravel().copy()
        return a0, a
