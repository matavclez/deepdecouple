"""CV-Hybrid: convex Softmax mixing of linear and neural branches."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .utils import BaseHybridRegressor


class CVHybrid(BaseHybridRegressor):
    """Convex-combination Hybrid (CV-Hybrid).

    Explicit linear branch (no bias) + MLP branch (final layer without bias) +
    a single global bias ``gamma``, mixed by Softmax weights ``(w_l, w_n)``:

        y_hat = gamma + w_l * y_l + w_n * y_n

    Freeze / Wake-up is available via ``freeze_nonlinear_epochs``: for the first
    ``freeze_nonlinear_epochs`` epochs only the linear branch and ``gamma`` are
    trained; afterwards the neural branch and mixture logits are activated.
    """

    def __init__(
        self,
        hidden_layers: Optional[Sequence[Tuple[int, str]]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        tau: float = 1.0,
        dropout: float = 0.0,
        freeze_nonlinear_epochs: int = 50,
        patience_fraction: float = 0.1,
        scale_features: bool = True,
        scale_target: bool = True,
        random_state: int = 0,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(
            hidden_layers=hidden_layers,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            tau=tau,
            dropout=dropout,
            freeze_nonlinear_epochs=freeze_nonlinear_epochs,
            use_orthogonalization=False,
            patience_fraction=patience_fraction,
            scale_features=scale_features,
            scale_target=scale_target,
            random_state=random_state,
            device=device,
            verbose=verbose,
        )
