"""BO-Hybrid: CV-Hybrid structure with Batch-wise Orthogonalisation."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .utils import BaseHybridRegressor


class BOHybrid(BaseHybridRegressor):
    """Batch-wise Orthogonal Hybrid (BO-Hybrid).

    Same architecture as CV-Hybrid, with neural contributions orthogonalised
    with respect to the linear design space ``[1, X]``.

    Training uses batch-wise QR orthogonalisation on each mini-batch.
    Validation and inference use fixed global residualisation coefficients
    ``(a0, a)`` estimated by least squares from the complete training set only.
    Those coefficients are stored with the model and are never recomputed from
    validation, test, or arbitrary prediction batches.

    Freeze / Wake-up is not used by default (``freeze_nonlinear_epochs=0``).
    """

    def __init__(
        self,
        hidden_layers: Optional[Sequence[Tuple[int, str]]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        tau: float = 1.0,
        dropout: float = 0.0,
        freeze_nonlinear_epochs: int = 0,
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
            use_orthogonalization=True,
            patience_fraction=patience_fraction,
            scale_features=scale_features,
            scale_target=scale_target,
            random_state=random_state,
            device=device,
            verbose=verbose,
        )
