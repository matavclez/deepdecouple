"""DeepDecouple: demonstration code for CV-Hybrid and BO-Hybrid regression."""

from .bo_hybrid import BOHybrid
from .cv_hybrid import CVHybrid

__version__ = "0.1.0"

__all__ = ["CVHybrid", "BOHybrid", "__version__"]
