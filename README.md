# DeepDecouple

**Interpretable neural networks by decoupling structured and flexible effects.**

DeepDecouple provides demonstration code accompanying the paper:

*Interpretable Neural Networks by Decoupling Linear and Non-linear Effects*

This repository currently contains the regression implementations of CV-Hybrid and BO-Hybrid described in the paper.

## Overview

CV-Hybrid combines an explicit linear branch with a flexible neural branch through learned convex mixing weights. Predictions take the form `y_hat = gamma + w_l * y_l + w_n * y_n`, where `(w_l, w_n)` are Softmax weights and `gamma` is a single global bias.

BO-Hybrid additionally orthogonalises the neural contribution with respect to the linear design space during training, preventing linearly explainable variation from being represented through the neural contribution. Training uses batch-wise QR orthogonalisation, whereas validation and inference use fixed global residualisation estimated from the complete training set.

## Installation

```bash
git clone https://github.com/matavclez/deepdecouple.git
cd deepdecouple
pip install -r requirements.txt
```

## Quick start

```python
from deepdecouple import BOHybrid
import numpy as np

# X_train, y_train, X_val, y_val, X_test prepared by the user
model = BOHybrid(hidden_layers=[(32, "ReLU"), (16, "ReLU")], epochs=100, tau=1.0)
model.fit(X_train, y_train, X_val, y_val)
pred = model.predict(X_test)
```

A runnable synthetic-data script is available at `examples/minimal_example.py`:

```bash
python examples/minimal_example.py
```

## Models

### CV-Hybrid

Explicit linear branch (no bias) plus an MLP branch (final layer without bias), combined by Softmax mixing weights with a single global bias. Optional Freeze / Wake-up trains the linear branch first before activating the neural branch and mixture logits.

### BO-Hybrid

Same architecture as CV-Hybrid, with batch-wise QR orthogonalisation of the neural contribution during training and fixed training-set global residualisation for validation and inference. Freeze / Wake-up is not used by default.

## Reproducibility

This repository provides the core model implementations and a minimal usage example. The full experimental pipeline used to generate the results reported in the paper is not included in this demonstration repository.

## Citation

If you use DeepDecouple in academic work, please cite:

Interpretable Neural Networks by Decoupling Linear and Non-linear Effects

Matías L. Avila, Andrés M. Alonso, Daniel Peña

Neurocomputing, manuscript under revision.

## Authors

- Matías L. Avila
- Andrés M. Alonso
- Daniel Peña

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
