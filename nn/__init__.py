"""Public API for the NumPy neural-network library."""

from nn.activations import ReLU, Sigmoid, Softmax, Tanh
from nn.layers import Dense, Dropout, Layer
from nn.losses import BinaryCrossEntropy, CategoricalCrossEntropy, Loss, MSE
from nn.model import Sequential
from nn.optimizers import Adam, Momentum, Optimizer, RMSProp, SGD

__version__ = "0.1.0"

__all__ = [
    "Adam",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "Dense",
    "Dropout",
    "Layer",
    "Loss",
    "MSE",
    "Momentum",
    "Optimizer",
    "RMSProp",
    "ReLU",
    "SGD",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "__version__",
]
