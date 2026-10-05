"""Lazy, portable ML backends with neutral weights and NumPy64 scoring."""
from .interface import MLPSpec, get_backend, resolve_backend, mse_loss, weighted_loss

__all__ = ["MLPSpec", "get_backend", "resolve_backend", "mse_loss", "weighted_loss"]
