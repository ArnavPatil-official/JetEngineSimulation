"""Phase 8 ML scaffolding (Track D3): MLX models M1/M3/M4, PyTorch twins for
parity, float64 scoring, non-dimensionalisation.

Infrastructure only. No model is trained on project data before gate G2.

Import rules (so the main ``.venv`` test suite never fails on import):
``nondim``, ``spec`` and ``weighting`` are numpy-only; ``score64`` imports
mlx lazily; ``models_torch`` imports torch at module level and never mlx;
``models_mlx``, ``train_mlx`` and ``parity`` import mlx at module level and
must only be imported inside the ``catjet-mlx`` env (envs/mlx/).
This ``__init__`` imports nothing.
"""
