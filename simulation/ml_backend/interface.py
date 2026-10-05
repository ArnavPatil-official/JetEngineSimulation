"""Backend selection, scalar losses and dtype-preserving neutral checkpoints."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

ENVIRONMENT_VARIABLE = "CATJET_ML_BACKEND"


@dataclass(frozen=True)
class MLPSpec:
    in_dim: int
    hidden: tuple[int, ...]
    out_dim: int
    activation: str = "silu"

    def __post_init__(self):
        if self.activation not in ("silu", "tanh"):
            raise ValueError("MLP activation must be silu or tanh")
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0
               for size in (self.in_dim, *self.hidden, self.out_dim)):
            raise ValueError("MLP dimensions must be positive integers")

    @property
    def layer_dims(self):
        widths = (self.in_dim, *self.hidden, self.out_dim)
        return tuple(zip(widths[:-1], widths[1:]))


def resolve_backend(name=None):
    name = (name if name is not None else os.environ.get(ENVIRONMENT_VARIABLE, "torch")).strip().lower()
    if name not in ("torch", "mlx"):
        raise ValueError(f"Unknown ML backend {name!r}; use torch or mlx")
    return name


def get_backend(name=None, *, device="cpu", dtype=None):
    """Explicit name overrides CATJET_ML_BACKEND; default is Torch float64 CPU."""
    selected = resolve_backend(name)
    if selected == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend(device=device, dtype=dtype)
    from .mlx_backend import MLXBackend
    return MLXBackend(device=device, dtype=dtype)


def mse_loss(prediction, target, xp):
    """One shared scalar loss, with no conversion or graph detachment."""
    return xp.mean((prediction-target)**2)


def weighted_loss(terms, weights):
    if set(terms) != set(weights) or not terms:
        raise ValueError("Loss terms and weights must have the same nonempty keys")
    return sum(weights[name]*terms[name] for name in terms)


def seeded_parameters(spec, seed, dtype="float64"):
    """Same local NumPy seed family for either backend; no global RNG mutation."""
    import numpy as np
    rng = np.random.default_rng(int(seed))
    params = {}
    for i, (incoming, outgoing) in enumerate(spec.layer_dims):
        bound = 1/np.sqrt(incoming)
        params[f"layers.{i}.weight"] = rng.uniform(-bound, bound, (outgoing, incoming)).astype(dtype)
        params[f"layers.{i}.bias"] = rng.uniform(-bound, bound, outgoing).astype(dtype)
    return params


def validate_parameters(params, spec=None):
    """Reject missing/extra keys, broken dimensions, nonfloat or nonfinite data."""
    import numpy as np
    if not params:
        raise ValueError("Checkpoint contains no weights")
    count = len(params)//2
    expected = {f"layers.{i}.{kind}" for i in range(count) for kind in ("weight", "bias")}
    if set(params) != expected:
        raise ValueError("Weights must use the complete canonical layers.i.weight/bias layout")
    dimensions = []
    prior = None
    for i in range(count):
        weight, bias = (np.asarray(params[f"layers.{i}.{kind}"]) for kind in ("weight", "bias"))
        if weight.ndim != 2 or bias.shape != (weight.shape[0],) or 0 in weight.shape:
            raise ValueError("Invalid MLP layer dimensions")
        if prior is not None and weight.shape[1] != prior:
            raise ValueError("MLP layers do not connect")
        if any(value.dtype not in (np.dtype("float32"), np.dtype("float64"))
               or not np.isfinite(value).all() for value in (weight, bias)):
            raise ValueError("MLP weights must be finite float32 or float64")
        dimensions.append((weight.shape[1], weight.shape[0])); prior = weight.shape[0]
    if spec is not None and tuple(dimensions) != spec.layer_dims:
        raise ValueError("Checkpoint dimensions differ from the destination model")
    return dimensions


class BackendBase:
    """Common native-model helpers; concrete backends own differentiation/updates."""
    @property
    def backend_name(self):
        return self.name

    @property
    def xp(self):
        return self.ops

    def forward(self, model, x):
        return model(self.array(x))

    def mse(self, prediction, target):
        return mse_loss(prediction, target, self.ops)

    def _set_parameters(self, model, params):
        return self.set_parameters(model, params)

    def score_numpy64(self, model, x):
        from .numpy_reference import forward
        return forward({name: self.to_numpy(value) for name, value in self.parameters(model).items()},
                       x, activation=model.spec.activation)

    def score64(self, model_or_params, x, *, activation=None):
        """Independent CPU64 scoring used by the MLX backend."""
        from .numpy_reference import forward
        if isinstance(model_or_params, dict) and "layers.0.weight" in model_or_params:
            params = {name: self.to_numpy(value) for name, value in model_or_params.items()}
            selected_activation = activation or "silu"
        else:
            model = model_or_params
            params = {name: self.to_numpy(value) for name, value in self.parameters(model).items()}
            selected_activation = activation or model.spec.activation
        return forward(params, x, activation=selected_activation)

    def save_npz(self, model, path, *, metadata=None):
        """Exclusive-create neutral NPZ; float64 Torch weights stay float64."""
        import numpy as np
        path = Path(path)
        params = {name: self.to_numpy(value).copy() for name, value in self.parameters(model).items()}
        validate_parameters(params, model.spec)
        record = {"schema": 1, "activation": model.spec.activation,
                  "training_backend": self.name, "dtype": str(next(iter(params.values())).dtype),
                  "metadata": metadata or {}}
        payload = {**params, "__metadata__": np.asarray(json.dumps(record, sort_keys=True, allow_nan=False))}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez(stream, **payload); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load_npz(self, path, model=None, *, activation=None):
        import numpy as np
        with np.load(path, allow_pickle=False) as archive:
            params = {name: archive[name].copy() for name in archive.files if name.startswith("layers.")}
            foreign = set(archive.files)-set(params)-{"__metadata__"}
            if foreign:
                raise ValueError("Unknown neutral checkpoint entries")
            record = json.loads(str(archive["__metadata__"].item())) if "__metadata__" in archive else {}
        dimensions = validate_parameters(params)
        stored_activation = record.get("activation")
        if activation is not None and stored_activation is not None and activation != stored_activation:
            raise ValueError("Checkpoint activation differs from requested activation")
        activation = activation or stored_activation or (model.spec.activation if model is not None else "silu")
        if model is None:
            model = self.mlp(dimensions[0][0], tuple(pair[1] for pair in dimensions[:-1]),
                             dimensions[-1][1], activation=activation, seed=0)
        elif activation != model.spec.activation:
            raise ValueError("Checkpoint activation differs from destination model")
        return self.set_parameters(model, params)
