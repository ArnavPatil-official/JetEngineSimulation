"""Native MLX float32 training and graph-preserving derivatives.

Exported parameters are scored separately with the independent NumPy64 model.
Imports remain lazy so the portable Torch route does not require MLX.
"""
from __future__ import annotations

from .interface import BackendBase, MLPSpec, seeded_parameters, validate_parameters


class MLXOps:
    def __init__(self, mx, nn):
        self.mx, self.nn = mx, nn
        self.float32, self.float64 = mx.float32, None
        self.training_dtype = mx.float32

    def array(self, value, dtype=None):
        import numpy as np
        dtype = self.mx.float32 if dtype is None else dtype
        if isinstance(dtype, (str, type, np.dtype)):
            names = {"float32": self.mx.float32, "int32": self.mx.int32,
                     "int64": self.mx.int64, "bool": self.mx.bool_}
            name = np.dtype(dtype).name
            if name not in names:
                raise ValueError("MLX arrays support float32 training; use NumPy for float64 scoring")
            dtype = names[name]
        if isinstance(value, self.mx.array):
            return value if value.dtype == dtype else value.astype(dtype)
        # NumPy64 values must be narrowed before MLX constructs an array.
        if isinstance(value, np.ndarray) and value.dtype == np.float64:
            value = value.astype(np.float32)
        return self.mx.array(value, dtype=dtype)

    asarray = array

    def mean(self, value, axis=None, keepdims=False): return self.mx.mean(value, axis=axis, keepdims=keepdims)
    def sum(self, value, axis=None, keepdims=False): return self.mx.sum(value, axis=axis, keepdims=keepdims)
    def max(self, value, axis=None, keepdims=False): return self.mx.max(value, axis=axis, keepdims=keepdims)
    def min(self, value, axis=None, keepdims=False): return self.mx.min(value, axis=axis, keepdims=keepdims)
    def exp(self, value): return self.mx.exp(value)
    def log(self, value): return self.mx.log(value)
    def tanh(self, value): return self.mx.tanh(value)
    def sqrt(self, value): return self.mx.sqrt(value)
    def abs(self, value): return self.mx.abs(value)
    def sigmoid(self, value): return self.mx.sigmoid(value)
    def softplus(self, value): return self.mx.logaddexp(value, 0)
    def zeros_like(self, value): return self.mx.zeros_like(value)
    def ones_like(self, value): return self.mx.ones_like(value)
    def isfinite(self, value): return self.mx.isfinite(value)
    def where(self, condition, a, b): return self.mx.where(condition, a, b)
    def maximum(self, a, b): return self.mx.maximum(a, b)
    def clip(self, value, a_min, a_max): return self.mx.clip(value, a_min, a_max)
    def stack(self, values, axis=0, dim=None): return self.mx.stack(values, axis=axis if dim is None else dim)
    def concatenate(self, values, axis=0): return self.mx.concatenate(values, axis=axis)
    def zeros(self, shape, dtype=None): return self.mx.zeros(shape, dtype=self.mx.float32 if dtype is None else dtype)
    def stop_gradient(self, value): return self.mx.stop_gradient(value)


class MLXBackend(BackendBase):
    name = "mlx"

    def __init__(self, *, device="cpu", dtype=None):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        if dtype is not None and dtype != mx.float32:
            try:
                valid_dtype = np.dtype(dtype).name == "float32"
            except TypeError:
                valid_dtype = False
            if not valid_dtype:
                raise ValueError("MLX trains in float32; NumPy CPU scoring uses float64")
        if device == "auto":
            device = "gpu" if mx.metal.is_available() else "cpu"
        if device not in ("cpu", "gpu"):
            raise ValueError("MLX supports cpu or gpu devices")
        if device == "gpu" and not mx.metal.is_available():
            raise RuntimeError("MLX GPU requested but Metal is unavailable")
        mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
        self.mx, self.nn, self.device, self.dtype = mx, nn, device, mx.float32
        self.ops = MLXOps(mx, nn)

    def array(self, value, dtype=None, *, requires_grad=False):
        return self.ops.array(value, dtype)

    def seed(self, seed):
        """Explicitly seed NumPy and native MLX random operations."""
        import numpy as np
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        np.random.seed(seed)
        self.mx.random.seed(seed)
        return seed

    def to_numpy(self, value):
        import numpy as np
        if isinstance(value, self.mx.array):
            self.mx.eval(value)
        return np.asarray(value).copy()

    def mlp(self, in_dim, hidden, out_dim, *, activation="silu", seed=0):
        mx, nn = self.mx, self.nn
        spec = MLPSpec(int(in_dim), tuple(hidden), int(out_dim), activation)
        params = seeded_parameters(spec, seed, "float32")
        class Dense(nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight, self.bias = mx.array(weight), mx.array(bias)

            def __call__(self, x):
                return x @ self.weight.T + self.bias

        class NativeMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.spec = spec
                # Direct parameters avoid consuming global MLX random state.
                self.layers = [Dense(params[f"layers.{i}.weight"], params[f"layers.{i}.bias"])
                               for i in range(len(spec.layer_dims))]

            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = layer(x)
                    x = x*mx.sigmoid(x) if spec.activation == "silu" else mx.tanh(x)
                return self.layers[-1](x)
        return NativeMLP()

    def parameters(self, model):
        return {f"layers.{i}.{kind}": getattr(layer, kind)
                for i, layer in enumerate(model.layers) for kind in ("weight", "bias")}

    def set_parameters(self, model, params):
        values = {name: self.to_numpy(value) for name, value in params.items()}
        validate_parameters(values, model.spec)
        for i, layer in enumerate(model.layers):
            layer.weight, layer.bias = (self.array(values[f"layers.{i}.{kind}"])
                                        for kind in ("weight", "bias"))
        return model

    def value_and_grad(self, model, loss_fn, *, create_graph=True):
        from mlx.utils import tree_flatten
        value, gradients = self.nn.value_and_grad(model, lambda: loss_fn(model))()
        if value.ndim:
            raise ValueError("Parameter differentiation requires a scalar loss")
        return value, dict(tree_flatten(gradients))

    def input_grad(self, fn, x, *, order=1):
        """Point-scalar first derivatives or diagonal second derivatives.

        For coupled points and multiple outputs use the full ``jacobian``.
        Nested MLX transformations retain dependence on model parameters.
        """
        if order not in (1, 2):
            raise ValueError("Input derivative order must be 1 or 2")
        x = self.array(x)
        first = self.mx.grad(lambda a: self.mx.sum(fn(a)))
        if order == 1:
            return first(x)
        if x.ndim == 0:
            return self.mx.grad(first)(x)
        return self.mx.stack([self.mx.grad(lambda a, i=i: self.mx.sum(first(a)[..., i]))(x)[..., i]
                              for i in range(x.shape[-1])], axis=-1)

    def jacobian(self, fn, x):
        x = self.array(x)
        output = fn(x)
        if output.size == 0:
            return self.mx.zeros((*output.shape, *x.shape), dtype=x.dtype)
        rows = [self.mx.grad(lambda a, i=i: fn(a).reshape(-1)[i])(x)
                for i in range(output.size)]
        return self.mx.stack(rows).reshape(*output.shape, *x.shape)

    def adam(self, model, *, lr=.001, betas=(.9, .999), eps=1e-8,
             weight_decay=0., amsgrad=False, **kwargs):
        import mlx.optimizers as optim
        if weight_decay != 0 or amsgrad:
            raise NotImplementedError("MLX Adam supports weight_decay=0 and amsgrad=False")
        # Torch Adam applies bias correction; use the same shared update.
        optimizer = optim.Adam(learning_rate=lr, betas=list(betas), eps=eps,
                               bias_correction=kwargs.pop("bias_correction", True), **kwargs)
        optimizer.init(model.trainable_parameters())
        return optimizer

    def lbfgs(self, model, **kwargs):
        raise NotImplementedError("LBFGS is available with the Torch backend")

    def step(self, model, optimizer, loss_fn):
        from mlx.utils import tree_unflatten
        value, gradients = self.value_and_grad(model, loss_fn)
        self.mx.eval(value, gradients)
        if not bool(self.mx.isfinite(value).item()):
            raise FloatingPointError("Nonfinite optimization loss")
        optimizer.update(model, tree_unflatten(list(gradients.items())))
        self.mx.eval(model.parameters(), optimizer.state)
        return value

    def info(self):
        from importlib.metadata import version
        return {"backend": self.name, "dtype": "float32", "device": self.device,
                "version": version("mlx"), "score_backend": "numpy", "score_dtype": "float64",
                "initialization": "numpy.default_rng(seed) uniform +/-1/sqrt(fan_in)"}
