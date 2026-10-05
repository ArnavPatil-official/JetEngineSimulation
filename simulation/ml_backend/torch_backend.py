"""Native PyTorch MLP, float64 CPU by default, with graph-preserving calculus."""
from __future__ import annotations

import os

from .interface import BackendBase, MLPSpec, seeded_parameters, validate_parameters


def torch_dtype(value=None):
    import torch
    import numpy as np
    if value is None:
        return torch.float64
    if isinstance(value, torch.dtype):
        return value
    name = np.dtype(value).name
    mapping = {"float64": torch.float64, "float32": torch.float32,
               "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}
    if name not in mapping:
        raise ValueError(f"Unsupported array dtype {name}")
    return mapping[name]


class TorchOps:
    def __init__(self, device="cpu", dtype=None):
        import torch
        self.torch, self.device, self.dtype = torch, torch.device(device), torch_dtype(dtype)
        self.training_dtype = self.dtype
        self.float32, self.float64 = torch.float32, torch.float64

    def array(self, value, dtype=None):
        return self.torch.as_tensor(value, dtype=self.dtype if dtype is None else torch_dtype(dtype), device=self.device)

    asarray = array

    def sum(self, value, axis=None, keepdims=False):
        return value.sum() if axis is None else value.sum(dim=axis, keepdim=keepdims)

    def mean(self, value, axis=None, keepdims=False):
        return value.mean() if axis is None else value.mean(dim=axis, keepdim=keepdims)

    def max(self, value, axis=None, keepdims=False):
        return self.torch.amax(value) if axis is None else self.torch.amax(value, dim=axis, keepdim=keepdims)

    def min(self, value, axis=None, keepdims=False):
        return self.torch.amin(value) if axis is None else self.torch.amin(value, dim=axis, keepdim=keepdims)

    def exp(self, value): return self.torch.exp(value)
    def log(self, value): return self.torch.log(value)
    def tanh(self, value): return self.torch.tanh(value)
    def sqrt(self, value): return self.torch.sqrt(value)
    def abs(self, value): return self.torch.abs(value)
    def sigmoid(self, value): return self.torch.sigmoid(value)
    def softplus(self, value): return self.torch.logaddexp(value, self.torch.zeros_like(value))
    def zeros_like(self, value): return self.torch.zeros_like(value)
    def ones_like(self, value): return self.torch.ones_like(value)
    def isfinite(self, value): return self.torch.isfinite(value)
    def where(self, condition, a, b): return self.torch.where(condition, a, b)
    def maximum(self, a, b): return self.torch.maximum(a, self.array(b, dtype=a.dtype))
    def clip(self, value, a_min, a_max): return self.torch.clamp(value, min=a_min, max=a_max)
    def stack(self, values, axis=0, dim=None): return self.torch.stack(tuple(values), dim=axis if dim is None else dim)
    def concatenate(self, values, axis=0): return self.torch.cat(tuple(values), dim=axis)
    def zeros(self, shape, dtype=None): return self.torch.zeros(shape, dtype=self.dtype if dtype is None else torch_dtype(dtype), device=self.device)
    def stop_gradient(self, value): return value.detach()


class TorchBackend(BackendBase):
    name = "torch"

    def __init__(self, *, device="cpu", dtype=None):
        import torch
        if str(device) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Torch scientific backend supports cpu or cuda")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but no usable CUDA device is available")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
        self.torch, self.device, self.dtype = torch, device, torch_dtype(dtype)
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Torch model dtype must be float32 or float64")
        self.ops = TorchOps(device, self.dtype)

    def array(self, value, dtype=None, *, requires_grad=False):
        value = self.ops.array(value, dtype=dtype)
        return value.requires_grad_(True) if requires_grad else value

    def seed(self, seed):
        """Explicitly seed NumPy and native Torch random operations."""
        import numpy as np
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        return seed

    def to_numpy(self, value):
        import numpy as np
        return value.detach().cpu().numpy().copy() if isinstance(value, self.torch.Tensor) else np.asarray(value).copy()

    def mlp(self, in_dim, hidden, out_dim, *, activation="silu", seed=0):
        torch, dtype, device = self.torch, self.dtype, self.device
        spec = MLPSpec(int(in_dim), tuple(hidden), int(out_dim), activation)
        params = seeded_parameters(spec, seed, str(dtype).split(".")[-1])
        class NativeMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.spec = spec
                # Meta construction avoids perturbing global Torch RNG state.
                self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o, dtype=dtype, device="meta")
                                                    for i, o in spec.layer_dims])
                for i, layer in enumerate(self.layers):
                    layer.weight = torch.nn.Parameter(torch.as_tensor(params[f"layers.{i}.weight"], dtype=dtype, device=device).clone())
                    layer.bias = torch.nn.Parameter(torch.as_tensor(params[f"layers.{i}.bias"], dtype=dtype, device=device).clone())

            def forward(self, x):
                for layer in self.layers[:-1]:
                    x = layer(x)
                    x = torch.nn.functional.silu(x) if spec.activation == "silu" else torch.tanh(x)
                return self.layers[-1](x)
        return NativeMLP()

    def parameters(self, model):
        return dict(model.named_parameters())

    def set_parameters(self, model, params):
        values = {name: self.to_numpy(value) for name, value in params.items()}
        validate_parameters(values, model.spec)
        with self.torch.no_grad():
            for name, parameter in self.parameters(model).items():
                parameter.copy_(self.array(values[name]))
        return model

    def value_and_grad(self, model, loss_fn, *, create_graph=True):
        params = self.parameters(model)
        value = loss_fn(model)
        if value.ndim:
            raise ValueError("Parameter differentiation requires a scalar loss")
        if not value.requires_grad:
            return value, {name: parameter*0 for name, parameter in params.items()}
        grads = self.torch.autograd.grad(value, tuple(params.values()), create_graph=create_graph,
                                        allow_unused=True)
        return value, {name: parameter*0 if grad is None else grad
                       for (name, parameter), grad in zip(params.items(), grads)}

    def input_grad(self, fn, x, *, order=1):
        """Point-scalar derivatives; order=2 returns each input's diagonal second derivative.

        ``fn`` must return independent point values, or their scalar sum. For
        coupled points/arbitrary multiple outputs use ``jacobian`` instead.
        Both derivatives retain the graph to model parameters.
        """
        if order not in (1, 2):
            raise ValueError("Input derivative order must be 1 or 2")
        x = self.array(x, requires_grad=True)
        def gradient(function, argument):
            y = function(argument)
            if not y.requires_grad:
                return argument*0
            value = self.torch.autograd.grad(y.sum(), argument, create_graph=True,
                                            retain_graph=True, allow_unused=True)[0]
            return argument*0 if value is None else value
        first = gradient(fn, x)
        if order == 1:
            return first
        if x.ndim == 0:
            return gradient(lambda _: first, x)
        return self.torch.stack([gradient(lambda _: first[..., i], x)[..., i]
                                 for i in range(x.shape[-1])], dim=-1)

    def jacobian(self, fn, x):
        return self.torch.autograd.functional.jacobian(fn, self.array(x), create_graph=True)

    def adam(self, model, *, lr=.001, betas=(.9, .999), eps=1e-8, **kwargs):
        return self.torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, **kwargs)

    def lbfgs(self, model, **kwargs):
        return self.torch.optim.LBFGS(model.parameters(), **kwargs)

    def step(self, model, optimizer, loss_fn):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            value, gradients = self.value_and_grad(model, loss_fn, create_graph=False)
            if not bool(self.torch.isfinite(value).detach()):
                raise FloatingPointError("Nonfinite optimization loss")
            for name, parameter in self.parameters(model).items():
                parameter.grad = gradients[name].detach()
            return value
        if isinstance(optimizer, self.torch.optim.LBFGS):
            return optimizer.step(closure).detach()
        value = closure(); optimizer.step()
        return value.detach()

    def score64(self, model_or_params, x, *, activation=None):
        """Actual Torch float64 CPU inference, returned as a NumPy64 array.

        A native CPU64 model is reused directly. Other models or canonical
        parameters are promoted from their stored precision into a CPU model.
        """
        if isinstance(model_or_params, dict):
            params = {name: self.to_numpy(value) for name, value in model_or_params.items()}
            selected_activation = activation or "silu"
            model = None
        else:
            model = model_or_params
            selected_activation = activation or model.spec.activation
            params = None
            if activation is not None and activation != model.spec.activation:
                raise ValueError("Scoring activation differs from model")
            if any(p.device.type != "cpu" or p.dtype != self.torch.float64
                   for p in model.parameters()):
                params = {name: self.to_numpy(value) for name, value in self.parameters(model).items()}
                model = None
        cpu = TorchBackend(device="cpu", dtype="float64")
        if model is None:
            dimensions = validate_parameters(params)
            model = cpu.mlp(dimensions[0][0], tuple(pair[1] for pair in dimensions[:-1]),
                            dimensions[-1][1], activation=selected_activation, seed=0)
            cpu.set_parameters(model, params)
        with self.torch.no_grad():
            return cpu.to_numpy(model(cpu.array(x)))

    def info(self):
        return {"backend": self.name, "dtype": str(self.dtype).split(".")[-1], "device": str(self.device),
                "version": self.torch.__version__, "score_backend": "torch", "score_dtype": "float64",
                "initialization": "numpy.default_rng(seed) uniform +/-1/sqrt(fan_in)"}
