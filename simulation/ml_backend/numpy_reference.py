"""Independent CPU64 MLP forward, analytic gradients and input Hessians."""
from __future__ import annotations


def activation_terms(z, activation):
    import numpy as np
    if activation == "tanh":
        value = np.tanh(z); first = 1-value**2
        return value, first, -2*value*first
    if activation != "silu":
        raise ValueError("Unsupported activation")
    positive = z >= 0
    s = np.empty_like(z)
    s[positive] = 1/(1+np.exp(-z[positive]))
    exp_z = np.exp(z[~positive]); s[~positive] = exp_z/(1+exp_z)
    ds = s*(1-s)
    return z*s, s+z*ds, 2*ds+z*ds*(1-2*s)


def _forward_cache(params, x, activation):
    import numpy as np
    from .interface import validate_parameters
    dimensions = validate_parameters(params)
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != dimensions[0][0]:
        raise ValueError("Input dimension differs from model")
    values, pre = [x.reshape(-1, x.shape[-1])], []
    for i in range(len(dimensions)):
        z = values[-1] @ np.asarray(params[f"layers.{i}.weight"], dtype=np.float64).T
        z += np.asarray(params[f"layers.{i}.bias"], dtype=np.float64)
        pre.append(z)
        values.append(activation_terms(z, activation)[0] if i+1 < len(dimensions) else z)
    return values, pre, x.shape[:-1]


def forward(params, x, *, activation="silu"):
    values, _, shape = _forward_cache(params, x, activation)
    return values[-1].reshape(*shape, values[-1].shape[-1])


def mse_parameter_gradients(params, x, target, *, activation="silu"):
    import numpy as np
    values, pre, _ = _forward_cache(params, x, activation)
    prediction = values[-1]
    target = np.asarray(target, dtype=np.float64).reshape(prediction.shape)
    error = prediction-target
    delta = 2*error/error.size
    grads = {}
    for i in range(len(pre)-1, -1, -1):
        grads[f"layers.{i}.weight"] = delta.T @ values[i]
        grads[f"layers.{i}.bias"] = delta.sum(axis=0)
        if i:
            delta = (delta @ np.asarray(params[f"layers.{i}.weight"], dtype=np.float64))*activation_terms(pre[i-1], activation)[1]
    return float(np.mean(error**2)), grads


def input_derivatives(params, x, *, activation="silu"):
    """Per-row Jacobian (..., output, input), Hessian (..., output, input, input)."""
    import numpy as np
    values, pre, shape = _forward_cache(params, x, activation)
    n, width = values[0].shape
    jac = np.broadcast_to(np.eye(width), (n, width, width)).copy()
    hess = np.zeros((n, width, width, width))
    for i, z in enumerate(pre):
        weight = np.asarray(params[f"layers.{i}.weight"], dtype=np.float64)
        jac = np.einsum("oi,bik->bok", weight, jac)
        hess = np.einsum("oi,bikl->bokl", weight, hess)
        if i+1 < len(pre):
            _, first, second = activation_terms(z, activation)
            hess = first[..., None, None]*hess + second[..., None, None]*jac[..., :, None]*jac[..., None, :]
            jac = first[..., None]*jac
    return jac.reshape(*shape, *jac.shape[-2:]), hess.reshape(*shape, *hess.shape[-3:])
