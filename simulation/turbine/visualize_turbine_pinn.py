
"""
Visualization utilities for turbine_pinn.pt.

Creates:
- turbine_states_normalized.png  (rho,u,p,T normalized by inlet values vs x)
- turbine_residuals.png          (EOS residual + mass residual vs x, log scale)
"""

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

import turbine  # expects turbine.py in same folder or on PYTHONPATH

HERE = Path(__file__).resolve().parent
CKPT = HERE / "turbine_pinn.pt"

def load_turbine_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    model = turbine.NormalizedTurbinePINN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # restore globals for predict_physical
    turbine.SCALES = ckpt.get("scales", turbine.SCALES)
    turbine.CONDITIONS = ckpt.get("conditions", turbine.CONDITIONS)
    return model, turbine.CONDITIONS, turbine.SCALES

def compute_residuals(model, x, conditions, scales):
    x = x.clone().requires_grad_(True)
    out_norm = model(x)

    rho = out_norm[:, 0:1] * scales["rho"]
    u   = out_norm[:, 1:2] * scales["u"]
    p   = out_norm[:, 2:3] * scales["p"]
    T   = out_norm[:, 3:4] * scales["T"]

    A = conditions["geometry"]["A_inlet"] + (conditions["geometry"]["A_outlet"] - conditions["geometry"]["A_inlet"]) * x

    R = conditions["physics"]["R"]
    eos_res = (p - rho * R * T) / scales["p"]

    mass_flow = rho * u * A
    mass_flow_x = torch.autograd.grad(mass_flow, x, torch.ones_like(mass_flow), create_graph=True)[0]
    m_scale = scales["rho"] * scales["u"] * conditions["geometry"]["A_inlet"]
    mass_res = mass_flow_x / m_scale

    return eos_res.detach().numpy().ravel(), mass_res.detach().numpy().ravel()

def main():
    if not CKPT.exists():
        print(f"❌ Missing {CKPT}. Put turbine_pinn.pt in {HERE} and re-run.")
        return

    model, conditions, scales = load_turbine_checkpoint(CKPT)

    x = torch.linspace(0, 1, 250).reshape(-1, 1)
    with torch.no_grad():
        state = model.predict_physical(x)  # [rho,u,p,T]
    rho = state[:, 0].numpy()
    u   = state[:, 1].numpy()
    p   = state[:, 2].numpy()
    T   = state[:, 3].numpy()

    # normalized by inlet values
    rho0, u0, p0, T0 = rho[0], u[0], p[0], T[0]
    plt.figure()
    plt.plot(x.numpy().ravel(), rho/rho0, label="ρ/ρ_in")
    plt.plot(x.numpy().ravel(), u/u0, label="u/u_in")
    plt.plot(x.numpy().ravel(), p/p0, label="p/p_in")
    plt.plot(x.numpy().ravel(), T/T0, label="T/T_in")
    plt.xlabel("Normalized axial position x*")
    plt.ylabel("Normalized state")
    plt.title("Turbine PINN: normalized state profiles")
    plt.legend()
    plt.tight_layout()
    out1 = HERE / "turbine_states_normalized.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # residuals
    eos_res, mass_res = compute_residuals(model, x, conditions, scales)
    plt.figure()
    plt.semilogy(x.numpy().ravel(), np.abs(eos_res) + 1e-16, label="|EOS residual|")
    plt.semilogy(x.numpy().ravel(), np.abs(mass_res) + 1e-16, label="|Mass residual|")
    plt.xlabel("Normalized axial position x*")
    plt.ylabel("Residual magnitude (log)")
    plt.title("Turbine PINN: physics residuals")
    plt.legend()
    plt.tight_layout()
    out2 = HERE / "turbine_residuals.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    print("✅ Saved:", out1.name, "and", out2.name)

if __name__ == "__main__":
    main()
