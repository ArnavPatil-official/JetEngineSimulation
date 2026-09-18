"""
Sajben transonic-diffuser validation figure + error table (Phase 2.5, V11).

Reuses the (read-only) machinery in scripts/validation/sajben_validation.py to
produce the manuscript figure: predicted vs experimental upper/lower wall
pressure-distribution shapes, for both available checkpoints, plus a CSV
error table.

Candid framing (the data decides): both checkpoints produce shape-L2 errors
0.3-1.1, far above the < 0.10 good-match threshold defined by the validation
script — the 1-D-trained LE-PINN does NOT quantitatively validate against the
2-D planar Sajben case. The figure documents that limitation; Sajben
retraining / 2-D promotion is explicitly deferred (plan: beyond Phase 2).

Outputs:
- outputs/sajben_validation_errors.csv
- outputs/plots/sajben_wall_cp_validation.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "validation"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import sajben_validation as sv

N_AXIAL, N_NORMAL = 60, 25  # grid sizes used by sajben_validation.main()

MODELS = {
    "le_pinn_sajben.pt": PROJECT_ROOT / "models" / "le_pinn_sajben.pt",
    "le_pinn_sajben_finetuned.pt": PROJECT_ROOT / "models" / "le_pinn_sajben_finetuned.pt",
}
OUT_CSV = PROJECT_ROOT / "outputs" / "sajben_validation_errors.csv"
OUT_PLOT = PROJECT_ROOT / "outputs" / "plots" / "sajben_wall_cp_validation.png"

# dataviz reference palette slots
COLORS = {"le_pinn_sajben.pt": "#2a78d6", "le_pinn_sajben_finetuned.pt": "#1baf7a"}


def _normalise_01(v):
    v = np.asarray(v, float)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-30)


def wall_curves(preds_phys, inputs_raw, x_vec, upper_y, exp_data):
    """Replicates compute_wall_cp_errors' extraction, returning the curves."""
    # Throat HEIGHT = 2 * sqrt(A5/pi); sqrt(A5/pi) alone is the throat radius
    # (P4.2 fix, mirrors compute_wall_cp_errors in sajben_validation.py).
    H_m = 2.0 * float((inputs_raw[0, 2].item() / np.pi) ** 0.5)
    P_pred = preds_phys[:, 3].numpy()
    n_axial = len(x_vec)
    n_normal = N_NORMAL

    upper_idx = [i * n_normal + (n_normal - 1) for i in range(n_axial)]
    lower_idx = [i * n_normal for i in range(n_axial)]
    P_upper = P_pred[upper_idx]
    P_lower = P_pred[lower_idx]
    P_in = float(np.mean(P_upper[:3]))

    i_thr = int(np.argmin(upper_y))
    x_thr = x_vec[i_thr]

    out = {}
    for wall, P_wall, exp in (("upper", P_upper, exp_data["top_wall"]),
                              ("lower", P_lower, exp_data["bot_wall"])):
        x_exp_m = exp["xh"] * H_m + x_thr
        mask = (x_exp_m >= x_vec[0]) & (x_exp_m <= x_vec[-1])
        cp_model_at_exp = np.interp(x_exp_m[mask], x_vec, P_wall / (P_in + 1e-30))
        out[wall] = {
            "xh": exp["xh"][mask],
            "cp_model_norm": _normalise_01(cp_model_at_exp),
            "cp_exp_norm": _normalise_01(exp["pp0"][mask].astype(float)),
        }
    return out


def main():
    exp_data = sv.parse_sajben_experimental_data(str(sv.DATA_FILE))
    geom = sv.parse_sajben_geometry(str(sv.GEOM_FILE))
    inputs_raw, x_vec, upper_y = sv.build_sajben_grid(geom, N_AXIAL, N_NORMAL)

    rows, curves = [], {}
    for label, path in MODELS.items():
        # metrics via the validation module's own main() would re-print
        # everything; recompute compactly here with its building blocks
        model = sv.LE_PINN()
        ckpt = torch.load(str(path), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        norm_in, norm_out_fresh = sv._build_sajben_normalizers(
            geom, N_AXIAL, N_NORMAL)
        if "output_norm_min" in ckpt:
            norm_out = sv.MinMaxNormalizer()
            norm_out.data_min = ckpt["output_norm_min"][:5]
            norm_out.data_max = ckpt["output_norm_max"][:5]
        else:
            norm_out = norm_out_fresh
        preds = sv.run_forward_pass(model, inputs_raw, norm_in, norm_out)

        cp = sv.compute_wall_cp_errors(preds, inputs_raw, x_vec, upper_y,
                                       exp_data, n_normal=N_NORMAL)
        vel = sv.compute_velocity_profile_errors(preds, inputs_raw, x_vec,
                                                 upper_y, exp_data,
                                                 n_normal=N_NORMAL)
        row = {"Model": label,
               "L2_Cp_upper": cp["l2_upper"], "L2_Cp_lower": cp["l2_bot"]}
        for lbl, info in sorted(vel.items(), key=lambda kv: float(kv[0])):
            row[f"L2_u_XH_{lbl}"] = info["l2_error"]
        rows.append(row)
        curves[label] = wall_curves(preds, inputs_raw, x_vec, upper_y, exp_data)
        print(f"{label}: Cp L2 upper={cp['l2_upper']:.3f} lower={cp['l2_bot']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # Figure: normalized wall-pressure shape, model vs experiment
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, wall in zip(axes, ("upper", "lower")):
        first = True
        for label in MODELS:
            c = curves[label][wall]
            if first:
                ax.scatter(c["xh"], c["cp_exp_norm"], s=26, color="#1a1a19",
                           zorder=3, label="Sajben experiment")
                first = False
            ax.plot(c["xh"], c["cp_model_norm"], lw=1.8, color=COLORS[label],
                    label=label.replace(".pt", ""))
        ax.set_xlabel("x / H*")
        ax.set_title(f"{wall.title()} wall")
        ax.grid(True, lw=0.4, alpha=0.4)
    axes[0].set_ylabel("Normalized wall pressure shape [0-1]")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("LE-PINN vs Sajben transonic diffuser — wall pressure shape\n"
                 "(1-D-trained checkpoints do not reproduce the 2-D planar case; "
                 "shape-L2 errors 0.71-1.09 vs <0.10 threshold)", y=1.06)
    fig.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_CSV}\n       {OUT_PLOT}")


if __name__ == "__main__":
    main()
