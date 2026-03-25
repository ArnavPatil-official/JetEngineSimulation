#!/usr/bin/env python3
"""
ISEF Poster – Turbofan Digital Twin PINN Architecture Diagram
Three PINN modules + shared loss function block, dark Cerulean Blue theme.
Output: outputs/pinn_architecture_diagram.png  (30×10 in, 200 dpi)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Palette (Cerulean Blue theme) ─────────────────────────────────────────────
BG       = '#04101D'   # deep navy background
CARD_BG  = '#0A1929'   # module card fill
EQ_BG    = '#020C14'   # physics equation box fill
LIGHT    = '#D6ECFF'   # near-white text
SUBLITE  = '#8BAEC8'   # dimmed text / separators
ACCENT   = '#1E90FF'   # cerulean – arrows, borders

M1_C  = '#00C8D7'   # teal   – 1D Nozzle PINN
M2_C  = '#FF6B35'   # orange – Turbine PINN
M3_C  = '#B06AE8'   # violet – 2D LE-PINN
LOSS_C = '#1565C0'   # deep cerulean – loss block

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'mathtext.fontset':  'stixsans',
    'mathtext.default':  'regular',
})

# ── Canvas ─────────────────────────────────────────────────────────────────────
FW, FH = 30, 10
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Layout constants ───────────────────────────────────────────────────────────
MW      = 8.50   # module width
MH      = 6.90   # module height
MY0     = 2.30   # module bottom y  (top → 9.20, title strip 9.20–10.00)
HDR_H   = 0.88   # coloured header strip height
EQ_H    = 2.05   # physics-constraint box height
EQ_PAD  = 0.13   # gap: module bottom → physics box

MXS = [0.50, 10.75, 21.00]   # left edges of modules 1-2-3

LY0 = 0.10   # loss block bottom
LH  = 1.85   # loss block height  (top → 1.95)
LX0 = 0.50   # loss block left
LW  = 29.00  # loss block width   (right → 29.50)

# ── Primitive helpers ──────────────────────────────────────────────────────────

def rbox(x, y, w, h, ec, fc=CARD_BG, lw=2.0, rad=0.07, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad={rad}',
        linewidth=lw, edgecolor=ec,
        facecolor=fc, zorder=z,
    )
    ax.add_patch(p)
    return p


def arrow(x0, y0, x1, y1, *, color=ACCENT, lw=2.0, ms=16,
          style='->', ls='solid', z=5):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color,
        linewidth=lw, mutation_scale=ms,
        linestyle=ls, zorder=z,
    ))


def T(x, y, s, *, size=9, color=LIGHT, ha='center', va='top',
       bold=False, italic=False, z=4):
    ax.text(x, y, s,
            ha=ha, va=va, fontsize=size, color=color,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            zorder=z)


# ── Module renderer ────────────────────────────────────────────────────────────

def draw_module(x0, color, badge, title, subtitle, inputs, outputs, eqs):
    y0 = MY0

    # ── outer card ──
    rbox(x0, y0, MW, MH, ec=color, fc=CARD_BG, lw=2.5, rad=0.09)

    # ── header strip ──
    hy = y0 + MH - HDR_H
    rbox(x0, hy, MW, HDR_H, ec='none', fc=color, lw=0, rad=0.07, z=3)

    # number badge (white pill)
    bw, bh = 0.56, 0.54
    rbox(x0 + 0.18, hy + (HDR_H - bh) / 2, bw, bh,
         ec='white', fc='white', lw=0, rad=0.05, z=5)
    T(x0 + 0.18 + bw / 2, hy + HDR_H / 2,
      badge, size=9, color=color, va='center', bold=True, z=6)

    T(x0 + MW / 2 + 0.15, hy + HDR_H * 0.60,
      title, size=11.5, color='white', va='center', bold=True, z=4)
    if subtitle:
        T(x0 + MW / 2 + 0.15, hy + 0.12,
          subtitle, size=7.5, color='white', va='bottom', italic=True, z=4)

    # ── I/O section ──
    io_top = y0 + MH - HDR_H - 0.22
    io_bot = y0 + EQ_H + EQ_PAD + 0.40
    mid_x  = x0 + MW / 2

    ax.plot([mid_x, mid_x], [io_bot + 0.08, io_top - 0.04],
            color=SUBLITE, lw=0.7, alpha=0.5, zorder=3)

    T(x0 + 0.22, io_top, 'INPUTS',  size=7, color=color, ha='left', bold=True)
    for i, s in enumerate(inputs):
        T(x0 + 0.30, io_top - 0.38 - i * 0.46,
          f'\u25b6  {s}', size=7.8, color=LIGHT, ha='left')

    T(mid_x + 0.22, io_top, 'OUTPUTS', size=7, color=color, ha='left', bold=True)
    for i, s in enumerate(outputs):
        T(mid_x + 0.30, io_top - 0.38 - i * 0.46,
          f'\u25b6  {s}', size=7.8, color=LIGHT, ha='left')

    # ── physics constraints box ──
    ey = y0 + EQ_PAD
    rbox(x0 + 0.14, ey, MW - 0.28, EQ_H,
         ec=color, fc=EQ_BG, lw=1.5, rad=0.05, z=3)

    T(x0 + MW / 2, ey + EQ_H - 0.09,
      'PHYSICS  CONSTRAINTS', size=6.5, color=color, bold=True)

    n = max(len(eqs), 1)
    step = (EQ_H - 0.42) / n
    for i, eq in enumerate(eqs):
        T(x0 + MW / 2, ey + EQ_H - 0.44 - i * step,
          eq, size=7.0, color=LIGHT)


# ── MODULE 1: 1D Nozzle PINN ──────────────────────────────────────────────────
draw_module(
    MXS[0], M1_C, '01',
    title    = '1D Nozzle PINN',
    subtitle = 'Converging\u2013Diverging Nozzle',
    inputs   = [
        r'$x$  (axial position)',
        r'$\rho_0,\; p_0,\; T_0$  (inlet)',
        r'$A(x)$  nozzle geometry',
    ],
    outputs  = [
        r'$\rho(x)$  density field',
        r'$u(x)$  velocity field',
        r'$p(x)$  pressure field',
        r'$T(x)$  temperature field',
    ],
    eqs = [
        r'$\partial(\rho u A)/\partial x = 0$',
        r'$\rho u\,(\partial u/\partial x) + \partial p/\partial x = 0$',
        r'$c_p T + u^2/2 = \mathrm{const}$',
        r'$p = \rho R T$',
        r'$A(x)=A_{in}+(A_{out}{-}A_{in})(1-\cos(\pi x/2))$',
    ],
)

# ── MODULE 2: Turbine PINN ────────────────────────────────────────────────────
draw_module(
    MXS[1], M2_C, '02',
    title    = 'Turbine PINN',
    subtitle = 'Shaft Work Extraction Model',
    inputs   = [
        r'$\rho,\; p,\; T$  (combustor exit)',
        r'$\dot{m}$  (mass flow rate)',
    ],
    outputs  = [
        r'$W$  (work extraction)',
        r'$\rho_{ex},\; p_{ex},\; T_{ex}$',
        r'Target:  $W \approx 57.4\;\mathrm{MW}$',
    ],
    eqs = [
        r'$\partial(\rho u)/\partial x = 0$   (continuity)',
        r'$\rho u\,(\partial u/\partial x) + \partial p/\partial x = 0$',
        r'$W = \dot{m}\,(h_{in} - h_{out})$',
        r'$\mathcal{L}_{E} = (W - W_{t})^2 / W_{t}^2$',
    ],
)

# ── MODULE 3: 2D Nozzle LE-PINN ───────────────────────────────────────────────
draw_module(
    MXS[2], M3_C, '03',
    title    = '2D Nozzle  LE-PINN',
    subtitle = '2D Extension  \u00b7  Radial Flow Gradients',
    inputs   = [
        r'$(x,\,r)$  2D spatial coordinates',
        r'$A_5,\,A_6$  (throat / exit area)',
        r'$p_{in},\; T_{in}$  (inlet BC)',
    ],
    outputs  = [
        r'$p(x,r),\; T(x,r)$  2D fields',
        r'$\rho,\,u,\,v,\,UU,\,VV,\,UV,\,\mu_{eff}$',
        'Global + boundary network fusion',
    ],
    eqs = [
        r'$\partial(\rho u)/\partial x + \partial(\rho v)/\partial r = 0$',
        r'RANS $x$-mom, $r$-mom, energy residuals',
        r'EOS:  $p = \rho R T$',
        r'Wall BC: $u_w\!=\!v_w\!=\!0,\;\partial T/\partial n\!=\!0$',
        r'LE-fusion: replace $p,T$ where $d < \delta$',
    ],
)

# ── Horizontal data-flow arrows between modules ───────────────────────────────
arr_y = MY0 + MH * 0.68   # ≈ y=7.0

for (xs, xe, lbl) in [
    (MXS[0] + MW, MXS[1], 'nozzle exit \u2192 turbine inlet'),
    (MXS[1] + MW, MXS[2], 'flow conditions'),
]:
    mid = (xs + xe) / 2
    arrow(xs + 0.10, arr_y, xe - 0.10, arr_y,
          color=ACCENT, lw=2.6, ms=20)
    T(mid, arr_y + 0.30, lbl, size=7.5, color=ACCENT)

# ── Connector arrows: modules → loss block ────────────────────────────────────
loss_top = LY0 + LH   # = 1.95

for x0 in MXS:
    cx = x0 + MW / 2
    arrow(cx, MY0 - 0.06, cx, loss_top + 0.06,
          color=ACCENT, lw=1.6, ms=11, ls='dashed')

# ── Loss Function Block ───────────────────────────────────────────────────────
# Soft glow border (slightly oversized semi-transparent fill)
glow = FancyBboxPatch(
    (LX0 - 0.12, LY0 - 0.12), LW + 0.24, LH + 0.24,
    boxstyle='round,pad=0.12',
    linewidth=0, edgecolor='none',
    facecolor=ACCENT, alpha=0.08, zorder=2,
)
ax.add_patch(glow)

rbox(LX0, LY0, LW, LH, ec=ACCENT, fc='#060F1F', lw=2.8, rad=0.10, z=3)

# ── Loss block header label (left-aligned) ──
T(LX0 + 0.48, LY0 + LH - 0.13,
  'SHARED  LOSS  FUNCTION',
  size=11, color=ACCENT, ha='left', bold=True)

# ── Main formula (centred) ──
T(LX0 + LW / 2, LY0 + LH - 0.18,
  (r'$\mathcal{L}_{total}'
   r'\;=\;\lambda_{BC}\!\cdot\!\mathcal{L}_{BC}'
   r'\;+\;\lambda_{phys}\!\cdot\!\mathcal{L}_{physics}'
   r'\;+\;\lambda_{E}\!\cdot\!\mathcal{L}_{energy}$'),
  size=13.5, color=LIGHT)

# ── Three term descriptions ──
TERMS = [
    (r'$\mathcal{L}_{BC}\;\;(\lambda = 10.0)$',
     'Boundary condition matching',
     r'MSE on $\rho,\,p,\,T$ at inlet',
     M1_C),
    (r'$\mathcal{L}_{physics}\;\;(\lambda = 1.0)$',
     'Physics residuals',
     r'Continuity $+$ momentum$^2$ residuals',
     M2_C),
    (r'$\mathcal{L}_{energy}\;\;(\lambda = 5.0)$',
     'Turbine work deviation',
     r'normalised from design $W_{target}$',
     M3_C),
]

txs = [LX0 + LW * f for f in [0.17, 0.50, 0.83]]

for (lbl, d1, d2, col), tx in zip(TERMS, txs):
    T(tx, LY0 + 0.96, lbl,  size=9.5, color=col, bold=True)
    T(tx, LY0 + 0.60, d1,   size=8.0, color=LIGHT)
    T(tx, LY0 + 0.35, d2,   size=8.0, color=SUBLITE)

# Vertical separators in loss block
for sx in [LX0 + LW / 3, LX0 + 2 * LW / 3]:
    ax.plot([sx, sx], [LY0 + 0.20, LY0 + 1.12],
            color=SUBLITE, lw=0.8, alpha=0.45, zorder=3)

# ── Main title strip ──────────────────────────────────────────────────────────
# Thin decorative rule below title area
ax.plot([0.5, FW - 0.5], [MY0 + MH + 0.04, MY0 + MH + 0.04],
        color=ACCENT, lw=0.6, alpha=0.35, zorder=2)

T(FW / 2, MY0 + MH + 0.72,
  'Turbofan Digital Twin \u2014 PINN Module Architecture',
  size=16, color=LIGHT, bold=True)
T(FW / 2, MY0 + MH + 0.36,
  'Three Physics-Informed Neural Network modules with a shared multi-component loss function',
  size=10, color=SUBLITE, italic=True)

# Module colour legend (top-right corner)
legend_x = FW - 7.6
legend_y = MY0 + MH + 0.72
for i, (lbl, col) in enumerate([
    ('1D Nozzle PINN',   M1_C),
    ('Turbine PINN',     M2_C),
    ('2D Nozzle LE-PINN', M3_C),
]):
    bx = legend_x + i * 2.55
    rbox(bx, legend_y - 0.26, 0.30, 0.30,
         ec=col, fc=col, lw=0, rad=0.04, z=5)
    T(bx + 0.45, legend_y - 0.11, lbl,
      size=7.5, color=col, ha='left', va='center')

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = 'outputs/pinn_architecture_diagram.png'
fig.savefig(
    out_path, dpi=200,
    bbox_inches='tight',
    facecolor=BG, edgecolor='none',
)
print(f'Saved \u2192 {out_path}')
plt.close(fig)
