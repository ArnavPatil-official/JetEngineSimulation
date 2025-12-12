"""
Nozzle Physics-Informed Neural Network (PINN) with Fuel-Dependent Thermodynamics - V2.

This module trains a PINN to model converging nozzle expansion physics while enforcing
conservation laws and supporting fuel-dependent thermodynamic properties at runtime.

KEY IMPROVEMENTS (V2):
- Reproducible training with fixed random seeds
- Robust file paths relative to repo root
- Enhanced checkpoint saving (includes thermo baseline, version tag)
- Choking detection and Mach-aware physics constraints
- Improved boundary condition handling for inlet/outlet
- Runtime thermo conditioning support for fuel blends

Model Architecture:
- Input: Normalized axial position x* ∈ [0,1]
- Output: Flow state [ρ*, u*, p*, T*] (normalized)
- Physics: Continuity, momentum, isentropic energy with fuel-specific cp/R/gamma

Training Baseline:
The PINN is trained on a reference thermodynamic state (default: air-like properties).
At inference time, fuel-dependent properties are injected through anchoring/rescaling.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Import conditions builder
from nozzle_conditions import load_engine_conditions_from_icao, build_nozzle_conditions_from_turbine_exit

# ============================================================================
# CONFIGURATION & REPRODUCIBILITY
# ============================================================================

# Random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Repo root for robust paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / 'data'
CHECKPOINT_DIR = REPO_ROOT

# Version tag for checkpoint tracking
VERSION_TAG = "v2.0_fuel_dependent_choking"

# ============================================================================
# LOAD DEFAULT CONDITIONS
# ============================================================================

# Load baseline training conditions (air-like properties)
CONDITIONS = load_engine_conditions_from_icao(
    filename='icao_engine_data.csv',
    mode='TAKE-OFF',
    thermo_props=None  # Use defaults for training baseline
)

# Normalization Scales
SCALES = {
    'rho': CONDITIONS['inlet']['rho'],
    'u': 650.0,
    'p': CONDITIONS['inlet']['p'],
    'T': CONDITIONS['inlet']['T'],
    'L': CONDITIONS['geometry']['length']
}

# Exergy efficiency constant
BETA_EXERGY = 1.06  # Chemical exergy factor for hydrocarbon fuels

# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================

class NozzlePINN(nn.Module):
    """
    Physics-Informed Neural Network for nozzle flow prediction.

    Architecture: 3 hidden layers × 64 neurons with Tanh activation
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)  # Output: [rho_norm, u_norm, p_norm, T_norm]
        )
        # Xavier initialization for better training
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        """Forward pass: x ∈ [0,1] → [ρ*, u*, p*, T*]"""
        return self.net(x)

    def predict_physical(self, x, scales=None):
        """
        Predict in physical units.

        Args:
            x: Normalized position [0, 1]
            scales: Optional custom scales (uses global SCALES if None)

        Returns:
            Physical state [rho, u, p, T] tensor
        """
        if scales is None:
            scales = SCALES

        out_norm = self.forward(x)
        return torch.cat([
            out_norm[:, 0:1] * scales['rho'],
            out_norm[:, 1:2] * scales['u'],
            out_norm[:, 2:3] * scales['p'],
            out_norm[:, 3:4] * scales['T']
        ], dim=1)

# ============================================================================
# GEOMETRY & PHYSICS
# ============================================================================

def get_area(x, conditions):
    """
    Converging nozzle area profile: A(x) smoothly decreases from inlet to exit.

    Uses cosine profile for smooth variation:
    A(x) = A_in + (A_exit - A_in) * (1 - cos(πx/2))

    Args:
        x: Normalized position [0, 1]
        conditions: CONDITIONS dict with geometry

    Returns:
        Cross-sectional area [m²]
    """
    A_in = conditions['geometry']['A_inlet']
    A_out = conditions['geometry']['A_exit']
    return A_in + (A_out - A_in) * (1 - torch.cos(x * np.pi / 2))


def compute_critical_pressure_ratio(gamma):
    """
    Calculate critical pressure ratio for choking.

    For isentropic flow, choking occurs when:
    p*/p0 = (2/(gamma+1))^(gamma/(gamma-1))

    Args:
        gamma: Heat capacity ratio

    Returns:
        Critical pressure ratio (p_throat / p_0) for M=1
    """
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def detect_choking(p_inlet, p_ambient, gamma):
    """
    Detect if nozzle is choked.

    Args:
        p_inlet: Nozzle inlet pressure [Pa]
        p_ambient: Ambient pressure [Pa]
        gamma: Heat capacity ratio

    Returns:
        Tuple of (is_choked: bool, p_critical: float)
    """
    pr_critical = compute_critical_pressure_ratio(gamma)
    p_critical = p_inlet * pr_critical

    is_choked = (p_ambient <= p_critical)

    return is_choked, p_critical


def compute_loss(model, x_col, device, conditions=None, scales=None):
    """
    Compute physics-based loss components.

    This loss function enforces:
    1. Equation of State (EOS): p = ρ R T
    2. Mass Conservation: d(ρ u A)/dx = 0
    3. Energy Conservation: cp T + u²/2 = constant (stagnation enthalpy)
    4. Mach number consistency (soft constraint)

    Args:
        model: NozzlePINN model
        x_col: Collocation points [0, 1]
        device: Torch device
        conditions: CONDITIONS dict (fuel-dependent cp, R, gamma)
        scales: SCALES dict

    Returns:
        Tuple of (loss_eos, loss_mass, loss_energy, loss_mach)
    """
    if conditions is None:
        conditions = CONDITIONS
    if scales is None:
        scales = SCALES

    x = x_col.clone().requires_grad_(True)
    out_norm = model(x)
    rho = out_norm[:, 0:1] * scales['rho']
    u   = out_norm[:, 1:2] * scales['u']
    p   = out_norm[:, 2:3] * scales['p']
    T   = out_norm[:, 3:4] * scales['T']

    A = get_area(x, conditions)

    # Extract fuel-dependent properties
    R = conditions['physics']['R']
    cp = conditions['physics']['cp']
    gamma = conditions['physics']['gamma']

    # 1. EOS Loss: p = ρ R T
    eos_res = (p - rho * R * T) / scales['p']

    # 2. Mass Flow Loss: d(ρ u A)/dx = 0
    m_flow = rho * u * A
    m_flow_x = torch.autograd.grad(
        m_flow, x,
        torch.ones_like(m_flow),
        create_graph=True
    )[0]
    mass_res = m_flow_x / conditions['physics']['mass_flow']

    # 3. Energy Loss: H0 = cp T + u²/2 = constant
    H0_target = cp * conditions['inlet']['T'] + 0.5 * conditions['inlet']['u']**2
    H0_current = cp * T + 0.5 * u**2
    energy_res = (H0_current - H0_target) / H0_target

    # 4. Mach number penalty (soft): discourage supersonic flow in converging section
    # Mach = u / sqrt(gamma * R * T)
    a_sound = torch.sqrt(gamma * R * T)  # Speed of sound
    mach = u / a_sound
    # Penalize if Mach > 1.05 (allow slight overshoot for numerical stability)
    mach_penalty = torch.relu(mach - 1.05)

    return (
        (eos_res**2).mean(),
        (mass_res**2).mean(),
        (energy_res**2).mean(),
        (mach_penalty**2).mean()
    )


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_model(
    model,
    filename="nozzle_pinn.pt",
    scales=None,
    conditions=None,
    training_info=None
):
    """
    Save model checkpoint with complete metadata.

    Args:
        model: Trained NozzlePINN
        filename: Save path
        scales: SCALES dict
        conditions: CONDITIONS dict (includes training thermo baseline)
        training_info: Optional dict with epoch, loss, etc.
    """
    if scales is None:
        scales = SCALES
    if conditions is None:
        conditions = CONDITIONS

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scales': scales,
        'conditions': conditions,
        'version': VERSION_TAG,
        'training_info': training_info or {},
        'random_seed': RANDOM_SEED
    }

    save_path = CHECKPOINT_DIR / filename
    torch.save(checkpoint, save_path)
    print(f"\n💾 Model checkpoint saved: {save_path}")
    print(f"   Version: {VERSION_TAG}")


def load_model(filename="nozzle_pinn.pt"):
    """
    Load model checkpoint with metadata.

    Returns:
        Tuple of (model, scales, conditions, info)
    """
    load_path = CHECKPOINT_DIR / filename
    checkpoint = torch.load(load_path, map_location='cpu')

    model = NozzlePINN()
    model.load_state_dict(checkpoint['model_state_dict'])

    scales = checkpoint.get('scales', SCALES)
    conditions = checkpoint.get('conditions', CONDITIONS)
    info = {
        'version': checkpoint.get('version', 'unknown'),
        'training_info': checkpoint.get('training_info', {}),
        'random_seed': checkpoint.get('random_seed', None)
    }

    print(f"✓ Loaded checkpoint: {load_path}")
    print(f"  Version: {info['version']}")

    return model, scales, conditions, info


# ============================================================================
# TRAINING
# ============================================================================

def train_nozzle(
    num_epochs=3001,
    lr=1e-3,
    save_path="nozzle_pinn.pt",
    verbose=True
):
    """
    Train nozzle PINN with improved loss weighting and boundary conditions.

    Args:
        num_epochs: Number of training epochs
        lr: Learning rate
        save_path: Checkpoint save path
        verbose: Print progress

    Returns:
        Tuple of (model, device, training_history)
    """
    device = torch.device("cpu")
    model = NozzlePINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Boundary condition targets
    # Inlet: strict enforcement
    t_in = torch.tensor([[
        CONDITIONS['inlet'][k]/SCALES[k]
        for k in ['rho', 'u', 'p', 'T']
    ]], device=device)

    # Outlet: pressure should match ambient (soft enforcement)
    p_amb_norm = CONDITIONS['ambient']['p'] / SCALES['p']
    t_p_out = torch.tensor([[p_amb_norm]], device=device)

    # Collocation points for physics
    x_col = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)

    # Training history
    history = {
        'loss': [],
        'loss_inlet': [],
        'loss_outlet': [],
        'loss_physics': []
    }

    if verbose:
        print("🚀 Starting Nozzle PINN Training")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning Rate: {lr}")
        print(f"   Baseline Thermo: R={CONDITIONS['physics']['R']:.1f}, γ={CONDITIONS['physics']['gamma']:.3f}")
        print("="*60)

    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # 1. Inlet BC (strict)
            pred_in = model(torch.zeros(1, 1, device=device))
            loss_inlet = 100.0 * ((pred_in - t_in)**2).mean()

            # 2. Outlet pressure BC (soft)
            pred_out = model(torch.ones(1, 1, device=device))
            p_exit_norm = pred_out[:, 2:3]
            loss_outlet = 1.0 * ((p_exit_norm - t_p_out)**2).mean()

            # Penalty for exit pressure below ambient (unphysical for unchoked)
            p_exit_phys = p_exit_norm * SCALES['p']
            p_amb = CONDITIONS['ambient']['p']
            if p_exit_phys < p_amb:
                loss_outlet += 10.0 * ((p_exit_phys - p_amb)/p_amb)**2

            # 3. Physics losses
            l_eos, l_mass, l_energy, l_mach = compute_loss(model, x_col, device)
            loss_physics = l_eos + l_mass + l_energy + 0.5 * l_mach

            # Total loss
            loss = loss_inlet + loss_outlet + loss_physics

            loss.backward()
            optimizer.step()

            # Record history
            history['loss'].append(loss.item())
            history['loss_inlet'].append(loss_inlet.item())
            history['loss_outlet'].append(loss_outlet.item())
            history['loss_physics'].append(loss_physics.item())

            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch:4d} | Total: {loss:.2e} | Inlet: {loss_inlet:.2e} | "
                      f"Outlet: {loss_outlet:.2e} | Physics: {loss_physics:.2e}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")

    # Save model with training info
    training_info = {
        'epochs_completed': epoch + 1,
        'final_loss': loss.item(),
        'learning_rate': lr
    }

    save_model(
        model,
        filename=save_path,
        scales=SCALES,
        conditions=CONDITIONS,
        training_info=training_info
    )

    if verbose:
        print("="*60)
        print("✅ Training complete!")

    return model, device, history


# ============================================================================
# VALIDATION & VISUALIZATION
# ============================================================================

def validate_nozzle(model, device, conditions=None, scales=None):
    """
    Validate trained nozzle PINN and compute performance metrics.

    Args:
        model: Trained NozzlePINN
        device: Torch device
        conditions: Optional CONDITIONS (uses global if None)
        scales: Optional SCALES (uses global if None)
    """
    if conditions is None:
        conditions = CONDITIONS
    if scales is None:
        scales = SCALES

    model.eval()
    x_test = torch.linspace(0, 1, 200, device=device).reshape(-1, 1)

    with torch.no_grad():
        preds = model.predict_physical(x_test, scales=scales).cpu().numpy()
        x_plot = x_test.cpu().numpy()

    # Extract exit state
    rho_exit = preds[-1, 0]
    u_exit = preds[-1, 1]
    p_exit = preds[-1, 2]
    T_exit = preds[-1, 3]

    # Mass flow check
    A_exit = conditions['geometry']['A_exit']
    m_dot_exit = rho_exit * u_exit * A_exit
    m_dot_target = conditions['physics']['mass_flow']

    # Thrust calculation
    p_amb = conditions['ambient']['p']
    u_inlet = conditions['inlet']['u']

    # Net thrust: F = m_dot * (u_exit - u_inlet) + (p_exit - p_amb) * A_exit
    F_momentum = m_dot_target * (u_exit - u_inlet)
    F_pressure = (p_exit - p_amb) * A_exit
    F_total = F_momentum + F_pressure

    # Choking check
    gamma = conditions['physics']['gamma']
    is_choked, p_critical = detect_choking(
        conditions['inlet']['p'],
        p_amb,
        gamma
    )

    # Mach number profile
    R = conditions['physics']['R']
    a_sound = np.sqrt(gamma * R * preds[:, 3])
    mach_profile = preds[:, 1] / a_sound

    # Print results
    print("\n" + "="*70)
    print("NOZZLE PERFORMANCE VALIDATION")
    print("="*70)
    print(f"Exit State:")
    print(f"  Velocity:     {u_exit:.1f} m/s")
    print(f"  Temperature:  {T_exit:.1f} K")
    print(f"  Pressure:     {p_exit/1e3:.1f} kPa  (Ambient: {p_amb/1e3:.1f} kPa)")
    print(f"  Mach Number:  {mach_profile[-1]:.3f}")
    print(f"\nMass Flow:")
    print(f"  PINN Exit:    {m_dot_exit:.2f} kg/s")
    print(f"  Target:       {m_dot_target:.2f} kg/s")
    print(f"  Error:        {abs(m_dot_exit - m_dot_target)/m_dot_target*100:.2f}%")
    print(f"\nThrust Components:")
    print(f"  Momentum:     {F_momentum/1e3:.2f} kN")
    print(f"  Pressure:     {F_pressure/1e3:.2f} kN")
    print(f"  NET THRUST:   {F_total/1e3:.2f} kN")
    print(f"\nChoking Status:")
    print(f"  Is Choked:    {is_choked}")
    print(f"  Critical p:   {p_critical/1e3:.1f} kPa")
    print("="*70)

    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Nozzle Flow Profiles (PINN) - {VERSION_TAG}', fontsize=14)

    # Pressure
    axs[0, 0].plot(x_plot, preds[:, 2]/1e3, 'b-', linewidth=2, label='Pressure')
    axs[0, 0].axhline(p_amb/1e3, color='r', linestyle='--', label='Ambient')
    if is_choked:
        axs[0, 0].axhline(p_critical/1e3, color='orange', linestyle=':', label='Critical')
    axs[0, 0].set_ylabel('Pressure (kPa)')
    axs[0, 0].set_xlabel('Normalized Position')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Velocity
    axs[0, 1].plot(x_plot, preds[:, 1], 'g-', linewidth=2)
    axs[0, 1].set_ylabel('Velocity (m/s)')
    axs[0, 1].set_xlabel('Normalized Position')
    axs[0, 1].grid(True, alpha=0.3)

    # Mach Number
    axs[1, 0].plot(x_plot, mach_profile, 'k-', linewidth=2, label='Mach')
    axs[1, 0].axhline(1.0, color='r', linestyle='--', label='Sonic')
    axs[1, 0].set_ylabel('Mach Number')
    axs[1, 0].set_xlabel('Normalized Position')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Temperature
    axs[1, 1].plot(x_plot, preds[:, 3], 'r-', linewidth=2)
    axs[1, 1].set_ylabel('Temperature (K)')
    axs[1, 1].set_xlabel('Normalized Position')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / 'nozzle_validation.png', dpi=150)
    print(f"\n📊 Validation plot saved: {CHECKPOINT_DIR / 'nozzle_validation.png'}")
    plt.show()

    return {
        'u_exit': u_exit,
        'p_exit': p_exit,
        'T_exit': T_exit,
        'mach_exit': mach_profile[-1],
        'thrust_N': F_total,
        'is_choked': is_choked
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Nozzle PINN Training Script")
    print(f"Version: {VERSION_TAG}")
    print(f"Random Seed: {RANDOM_SEED}\n")

    # Train the model
    trained_model, device, history = train_nozzle(
        num_epochs=3001,
        lr=1e-3,
        save_path="nozzle_pinn.pt",
        verbose=True
    )

    # Validate the model
    results = validate_nozzle(trained_model, device)

    print("\n✅ All done!")
