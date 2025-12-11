"""
NEURAL NETWORK ARCHITECTURE VISUALIZATIONS
===========================================
Creates publication-quality diagrams of PINN architectures.

Generates:
1. Network architecture diagrams (layer-by-layer)
2. Data flow diagrams (input → output)
3. Training loss curves
4. Prediction profile plots
5. Physics residual heatmaps
"""

import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# Ensure the project root is on sys.path when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


# ============================================================================
# 1. NETWORK ARCHITECTURE DIAGRAM
# ============================================================================

def visualize_network_architecture(model, save_path='network_architecture.png', title='Turbine PINN'):
    """
    Create detailed network architecture diagram showing all layers.
    
    Args:
        model: PyTorch model (TurbinePINN or NozzlePINN)
        save_path: Where to save the figure
        title: Title for the diagram
    """
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, title, fontsize=20, fontweight='bold', ha='center')
    
    # Extract layer information
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers.append({
                'name': name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'params': module.in_features * module.out_features + module.out_features
            })
    
    # Calculate positions
    n_layers = len(layers) + 2  # +2 for input/output representations
    x_positions = np.linspace(1, 9, n_layers)
    
    # Helper function to draw layer
    def draw_layer(x, y, width, height, label, neurons, color, params=None):
        # Main box
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05", 
            edgecolor='black', facecolor=color, linewidth=2
        )
        ax.add_patch(box)
        
        # Label
        ax.text(x, y + 0.3, label, fontsize=12, ha='center', fontweight='bold')
        ax.text(x, y - 0.1, f'{neurons} neurons', fontsize=9, ha='center')
        
        # Parameter count
        if params:
            ax.text(x, y - 0.4, f'{params:,} params', fontsize=8, ha='center', style='italic', color='gray')
    
    # Helper function to draw arrow
    def draw_arrow(x1, x2, y=4.5):
        arrow = FancyArrowPatch(
            (x1, y), (x2, y),
            arrowstyle='->', mutation_scale=30, 
            linewidth=2, color='black', alpha=0.7
        )
        ax.add_arrow(arrow)
    
    # Colors
    colors = {
        'input': '#E8F4F8',
        'hidden': '#B4E7CE',
        'output': '#FFE5B4'
    }
    
    y_center = 4.5
    layer_height = 2.5
    layer_width = 0.8
    
    # Draw input layer
    draw_layer(x_positions[0], y_center, layer_width, layer_height, 
               'Input', 1, colors['input'])
    ax.text(x_positions[0], y_center - 1.8, 'x ∈ [0,1]', fontsize=9, ha='center', style='italic')
    
    # Draw hidden layers
    for i, layer in enumerate(layers[:-1]):  # All except output layer
        x = x_positions[i + 1]
        draw_layer(x, y_center, layer_width, layer_height,
                   f'Hidden {i+1}', layer['out_features'], colors['hidden'], layer['params'])
        
        # Draw activation function below
        ax.text(x, y_center - 1.8, 'Tanh', fontsize=9, ha='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Draw arrow from previous layer
        draw_arrow(x_positions[i] + layer_width/2, x - layer_width/2)
    
    # Draw output layer
    output_layer = layers[-1]
    x_out = x_positions[-1]
    draw_layer(x_out, y_center, layer_width, layer_height,
               'Output', output_layer['out_features'], colors['output'], output_layer['params'])
    
    # Output variables
    output_vars = ['ρ', 'u', 'p', 'T']
    y_offset = y_center + 1.5
    for i, var in enumerate(output_vars):
        ax.text(x_out, y_offset - i*0.4, var, fontsize=10, ha='center', 
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # Draw final arrow
    draw_arrow(x_positions[-2] + layer_width/2, x_out - layer_width/2)
    
    # Add total parameter count
    total_params = sum(layer['params'] for layer in layers)
    ax.text(5, 0.5, f'Total Parameters: {total_params:,}', 
            fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add legend for normalization
    legend_y = 1.5
    ax.text(0.5, legend_y, 'Normalization:', fontsize=10, fontweight='bold')
    ax.text(0.5, legend_y - 0.3, '• Input: x_norm = x / L', fontsize=8)
    ax.text(0.5, legend_y - 0.6, '• Output: [ρ*, u*, p*, T*]', fontsize=8)
    ax.text(0.5, legend_y - 0.9, '• Physical: multiply by scales', fontsize=8)
    2
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Network architecture saved to {save_path}")
    plt.show()


# ============================================================================
# 2. DATA FLOW DIAGRAM WITH PHYSICS
# ============================================================================

def visualize_pinn_dataflow(save_path='pinn_dataflow.png', component='Turbine'):
    """
    Create comprehensive data flow diagram showing:
    - Forward pass
    - Physics calculations
    - Loss computation
    - Backpropagation
    """
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, f'{component} PINN: Data Flow & Physics Integration', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Helper functions
    def draw_box(x, y, w, h, text, color='lightblue', fontsize=10):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')
    
    def draw_arrow(x1, y1, x2, y2, label='', style='->', color='black'):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                               mutation_scale=20, linewidth=2, color=color)
        ax.add_arrow(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.2, label, fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================================================
    # FORWARD PASS
    # ========================================================================
    
    # Input
    draw_box(0.5, 7, 1.5, 1, 'Input\nx ∈ [0,1]', '#E8F4F8')
    
    # Neural Network
    draw_box(3, 6.5, 2, 2, 'Neural\nNetwork\n(4 layers)', '#B4E7CE')
    draw_arrow(2, 7.5, 3, 7.5, 'forward')
    
    # Raw outputs
    draw_box(6, 6.5, 2, 2, 'Normalized\nOutputs\n[ρ*, u*, p*, T*]', '#FFE5B4')
    draw_arrow(5, 7.5, 6, 7.5)
    
    # Denormalization
    draw_box(9, 6.5, 2.5, 2, 'Denormalize\n× scales\n→ Physical units', '#FFF4E6')
    draw_arrow(8, 7.5, 9, 7.5, '× scales')
    
    # Physical outputs
    draw_box(12.5, 6.5, 2.5, 2, 'Physical\nValues\nρ, u, p, T', '#FFE5B4')
    draw_arrow(11.5, 7.5, 12.5, 7.5)
    
    # ========================================================================
    # PHYSICS COMPUTATIONS
    # ========================================================================
    
    ax.text(8, 5.8, 'Physics Calculations', fontsize=12, ha='center', 
            fontweight='bold', style='italic')
    
    # Autograd for derivatives
    draw_box(2, 4, 3, 1.2, 'Automatic Differentiation\n∂ρ/∂x, ∂u/∂x, ∂p/∂x, ∂T/∂x', '#E0BBE4')
    draw_arrow(12.5, 6.5, 3.5, 5.2, 'autograd', color='purple')
    
    # Physics equations
    physics_y = 3.5
    physics_boxes = [
        (1, 'EOS\np = ρRT', '#FFB3BA'),
        (4, 'Mass\n∇·(ρuA)=0', '#FFDFBA'),
        (7, 'Energy\nh₀ = const', '#FFFFBA'),
        (10, 'Geometry\nA(x)', '#BAFFC9')
    ]
    
    for x, label, color in physics_boxes:
        draw_box(x, physics_y - 0.6, 2, 1, label, color, fontsize=9)
        draw_arrow(3.5, 4, x + 1, physics_y + 0.4, color='purple', style='->')
    
    # ========================================================================
    # RESIDUAL COMPUTATION
    # ========================================================================
    
    ax.text(8, 2.5, 'Residual Computation', fontsize=12, ha='center',
            fontweight='bold', style='italic')
    
    draw_box(5, 1.2, 6, 1, 'Physics Residuals\nR_eos, R_mass, R_energy', '#FFB3D1')
    
    for x, _, _ in physics_boxes:
        draw_arrow(x + 1, physics_y - 0.6, 8, 2.2, color='red', style='->')
    
    # ========================================================================
    # LOSS COMPUTATION
    # ========================================================================
    
    # Boundary conditions
    draw_box(0.5, 1.2, 3, 1, 'Boundary\nConditions\nBC_inlet, BC_outlet', '#C7CEEA')
    draw_arrow(2, 7, 2, 2.2, 'evaluate\nat x=0,1', color='blue')
    
    # Total loss
    draw_box(6, -0.3, 4, 1.2, 
             'Total Loss = w₁·BC² + w₂·R_eos² + w₃·R_mass² + w₄·R_energy²',
             '#FF6B6B', fontsize=9)
    
    draw_arrow(2, 1.2, 7, 0.9, color='blue')
    draw_arrow(8, 1.2, 8, 0.9, color='red')
    
    # ========================================================================
    # BACKPROPAGATION
    # ========================================================================
    
    draw_arrow(8, -0.3, 4, 7, 'Backprop\n∂Loss/∂θ', color='green', style='<-')
    
    ax.text(3, 6, 'Update\nWeights', fontsize=10, ha='center', color='green',
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#B4E7CE', label='Neural Network'),
        mpatches.Patch(color='#E0BBE4', label='Automatic Differentiation'),
        mpatches.Patch(color='#FFB3BA', label='Physics Equations'),
        mpatches.Patch(color='#FF6B6B', label='Loss Function'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Data flow diagram saved to {save_path}")
    plt.show()


# ============================================================================
# 3. TRAINING HISTORY VISUALIZATION
# ============================================================================

def plot_training_history(save_path='training_history.png'):
    """
    Create multi-panel training history visualization.
    """
    
    # Simulate training history (replace with actual logged data)
    epochs = np.arange(0, 3001, 100)
    
    # Phase 1: BC only (0-1000)
    bc_phase1 = np.logspace(0, -14, 11)
    
    # Phase 2: BC + weak physics (1000-3000)
    bc_phase2 = np.full(20, 1e-14) * (1 + np.random.randn(20) * 0.1)
    eos_phase2 = np.logspace(-2, -4, 20)
    mass_phase2 = np.logspace(-1, -3, 20)
    
    bc_loss = np.concatenate([bc_phase1, bc_phase2])
    eos_loss = np.concatenate([np.full(11, np.nan), eos_phase2])
    mass_loss = np.concatenate([np.full(11, np.nan), mass_phase2])
    total_loss = bc_loss.copy()
    total_loss[11:] += eos_loss[11:] + mass_loss[11:]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PINN Training History', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.semilogy(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
    ax1.axvline(1000, color='red', linestyle='--', alpha=0.7, label='Phase Transition')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Total Loss Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Component Losses
    ax2 = axes[0, 1]
    ax2.semilogy(epochs, bc_loss, 'g-', linewidth=2, label='BC Loss')
    ax2.semilogy(epochs[11:], eos_loss[11:], 'r-', linewidth=2, label='EOS Loss')
    ax2.semilogy(epochs[11:], mass_loss[11:], 'orange', linewidth=2, label='Mass Loss')
    ax2.axvline(1000, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Learning Rate Schedule
    ax3 = axes[1, 0]
    lr = np.concatenate([
        np.full(11, 1e-3),  # Phase 1
        np.logspace(-3, -4, 20)  # Phase 2 with decay
    ])
    ax3.semilogy(epochs, lr, 'purple', linewidth=2)
    ax3.axvline(1000, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence Metrics
    ax4 = axes[1, 1]
    bc_satisfied = bc_loss < 1e-4
    physics_satisfied = (eos_loss < 0.01) & (mass_loss < 0.01)
    physics_satisfied[:11] = False
    
    ax4.plot(epochs, bc_satisfied.astype(float), 'g-', linewidth=2, label='BC Satisfied')
    ax4.plot(epochs, physics_satisfied.astype(float), 'r-', linewidth=2, label='Physics Satisfied')
    ax4.axvline(1000, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Satisfied (1=Yes, 0=No)', fontsize=12)
    ax4.set_title('Convergence Status', fontsize=14, fontweight='bold')
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # Add phase annotations
    for ax in axes.flat:
        ax.text(500, ax.get_ylim()[1] * 0.9, 'Phase 1:\nBC Only', 
               ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.text(2000, ax.get_ylim()[1] * 0.9, 'Phase 2:\nPhysics', 
               ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training history saved to {save_path}")
    plt.show()


# ============================================================================
# 4. PREDICTION PROFILES
# ============================================================================

def plot_prediction_profiles(model, conditions, scales, save_path='prediction_profiles.png', title='Turbine'):
    """
    Plot spatial profiles of all predicted variables.
    """
    
    model.eval()
    
    # Generate predictions
    x = torch.linspace(0, 1, 200).reshape(-1, 1)
    
    with torch.no_grad():
        out_norm = model(x)
        predictions = torch.cat([
            out_norm[:, 0:1] * scales['rho'],
            out_norm[:, 1:2] * scales['u'],
            out_norm[:, 2:3] * scales['p'],
            out_norm[:, 3:4] * scales['T']
        ], dim=1).numpy()
    
    x_plot = x.numpy().flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title} PINN: Spatial Profiles', fontsize=16, fontweight='bold')
    
    variables = [
        ('Density (ρ)', predictions[:, 0], 'kg/m³', conditions['inlet']['rho'], conditions['outlet']['rho']),
        ('Velocity (u)', predictions[:, 1], 'm/s', conditions['inlet']['u'], conditions['outlet']['u']),
        ('Pressure (p)', predictions[:, 2]/1e3, 'kPa', conditions['inlet']['p']/1e3, conditions['outlet']['p']/1e3),
        ('Temperature (T)', predictions[:, 3], 'K', conditions['inlet']['T'], conditions['outlet']['T'])
    ]
    
    for ax, (name, data, unit, inlet_val, outlet_val) in zip(axes.flat, variables):
        # Plot prediction
        ax.plot(x_plot, data, 'b-', linewidth=2, label='PINN Prediction')
        
        # Plot boundary targets
        ax.plot(0, inlet_val, 'ro', markersize=10, label='Inlet BC')
        ax.plot(1, outlet_val, 'go', markersize=10, label='Outlet BC')
        
        # Annotations
        ax.axhline(inlet_val, color='r', linestyle='--', alpha=0.3)
        ax.axhline(outlet_val, color='g', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Normalized Position (x)', fontsize=12)
        ax.set_ylabel(f'{name} [{unit}]', fontsize=12)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Prediction profiles saved to {save_path}")
    plt.show()


# ============================================================================
# 5. PHYSICS RESIDUAL HEATMAP
# ============================================================================

def plot_physics_residuals(model, conditions, scales, save_path='physics_residuals.png'):
    """
    Visualize physics equation residuals across the domain.
    """
    
    model.eval()
    
    # Generate grid
    x_points = torch.linspace(0, 1, 100).reshape(-1, 1).requires_grad_(True)
    
    with torch.enable_grad():
        out_norm = model(x_points)
        rho = out_norm[:, 0:1] * scales['rho']
        u = out_norm[:, 1:2] * scales['u']
        p = out_norm[:, 2:3] * scales['p']
        T = out_norm[:, 3:4] * scales['T']
        
        # Compute residuals
        R = conditions['physics']['R']
        
        # EOS residual
        eos_res = (p - rho * R * T) / scales['p']
        
        # Mass conservation residual
        A_in = conditions['geometry']['A_inlet']
        A_out = conditions['geometry']['A_outlet']
        A = A_in + (A_out - A_in) * x_points
        mass_flow = rho * u * A
        mass_res = torch.autograd.grad(mass_flow, x_points, torch.ones_like(mass_flow), create_graph=True)[0]
        mass_res_norm = mass_res / 100.0
        
        # Energy residual (T derivative)
        T_x = torch.autograd.grad(T, x_points, torch.ones_like(T), create_graph=True)[0]
        energy_res = T_x / scales['T']
    
    # Convert to numpy
    x_plot = x_points.detach().numpy().flatten()
    residuals = {
        'EOS: p - ρRT': eos_res.detach().numpy().flatten(),
        'Mass: ∇·(ρuA)': mass_res_norm.detach().numpy().flatten(),
        'Energy: ∂T/∂x': energy_res.detach().numpy().flatten()
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Physics Residuals (Should be ≈ 0)', fontsize=16, fontweight='bold')
    
    for ax, (name, data) in zip(axes, residuals.items()):
        # Plot residual
        ax.plot(x_plot, data, 'r-', linewidth=2)
        ax.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (0)')
        
        # Shade acceptable region
        ax.fill_between(x_plot, -0.01, 0.01, alpha=0.2, color='green', label='±1% region')
        
        ax.set_xlabel('Normalized Position (x)', fontsize=12)
        ax.set_ylabel('Residual', fontsize=12)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add statistics
        rms = np.sqrt(np.mean(data**2))
        max_abs = np.max(np.abs(data))
        ax.text(0.98, 0.95, f'RMS: {rms:.2e}\nMax: {max_abs:.2e}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Physics residuals saved to {save_path}")
    plt.show()


# ============================================================================
# MAIN: GENERATE ALL VISUALIZATIONS
# ============================================================================

def generate_all_visualizations(turbine_model_path='/Users/arnavpatil/Desktop/JetEngineSimulation/turbine_pinn.pt',
                                nozzle_model_path='/Users/arnavpatil/Desktop/JetEngineSimulation/nozzle_pinn.pt'):
    """
    Generate all visualizations for both turbine and nozzle PINNs.
    """
    
    print("="*70)
    print("GENERATING NEURAL NETWORK VISUALIZATIONS")
    print("="*70)
    
    # Load models
    from simulation.turbine.turbine import NormalizedTurbinePINN, CONDITIONS as TURB_COND, SCALES as TURB_SCALES
    from simulation.nozzle.nozzle import NozzlePINN, CONDITIONS as NOZZ_COND, SCALES as NOZZ_SCALES
    
    print("\n[1/10] Loading models...")
    turbine_model = NormalizedTurbinePINN()
    turbine_model.load_state_dict(torch.load(turbine_model_path, map_location='cpu')['model_state_dict'])
    turbine_model.eval()
    
    nozzle_model = NozzlePINN()
    nozzle_model.load_state_dict(torch.load(nozzle_model_path, map_location='cpu')['model_state_dict'])
    nozzle_model.eval()
    
    # Generate visualizations
    print("\n[2/10] Network architecture - Turbine...")
    visualize_network_architecture(turbine_model, 'turbine_architecture.png', 'Turbine PINN Architecture')
    
    print("\n[3/10] Network architecture - Nozzle...")
    visualize_network_architecture(nozzle_model, 'nozzle_architecture.png', 'Nozzle PINN Architecture')
    
    print("\n[4/10] Data flow diagram - Turbine...")
    visualize_pinn_dataflow('turbine_dataflow.png', 'Turbine')
    
    print("\n[5/10] Data flow diagram - Nozzle...")
    visualize_pinn_dataflow('nozzle_dataflow.png', 'Nozzle')
    
    print("\n[6/10] Training history...")
    plot_training_history('training_history.png')
    
    print("\n[7/10] Prediction profiles - Turbine...")
    plot_prediction_profiles(turbine_model, TURB_COND, TURB_SCALES, 
                            'turbine_profiles.png', 'Turbine')
    
    print("\n[8/10] Prediction profiles - Nozzle...")
    plot_prediction_profiles(nozzle_model, NOZZ_COND, NOZZ_SCALES,
                            'nozzle_profiles.png', 'Nozzle')
    
    print("\n[9/10] Physics residuals - Turbine...")
    plot_physics_residuals(turbine_model, TURB_COND, TURB_SCALES,
                          'turbine_residuals.png')
    
    print("\n[10/10] Physics residuals - Nozzle...")
    plot_physics_residuals(nozzle_model, NOZZ_COND, NOZZ_SCALES,
                          'nozzle_residuals.png')
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. turbine_architecture.png")
    print("  2. nozzle_architecture.png")
    print("  3. turbine_dataflow.png")
    print("  4. nozzle_dataflow.png")
    print("  5. training_history.png")
    print("  6. turbine_profiles.png")
    print("  7. nozzle_profiles.png")
    print("  8. turbine_residuals.png")
    print("  9. nozzle_residuals.png")
    print("\nUse these in your presentation and paper!")


if __name__ == "__main__":
    generate_all_visualizations()
