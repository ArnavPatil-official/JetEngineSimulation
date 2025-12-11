"""
BAYESIAN OPTIMIZATION FOR FUEL BLENDS
======================================
Uses Optuna to find optimal SAF blends that minimize TSFC and CO2
while maximizing exergy efficiency.

Constraint: Jet-A1 >= 50%
"""

# Ensure project root on sys.path when running directly
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import numpy as np
import pandas as pd
from engine import IntegratedTurbofanEngine, FuelBlend


class FuelBlendOptimizer:
    """
    Bayesian optimizer for fuel blend composition.
    """
    
    def __init__(self, engine: IntegratedTurbofanEngine):
        """
        Args:
            engine: Initialized IntegratedTurbofanEngine
        """
        self.engine = engine
        self.n_evaluations = 0
    
    def objective(self, trial: optuna.Trial) -> tuple:
        """
        Multi-objective optimization function.
        
        Returns:
            tuple: (TSFC, CO2_specific, -eta_exergy)
                   Optuna minimizes all objectives
        """
        
        # Sample fuel fractions using Dirichlet-like approach
        # Ensure Jet-A1 >= 0.5
        
        # Sample SAF fractions (sum to <= 0.5)
        max_saf = 0.5
        
        hefa_frac = trial.suggest_float('hefa', 0.0, max_saf)
        remaining_saf = max_saf - hefa_frac
        
        ft_frac = trial.suggest_float('ft', 0.0, remaining_saf)
        remaining_saf -= ft_frac
        
        atj_frac = trial.suggest_float('atj', 0.0, remaining_saf)
        
        # Jet-A1 makes up the rest
        jet_a1_frac = 1.0 - (hefa_frac + ft_frac + atj_frac)
        
        # Ensure valid blend
        if jet_a1_frac < 0.5 or jet_a1_frac > 1.0:
            # Should not happen, but return bad values if it does
            return 1000.0, 1000.0, 1000.0
        
        try:
            # Create fuel blend
            blend = FuelBlend(
                jet_a1=jet_a1_frac,
                hefa=hefa_frac,
                ft=ft_frac,
                atj=atj_frac
            )
            
            # Run simulation
            result = self.engine.simulate(blend)
            
            self.n_evaluations += 1
            
            # Objectives to minimize:
            # 1. TSFC (lower is better)
            # 2. CO2 emissions (lower is better)
            # 3. -Exergy efficiency (negate to convert max to min)
            
            return (
                result['TSFC'],                    # kg/kN-hr
                result['CO2_specific'],            # g/kN-s
                -result['eta_exergy']              # Negative for minimization
            )
            
        except Exception as e:
            print(f"⚠️  Simulation failed: {e}")
            return 1000.0, 1000.0, 1000.0
    
    def optimize(self, n_trials: int = 50, timeout: int = 3600) -> pd.DataFrame:
        """
        Run multi-objective Bayesian optimization.
        
        Args:
            n_trials: Number of fuel blends to test
            timeout: Maximum time in seconds
        
        Returns:
            DataFrame with Pareto-optimal solutions
        """
        
        print("="*70)
        print("BAYESIAN OPTIMIZATION: FUEL BLEND SEARCH")
        print("="*70)
        print(f"Objectives:")
        print(f"  1. Minimize TSFC")
        print(f"  2. Minimize CO2 emissions")
        print(f"  3. Maximize exergy efficiency")
        print(f"\nConstraint: Jet-A1 >= 50%")
        print(f"Search space: {n_trials} trials\n")
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=['minimize', 'minimize', 'minimize'],  # All objectives minimized
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name='fuel_blend_optimization'
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        print(f"\n✅ Optimization complete: {self.n_evaluations} evaluations")
        
        # Extract Pareto-optimal solutions
        pareto_trials = study.best_trials
        
        results = []
        for trial in pareto_trials:
            params = trial.params
            values = trial.values
            
            jet_a1 = 1.0 - (params['hefa'] + params['ft'] + params['atj'])
            
            results.append({
                'Trial': trial.number,
                'Jet-A1 (%)': jet_a1 * 100,
                'HEFA (%)': params['hefa'] * 100,
                'FT (%)': params['ft'] * 100,
                'ATJ (%)': params['atj'] * 100,
                'TSFC (kg/kN-hr)': values[0],
                'CO2 (g/kN-s)': values[1],
                'η_exergy (%)': -values[2] * 100  # Convert back to positive
            })
        
        df_pareto = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print(f"PARETO-OPTIMAL SOLUTIONS ({len(pareto_trials)} found)")
        print("="*70)
        print(df_pareto.to_string(index=False))
        print("="*70)
        
        return df_pareto, study
    
    def plot_pareto_front(self, study: optuna.Study, save_path: str = 'pareto_front.png'):
        """
        Visualize Pareto front (requires optuna visualization).
        """
        try:
            import plotly.graph_objects as go
            from optuna.visualization import plot_pareto_front
            
            fig = plot_pareto_front(study, target_names=['TSFC', 'CO2', '-η_exergy'])
            fig.write_html('pareto_front.html')
            print(f"\n📊 Pareto front saved to 'pareto_front.html'")
            
        except ImportError:
            print("⚠️  Install plotly for visualization: pip install plotly")


# ============================================================================
# HELPER: GENERATE CANDIDATE BLENDS
# ============================================================================

def generate_candidate_blends(n_samples: int = 20, seed: int = 42) -> list[FuelBlend]:
    """
    Generate random fuel blends for testing.
    Uses Dirichlet distribution to ensure sum = 1.0 and Jet-A1 >= 0.5
    
    Args:
        n_samples: Number of blends to generate
        seed: Random seed
    
    Returns:
        List of FuelBlend objects
    """
    
    np.random.seed(seed)
    blends = []
    
    for _ in range(n_samples):
        # Sample SAF fractions (max 50%)
        saf_fractions = np.random.dirichlet([1, 1, 1])  # HEFA, FT, ATJ
        saf_fractions *= np.random.uniform(0.0, 0.5)  # Scale to max 50%
        
        hefa, ft, atj = saf_fractions
        jet_a1 = 1.0 - saf_fractions.sum()
        
        # Ensure Jet-A1 >= 50%
        if jet_a1 >= 0.5:
            blends.append(FuelBlend(
                jet_a1=float(jet_a1),
                hefa=float(hefa),
                ft=float(ft),
                atj=float(atj)
            ))
    
    return blends


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    from engine import IntegratedTurbofanEngine
    
    # Initialize engine
    print("Initializing engine...")
    engine = IntegratedTurbofanEngine(
        turbine_model_path="turbine_pinn.pt",
        nozzle_model_path="nozzle_pinn.pt",
        mechanism_file="gri30.yaml"
    )
    
    # Option 1: Test random candidates
    print("\n" + "█"*70)
    print("OPTION 1: TESTING RANDOM CANDIDATE BLENDS")
    print("█"*70)
    
    candidates = generate_candidate_blends(n_samples=10)
    
    candidate_results = []
    for i, blend in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] Testing: {blend}")
        result = engine.simulate(blend)
        
        candidate_results.append({
            'Blend': str(blend),
            'TSFC': result['TSFC'],
            'CO2': result['CO2_specific'],
            'η_exergy': result['eta_exergy'] * 100
        })
    
    df_candidates = pd.DataFrame(candidate_results)
    print("\n" + "="*70)
    print("CANDIDATE RESULTS")
    print("="*70)
    print(df_candidates.to_string(index=False))
    
    # Option 2: Bayesian Optimization
    print("\n\n" + "█"*70)
    print("OPTION 2: BAYESIAN OPTIMIZATION")
    print("█"*70)
    
    user_input = input("\nRun Bayesian optimization? (y/n): ").lower()
    
    if user_input == 'y':
        optimizer = FuelBlendOptimizer(engine)
        
        n_trials = int(input("Number of trials (default=50): ") or "50")
        
        df_pareto, study = optimizer.optimize(n_trials=n_trials)
        
        # Save results
        df_pareto.to_csv('pareto_optimal_blends.csv', index=False)
        print("\n💾 Pareto-optimal blends saved to 'pareto_optimal_blends.csv'")
        
        # Try to plot
        optimizer.plot_pareto_front(study)
    
    else:
        print("Skipping Bayesian optimization.")
