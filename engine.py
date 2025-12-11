"""
INTEGRATED JET ENGINE SIMULATION
=================================
File: JetEngineSimulation/integrated_engine.py
Usage: python integrated_engine.py

Combines: Compressor → Combustor → Turbine PINN → Nozzle PINN
Uses: CRECK C1-C16 Mechanism for realistic Biofuel/Jet-A chemistry.
"""

import torch
import cantera as ct
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, Any, List

# ============================================================================
# 0. PATH SETUP & IMPORTS
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from simulation.compressor.compressor import Compressor
    from simulation.combustor.combustor import Combustor
    from simulation.turbine.turbine import NormalizedTurbinePINN 
    from simulation.nozzle.nozzle import NozzlePINN
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


# ============================================================================
# 1. FUEL BLEND CLASS (Updated for CRECK Mechanism)
# ============================================================================

class LocalFuelBlend:
    """
    Represents a blend of aviation fuels.
    Mapped to species available in CRECK C1-C16 mechanism.
    """
    def __init__(self, jet_a1: float, hefa: float = 0.0, ft: float = 0.0, atj: float = 0.0):
        total = jet_a1 + hefa + ft + atj
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Fuel blend must sum to 1.0 (got {total:.3f})")
        
        self.jet_a1 = jet_a1
        self.hefa = hefa
        self.ft = ft
        self.atj = atj
    
    def as_composition_string(self) -> str:
        """
        Convert to Cantera composition string using CRECK species names.
        """
        # Mapping for CRECK Mechanism
        # NC12H26: n-Dodecane (Jet A / HEFA surrogate)
        # NC10H22: n-Decane (FT surrogate)
        # IC8H18: Iso-Octane (ATJ surrogate)
        # Note: Verify these exact string keys exist in your YAML file.
        # Sometimes they are 'nC12H26' or 'C12H26'. 
        # Assuming standard CRECK naming:
        formulas = {
            'jet_a1': 'NC12H26',   
            'hefa': 'NC12H26',     
            'ft': 'NC10H22',       
            'atj': 'IC8H18'        
        }
        
        # Aggregate duplicates (Fixes "Duplicate Key" error)
        species_counts = {}
        for fuel_type, fraction in [('jet_a1', self.jet_a1), 
                                    ('hefa', self.hefa), 
                                    ('ft', self.ft), 
                                    ('atj', self.atj)]:
            if fraction > 0:
                sp_name = formulas[fuel_type]
                species_counts[sp_name] = species_counts.get(sp_name, 0.0) + fraction
                
        return ', '.join([f"{sp}:{val:.4f}" for sp, val in species_counts.items()])
    
    def __repr__(self):
        return (f"Blend(JetA={self.jet_a1:.0%}, HEFA={self.hefa:.0%}, FT={self.ft:.0%}, ATJ={self.atj:.0%})")


# ============================================================================
# 2. INTEGRATED ENGINE CLASS
# ============================================================================

class IntegratedTurbofanEngine:
    def __init__(self, 
                 turbine_model_path: str = "turbine_pinn.pt",
                 nozzle_model_path: str = "nozzle_pinn.pt",
                 mechanism_file: str = "data/creck_c1c16_full.yaml"): # <-- TARGET FILE
        
        self.device = torch.device("cpu")
        print(f"🔧 Initializing Engine with {mechanism_file}...")

        # 1. Load PINNs
        # Unpack 3 values: model, scales, (ignore conditions)
        self.turbine_model, self.t_scales, _ = self._load_pinn(turbine_model_path, NormalizedTurbinePINN)
        # Unpack 3 values: model, scales, conditions
        self.nozzle_model, self.n_scales, self.n_conds = self._load_pinn(nozzle_model_path, NozzlePINN)
        
        # 2. Initialize Physics Components
        self.mech_file = mechanism_file
        
        if not os.path.exists(self.mech_file):
             print(f"❌ Critical Error: Mechanism file not found at {self.mech_file}")
             print("Please check the path.")
             sys.exit(1)

        try:
            self.gas = ct.Solution(self.mech_file)
            print("   - Cantera Mechanism Loaded Successfully")
        except Exception as e:
            print(f"❌ Cantera Load Error: {e}")
            sys.exit(1)

        # 3. Initialize Component Classes
        # Pass gas object to Compressor if required, else fallback
        try:
            self.compressor = Compressor(gas=self.gas, eta_c=0.85, pi_c=43.2)
        except TypeError:
            self.compressor = Compressor(eta_c=0.85, pi_c=43.2)

        self.combustor = Combustor(mechanism_file=self.mech_file)
        
        # Specs (Trent 1000)
        self.bypass_ratio = 9.1
        self.design_mass_flow = 79.9 
        
        print(f"✅ Integrated Engine Ready.")

    def _load_pinn(self, path, ModelClass):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        data = torch.load(path, map_location=self.device)
        model = ModelClass().to(self.device)
        model.load_state_dict(data['model_state_dict'])
        model.eval()
        return model, data['scales'], data.get('conditions', None)
    
    def simulate(self, fuel_blend, phi: float = 0.4):
        # 1. Compressor
        P_amb, T_amb = 101325.0, 288.15
        comp_out = self.compressor.compute_outlet_state(T_in=T_amb, p_in=P_amb)
        
        # 2. Combustor
        fuel_str = fuel_blend.as_composition_string()
        comb_out = self.combustor.run(
            T_in=comp_out['T_out'],
            p_in=comp_out['p_out'],
            fuel_blend=fuel_blend, # Pass object
            phi=phi
        )
        
        # Mass Flow
        m_dot_core_air = self.design_mass_flow 
        # Try to get f from combustor output, else approximate
        f_ratio = comb_out.get('f', 0.03) 
        m_dot_fuel = m_dot_core_air * f_ratio
        m_dot_total = m_dot_core_air + m_dot_fuel
        
        # 3. Turbine PINN (With Offset)
        T_turb_in = comb_out['T_out']
        x_turb_exit = torch.ones(1, 1, device=self.device)
        with torch.no_grad():
            t_pred = self.turbine_model.predict_physical(x_turb_exit).cpu().numpy()[0]
        
        T_train_in = 1700.0 
        delta_T = T_turb_in - T_train_in
        T_turb_exit = t_pred[3] + delta_T
        P_turb_exit = t_pred[2] 
        
        # 4. Nozzle PINN (With Offset & Scaling)
        x_nozz_exit = torch.ones(1, 1, device=self.device)
        with torch.no_grad():
            # Manual Un-normalization 
            n_out_norm = self.nozzle_model(x_nozz_exit).cpu().numpy()[0]
            rho_n = n_out_norm[0] * self.n_scales['rho']
            u_n   = n_out_norm[1] * self.n_scales['u']
            p_n   = n_out_norm[2] * self.n_scales['p']
            T_n   = n_out_norm[3] * self.n_scales['T']

        T_nozz_train_in = 1005.0 
        delta_T_nozz = T_turb_exit - T_nozz_train_in
        T_nozz_exit = T_n + delta_T_nozz
        
        # Velocity V ~ sqrt(T)
        u_exit = u_n * np.sqrt(T_nozz_exit / T_n)
        
        # Thrust
        P_exit = p_n # Assume pressure profile is geometrically dominated
        A_exit = self.n_conds['geometry']['A_exit']
        F_mom = m_dot_total * u_exit
        F_pres = (P_exit - P_amb) * A_exit
        F_core = F_mom + F_pres
        
        # 5. Bypass
        m_dot_bypass = m_dot_core_air * self.bypass_ratio
        T_fan_exit = T_amb * (1.6**0.286)
        u_bypass = np.sqrt(2 * 1005 * (T_fan_exit - T_amb))
        F_bypass = m_dot_bypass * u_bypass
        
        F_total = F_core + F_bypass
        
        # 6. Metrics
        tsfc = (m_dot_fuel * 3600) / (F_total / 1000)
        co2_emission = m_dot_fuel * 3.16 # Approx
        
        return {
            'Thrust_Total_kN': F_total/1000,
            'TSFC': tsfc,
            'T_Combustor': T_turb_in,
            'T_EGT': T_nozz_exit
        }

# ============================================================================
# MAIN
# ============================================================================

def compare_fuel_blends():
    # Use your specific data path
    mech_path = "/Users/arnavpatil/Desktop/JetEngineSimulation/data/creck_c1c16_full.yaml"
    
    engine = IntegratedTurbofanEngine(mechanism_file=mech_path)
    
    # Define Blends
    # Note: Ensure LocalFuelBlend uses species names that exist in your CRECK file
    blends = [
        LocalFuelBlend(jet_a1=1.0),
        LocalFuelBlend(jet_a1=0.5, hefa=0.5),
        LocalFuelBlend(jet_a1=0.5, atj=0.5),
    ]
    
    results = []
    print("\n" + "="*70)
    print("RUNNING CRECK BIOFUEL COMPARISON")
    print("="*70)
    
    for blend in blends:
        print(f"Testing: {blend}")
        try:
            res = engine.simulate(blend)
            res['Blend'] = str(blend)
            results.append(res)
        except Exception as e:
            print(f"❌ Failed: {e}")
            # If species error persists, print the error details
            if "Species" in str(e):
                print("   -> Check species names in LocalFuelBlend vs YAML file")
    
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*70)
        print(df[['Blend', 'Thrust_Total_kN', 'TSFC', 'T_EGT']].to_string(index=False))
        print("="*70)
        df.to_csv("creck_results.csv", index=False)

if __name__ == "__main__":
    compare_fuel_blends()