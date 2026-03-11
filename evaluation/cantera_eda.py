import matplotlib.pyplot as plt
import cantera as ct
import numpy as np
from pathlib import Path

# Define base path for mechanisms (relative to project root)
base_path = Path(__file__).resolve().parent.parent / "data"

# Define mechanisms to evaluate
mechanisms = [
    "creck_c1c16_full.yaml",
    "n_dodecane_hychem.yaml",
    "isooctane.yaml",
    "A1highT.yaml",
    "A2NOx.yaml",
]

def soundspeed(gas):
    """Calculate speed of sound for a gas."""
    return np.sqrt(gas.cp / gas.cv * ct.gas_constant / gas.mean_molecular_weight * gas.T)

def run_isentropic(mech_file):
    """Isentropic expansion test for a given mechanism."""

    print(f"\n=== Running Isentropic Analysis: {mech_file} ===")

    mech_path = base_path / mech_file
    if not mech_path.exists():
        print(f"  WARNING: Mechanism not found → skipping: {mech_path}")
        return None

    try:
        gas = ct.Solution(str(mech_path))
    except Exception as e:
        print(f"  ERROR: Could not load mechanism {mech_file}: {e}")
        return None

    T0 = 1500.0
    p0 = 10 * ct.one_atm

    if "NC12H26" in gas.species_names:
        gas.TPX = T0, p0, {"NC12H26": 1e-5, "O2": 1, "N2": 3.76}
    elif "IC8H18" in gas.species_names:
        gas.TPX = T0, p0, {"IC8H18": 1e-5, "O2": 1, "N2": 3.76}
    else:
        gas.TPX = T0, p0, {"O2": 1, "N2": 3.76}

    s0 = gas.s
    h0 = gas.h

    areas = []
    machs = []
    temps = []
    prats = []

    p_ratios = np.logspace(-3, 0, 20)

    for pr in p_ratios:
        p = pr * p0
        gas.SP = s0, p

        # compute v safely
        v_sq = 2 * (h0 - gas.h)
        if v_sq <= 0 or not np.isfinite(v_sq):
            continue
        v = np.sqrt(v_sq)

        if not np.isfinite(v) or gas.density <= 0 or not np.isfinite(gas.density):
            continue

        a = soundspeed(gas)
        M = v / a
        A = 1.0 / (gas.density * v)  # mdot=1

        if not np.isfinite(A):
            continue

        areas.append(A)
        machs.append(M)
        temps.append(gas.T)
        prats.append(pr)

    if len(areas) == 0:
        print(f"  WARNING: No valid states for {mech_file}")
        return None

    areas = np.array(areas)
    machs = np.array(machs)
    temps = np.array(temps)
    prats = np.array(prats)

    A_min = np.min(areas)
    area_ratio = areas / A_min

    results = np.column_stack([area_ratio, machs, temps, prats])

    print(f"  Successfully computed isentropic curve for {mech_file} "
          f"({len(results)} valid points)")
    return results

# Store results for all mechanisms
all_results = {}

for mech in mechanisms:
    data = run_isentropic(mech)
    if data is not None:
        all_results[mech] = np.array(data)


# --- VISUALIZATION SECTION ---

# 1. Mach number vs Area Ratio
plt.figure(figsize=(10,6))
for mech, arr in all_results.items():
    area_ratio = arr[:,0]
    mach = arr[:,1]
    plt.plot(area_ratio, mach, marker="o", label=mech)

plt.xlabel("Area Ratio (A/A*)")
plt.ylabel("Mach Number")
plt.title("Mach Number vs Area Ratio Across Mechanisms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 2. Temperature vs Pressure Ratio
plt.figure(figsize=(10,6))
for mech, arr in all_results.items():
    T = arr[:,2]
    p_ratio = arr[:,3]
    plt.plot(p_ratio, T, marker="s", label=mech)

plt.xlabel("Pressure Ratio (p/p0)")
plt.ylabel("Temperature (K)")
plt.title("Temperature vs Pressure Ratio Across Mechanisms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. Mach vs Pressure Ratio (Nozzle behavior)
plt.figure(figsize=(10,6))
for mech, arr in all_results.items():
    mach = arr[:,1]
    p_ratio = arr[:,3]
    plt.plot(p_ratio, mach, marker="^", label=mech)

plt.xlabel("Pressure Ratio (p/p0)")
plt.ylabel("Mach Number")
plt.title("Mach Number vs Pressure Ratio Across Mechanisms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 4. Temperature vs Mach
plt.figure(figsize=(10,6))
for mech, arr in all_results.items():
    mach = arr[:,1]
    T = arr[:,2]
    plt.plot(mach, T, marker="D", label=mech)

plt.xlabel("Mach Number")
plt.ylabel("Temperature (K)")
plt.title("Temperature vs Mach Number Across Mechanisms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 5. Summary: overlay area ratio–temperature curves
plt.figure(figsize=(10,6))
for mech, arr in all_results.items():
    area_ratio = arr[:,0]
    T = arr[:,2]
    plt.plot(area_ratio, T, marker=".", label=mech)

plt.xlabel("Area Ratio (A/A*)")
plt.ylabel("Temperature (K)")
plt.title("Temperature vs Area Ratio Across Mechanisms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
