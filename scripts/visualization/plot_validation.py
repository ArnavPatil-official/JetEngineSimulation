import sys
from pathlib import Path
# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

# Data from your final log output
modes = ['Idle', 'Approach', 'Climb', 'Takeoff']
icao_data = [0.244, 0.643, 2.050, 2.327]  # Real World (kg/s)

# These are approximations based on your logs (Fill in exacts if you have them)
# Idle ~ 0.23 kg/s, Approach ~ 0.66 kg/s, Climb ~ 2.10 kg/s, Takeoff ~ 2.68 kg/s
sim_data = [0.2316, 0.6620, 2.1066, 2.6809] 

x = np.arange(len(modes))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, icao_data, width, label='ICAO Data (Trent 1000)', color='black', alpha=0.7)
rects2 = ax.bar(x + width/2, sim_data, width, label='Digital Twin (Mine)', color='royalblue')

ax.set_ylabel('Fuel Flow Rate (kg/s)', fontsize=12)
ax.set_title('Validation: Digital Twin vs. Real Certification Data', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=12)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/validation_chart.png', dpi=300)
print("✅ Validation chart saved!")
plt.show()