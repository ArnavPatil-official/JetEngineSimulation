# P7.1 surrogate property check

Tolerances fixed before computation (`data/fuel_properties_v7.yaml`, commit `4b8eeda`): hydrogen mass % within ±0.3 points; liquid-basis LHV (gas-phase CRECK LHV − 0.360 MJ/kg n-dodecane heat of vaporization) within ±0.4 MJ/kg of the measured value; Jet A aromatic mass % within ±6 points of 18.7.

| Surrogate | Production role | Formula | H mass % (target) | LHV gas | LHV liquid basis (measured) | Aromatics mass % (target) | H | LHV | Arom. |
|---|---|---|---|---|---|---|---|---|---|
| JetA_dooley2012 | JetA | C9.92H19.43 | 14.12 (13.77) | 43.739 | 43.379 (43.1) | 26.1 (18.7) | **fail** | pass | **fail** |
| JetA_dooley2010 | spread only | C8.61H17.28 | 14.41 (13.77) | 43.918 | 43.558 (43.1) | 18.5 (18.7) | **fail** | **fail** | pass |
| FT_dooley2012 | FT | C10.08H22.15 | 15.58 (15.30) | 44.519 | 44.159 (—) | 0.0 (—) | pass | — | — |
| HEFA_liang2025 | HEFA | C12.00H26.00 | 15.39 (15.30) | 44.462 | 44.102 (—) | 0.0 (—) | pass | — | — |
| ATJ_C1matched | ATJ | C12.50H27.00 | 15.35 (15.39) | 44.385 | 44.025 (43.9) | 0.0 (—) | pass | pass | — |

v5 Jet-A1 (pure n-dodecane) for comparison: H 15.39 %, LHV gas 44.462 MJ/kg (reproduces manifest E10, 44.462).

Reading: the production Jet A surrogate passes the heating-value check, which is the property the thermodynamic cycle uses, and fails the hydrogen and aromatic checks. Its target fuel is Jet-A POSF 4658; the measured targets are NJFCP A-2 (POSF 10325), and the A-2 hydrogen value here is computed from the HyChem average formula C11.4H21.7. The nvPM relations therefore use the reference hydrogen contents (13.8 % Jet A-1, 15.30 % neat SAF; Teoh et al. 2022), not surrogate hydrogen. Registered outcomes; not re-tuned.
