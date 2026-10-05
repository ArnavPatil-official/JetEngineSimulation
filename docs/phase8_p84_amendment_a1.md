# P8.4-A1 — thermo-matched mode follows pyCycle CEA conventions

Date: 2026-10-01. Prospective amendment before any P8.4 solve or
comparison (only the registration `f3be816` and the map export `67eab80`
exist). Found by reading the pinned pyCycle 4.4.0 source.

1. **Shifting equilibrium everywhere (thermo-matched mode only).** In CEA
   mode every pyCycle flow-station property comes from a chemical-equilibrium
   solve at fixed element amounts: total and static states at (h, P),
   (S, P) or (T, P), not only the burner. The thermo-matched mode therefore
   evaluates every state by Cantera equilibrium (HP, SP or TP) over the
   exported JANAF species at the stream's fixed element amounts. Production
   mode is unchanged: frozen composition outside the burner, as in P8.2/P8.3.
2. **Fuel enthalpy.** The HBTF example leaves the combustor input `mix:h`
   unconnected, so Jet-A(g) enters with h = 0 Btu/lbm on the JANAF scale,
   with elements C12H23 (`reactants['Jet-A(g)']`). The thermo-matched burner
   uses the same mass-averaged enthalpy and element addition.
3. **Air.** Element amounts from pyCycle `CEA_AIR_COMPOSITION`
   (N, O, Ar, C per unit mass), not `O2:1, N2:3.76`.

Registration section 3's "equilibrium at each burner/mixer" is replaced by
item 1 for the thermo-matched mode. Tolerances and checks are unchanged.
