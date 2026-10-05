# P8.4b-A1 — IPC handling bleed held at a cited surge-margin floor

Date: 2026-10-01. User decision of 2026-10-01 after the registered P8.4b
checks failed (`1cb32e4`: AE3 APPROACH/IDLE unreachable because the IPC
operating line reaches the map stall line at 0.5 x rated). Known when
written: that failure, the AE3 take-off fuel flow 2.700 kg/s (ICAO 2.327),
and the registered sensitivities. No A4 score exists. The failed check
record stays as it is.

## Rule

- Source: Chapman, "Utilizing Electrical Power Extraction for Stability Bleed
  Reduction within Gas Turbine Engines" (NASA Glenn, AIAA; NTRS 20210017572):
  stability bleed through a variable bleed valve behind the low/intermediate
  compressor, dumped to the bypass, is the standard stall-margin mitigation
  and is scheduled so the compressor keeps **surge margin >= 10 %**.
- Surge margin is pyCycle's constant-speed definition (`StallCalcs.SMN`),
  on unscaled map values: `SMN = ((WcMap/WcMap_stall)/(PRmap/PRmap_stall) - 1)*100`,
  with the stall point at `RlineStall` and the same `NcMap`.
- Handling bleed `beta` = fraction of IPC exit flow, taken at the IPC exit
  state and mixed adiabatically into the bypass stream at bypass pressure
  (enthalpy and composition mixing, P8.2 rule) before the bypass duct.
- At each off-design point the solve first runs with `beta = 0`. If it does
  not converge, or converges with IPC SMN < 10 %, it is re-solved with
  `beta` as a 13th unknown and the residual `(SMN - 10)/100`; a solution
  with `beta < 0` reverts to `beta = 0`. Bounds `0 <= beta <= 0.5`. The
  design point keeps `beta = 0`; its SMN is reported.
- No other input, range, tolerance or check changes. Sensitivity also
  reports the SMN floor at 5 % and 15 % (reported, not gating).

## Checks and score

Re-run the registered P8.4b checks with this rule to a new write-once file,
`outputs/phase8/p84b_a1_g1.json`; A4 is scored once only if it passes.
