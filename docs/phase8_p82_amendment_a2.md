# P8.2-A2 — enthalpy inversion accuracy correction

Date: 2026-09-29. This corrective numerical amendment is committed before
the P8.2 G1 verdict. The G1 threshold remains 1e-10 for energy closure.

Known when written: an uncommitted development smoke probe of the P8.2
implementation at the frozen AE3 take-off configuration returned A1
energy relative error 1.14e-16 and A2 maximum 1.25e-9. A separate two-stream
mix probe returned 4.5e-10 energy relative error. Source inspection showed
`ThermoPhase::setState_HP` defaults to tolerance 1e-9. No held-out row,
ablation calibration, or official G1 verdict was computed. These preliminary
failures are kept here rather than erased.

For every `h(T,Y)` inversion in the P8.2 state/mixing module, call Cantera
`setState_HP(h, P, 1e-13)`. This tightens a numerical root tolerance without
changing the model or the registered closure threshold. The final G1 record
will include the above development probe and the post-correction numerical
closure. If the fixed rule still fails, record G1 failure and do not begin
P8.3.
