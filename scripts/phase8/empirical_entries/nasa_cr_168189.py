"""
NASA CR-168189 (Leach, K. P., 1983): Energy Efficient Engine high-pressure
turbine component rig performance test report (Pratt & Whitney, PWA-5594-243).
NTRS 19850021643. Entered from Table 5.3.1-II (PDF page 51, rotated scan,
report page 39): full-stage turbine warm rig test results, 27 test points.

Values are entered in the table's parenthesised US units (psia, degF, in),
which carry more digits than the SI columns. The SI columns, and the
report's own clearance-adjusted efficiency column, are used only as
transcription checks (CHECKS below); a failed check stops the build.

Definitions from the report:
- efficiency: thermodynamic efficiency sum(m_i dh_i) / sum(m_i dh_i') over
  the primary, coolant and leakage streams, ideal expansion to turbine exit
  pressure (section 4.3.1) -> eta_thermo_cooled, measured (mass-averaged).
- speed parameter N/sqrt(T_T), rpm/sqrt(degR) (design 13232 rpm at 2940 degR
  gives 244.0 = the stated design value) -> speed_param.
- corrected flow = flow parameter W sqrt(T_T)/P_T, lbm sqrt(degR)/(s psia)
  (Table 3.2.2-I "FPin, (W sqrt(TT)/PT)") -> flow_function.
- pressure ratio: "Press Ratio" / "Expansion Ratio"; the total/static basis is
  not stated in the text -> PR_reported.
- reaction: static-pressure reaction (section 5.3.1.1) -> reaction_p.
Uncertainties, Table 3.5.3-I/II at 95 % (2 sigma), halved to 1 sigma:
efficiency 0.38 pt, pressure ratio 0.04, speed parameter 0.16 (reported);
temperature bias 0.41 degF and precision 0.38 degF combined (derived); tip
clearance precision 0.001 in (reported); flow function from the air-flow
bias 0.20 % and precision 0.30 % only (derived; P and T terms not included).
Pressure precision is given as % of an unstated transducer range: no sigma.

Scan reading: digits marked "scan-unclear" in the location field were hard
to read; the value entered is the best reading and is consistent with its
neighbours. Source inconsistency: point 20 lists 0.459 MPa and 66.26 psia
(= 0.4569 MPa); the psia value is entered.
"""

import math

SOURCE_ID = "NASA-CR-168189"
PDF = "19850021643.pdf"
PDF_SHA256 = "9b70c6c1ba4bb515e035571e54bfcfabaad7facdc099b7738793491ea3db754d"
EXPERIMENT_ID = "E3-HPT-rig-PW-1983"
TABLE = "Table 5.3.1-II (p. 39; PDF p. 51)"

# pt: (phase, PR, speed param, flow fn, reaction %, Pt MPa, Pt psia, Tt_in C, Tt_in F,
#      Tt_out C, Tt_out F, eta meas %, clearance cm, clearance in, eta adj % @0.019 in)
ROWS = {
    1:  ("I",   4.01, 244.8, 17.19, 43.1, 0.454, 65.87, 315.4, 599.8, 128.2, 262.9, 88.37, 0.053, 0.021, 88.54),
    2:  ("I",   4.03, 234.1, 17.05, 43.7, 0.475, 68.87, 367.2, 693.0, 163.2, 325.9, 87.45, 0.053, 0.021, 87.62),
    3:  ("I",   3.91, 222.3, 17.07, 42.9, 0.462, 67.05, 366.6, 692.0, 167.9, 334.3, 86.64, 0.058, 0.023, 86.98),
    4:  ("I",   3.98, 253.1, 17.18, 42.5, 0.446, 64.74, 309.7, 589.5, 125.0, 257.1, 88.88, 0.048, 0.019, 88.88),
    5:  ("I",   3.49, 244.4, 17.13, 39.2, 0.465, 67.44, None,  601.1, 142.8, 289.1, 88.03, 0.050, 0.020, 88.12),
    6:  ("I",   3.48, 235.5, 17.20, 39.4, 0.474, 68.72, 311.1, 592.1, 141.5, 286.7, 87.35, 0.053, 0.021, 87.52),
    7:  ("I",   3.49, 223.0, 17.15, 39.8, 0.477, 69.21, 312.8, 595.1, 142.4, 288.4, 86.50, 0.060, 0.024, 86.93),
    9:  ("I",   4.43, 245.3, 17.22, 45.0, 0.541, 78.44, 311.9, 593.5, 117.1, 242.9, 88.29, 0.050, 0.020, 88.38),
    11: ("II",  4.06, 243.3, 17.38, 42.9, 0.465, 67.39, 148.8, 300.0, None,  None,  None,  0.066, 0.026, None),
    14: ("II",  4.05, 243.9, 17.37, 43.8, 0.463, 67.15, 148.8, 299.9, None,  None,  None,  0.048, 0.019, None),
    15: ("III", 4.00, 246.2, 17.21, 42.9, 0.459, 66.56, 309.3, 588.9, None,  None,  None,  0.053, 0.021, None),
    18: ("III", 4.03, 244.6, 17.15, 43.7, 0.458, 66.46, 315.7, 600.4, 127.1, 260.9, 88.06, 0.055, 0.022, 88.32),
    19: ("III", 4.00, 244.1, 17.17, 42.7, 0.458, 66.36, 316.3, 601.5, None,  None,  None,  0.053, 0.021, None),
    20: ("III", 4.00, 244.2, 17.21, 43.2, 0.459, 66.26, 316.9, 602.5, None,  None,  None,  0.053, 0.021, None),
    21: ("III", 4.02, 244.2, 17.18, 43.4, 0.457, 66.21, 316.2, 601.3, 127.6, 261.8, 88.27, 0.053, 0.021, 88.44),
    28: ("III", 4.01, 244.8, 17.28, 43.1, 0.454, 65.82, 314.1, 597.5, 129.2, 264.6, 88.31, 0.050, 0.020, 88.40),
    22: ("IV",  4.30, 234.4, 17.22, 44.8, 0.523, 75.84, 314.5, 598.1, 123.2, 253.9, 87.36, 0.055, 0.022, 87.62),
    23: ("IV",  3.47, 253.8, 17.17, 39.2, 0.449, 65.18, 315.6, 600.1, 141.6, 287.0, 88.72, 0.048, 0.019, 88.72),
    24: ("IV",  4.04, 244.8, 17.19, 43.1, 0.457, 66.26, 310.5, 591.0, 124.3, 255.9, 88.39, 0.053, 0.021, 88.56),
    29: ("IV",  4.75, 243.8, 17.36, 46.3, 0.602, 87.38, 288.1, 550.6, 98.9,  210.1, 87.56, 0.053, 0.021, 87.73),
    25: ("V",   4.00, 244.2, 17.19, 43.1, 0.454, 65.87, 316.1, 601.0, None,  None,  None,  0.053, 0.021, None),
    26: ("V",   4.00, 245.0, 17.17, 43.2, 0.453, 65.77, 316.3, 601.5, None,  None,  None,  0.053, 0.021, None),
    27: ("V",   4.00, 244.5, 17.19, 43.2, 0.453, 65.67, 316.6, 602.0, None,  None,  None,  0.053, 0.021, None),
}
SCAN_UNCLEAR = {(4, "reaction_p"), (6, "PR_reported"), (18, "flow_function"), (26, "reaction_p"),
                (26, "flow_function")}
KNOWN_SOURCE_INCONSISTENCY = {(20, "Pt")}

SIGMA = {   # 1 sigma, original units, kind
    "eta_thermo_cooled": (0.19, "reported"),
    "PR_reported": (0.02, "reported"),
    "speed_param": (0.08, "reported"),
    "Tt_in": (math.hypot(0.41, 0.38) / 2, "derived"),
    "Tt_out": (math.hypot(0.41, 0.38) / 2, "derived"),
    "tip_clearance": (0.0005, "reported"),
}
FLOW_REL_SIGMA = math.hypot(0.20, 0.30) / 2 / 100


def check_row(pt: int, r: tuple) -> None:
    """Transcription checks against the table's own redundant columns."""
    _, _, _, _, _, mpa, psia, tc, tf, toc, tof, eta, ccm, cin, eadj = r
    if (pt, "Pt") not in KNOWN_SOURCE_INCONSISTENCY:
        assert abs(psia * 0.006894757293168361 - mpa) <= 0.00051, f"pt {pt}: MPa vs psia"
    for c, f in ((tc, tf), (toc, tof)):
        if c is not None and f is not None:
            assert abs((f - 32.0) / 1.8 - c) <= 0.1, f"pt {pt}: degC vs degF"
    assert abs(cin * 2.54 - ccm) <= 0.0015, f"pt {pt}: cm vs in"
    if eadj is not None:
        # report: +0.09 % efficiency per 0.001 in clearance reduction; the table
        # applies it from the measured clearance to 0.019 in
        pred = eta + 0.085 * (cin - 0.019) / 0.001
        assert abs(pred - eadj) <= 0.011, f"pt {pt}: clearance-adjusted efficiency"


def enter(conn, edb, pdf_dir) -> None:
    edb.add_source(conn, SOURCE_ID,
                   "Leach, K. P. (1983). Energy Efficient Engine high-pressure turbine component rig "
                   "performance test report. NASA CR-168189 (PWA-5594-243), Pratt & Whitney Aircraft.",
                   "nasa_report", "2026-09-29", identifier="NTRS 19850021643",
                   pdf_path=(pdf_dir / PDF) if (pdf_dir / PDF).exists() else None,
                   pdf_sha256=PDF_SHA256)
    edb.add_experiment(
        conn, EXPERIMENT_ID, SOURCE_ID, "PW-E3-HPT-cooled-rig", "E3 HPT full-stage cooled warm rig",
        "turbine", "heated_air", "experiment", 4, "A", facility="Pratt & Whitney test stand X-203",
        geometry={"stages": 1, "vanes": 24, "blades": 54, "design_speed_rpm": 13232,
                  "design_clearance_in": 0.0185, "cooled": True},
        notes="Single-stage cooled HPT rig, heated air (inlet ~590-600 degF); efficiency is the "
              "report's cooled thermodynamic efficiency (all streams), measured, not the clearance-"
              "adjusted column (adjustment: +0.09 %/0.001 in). Pressure-ratio basis not stated.")
    for pt, r in ROWS.items():
        check_row(pt, r)
        phase, pr, sp, ff, reac, _mpa, psia, _tc, tf, _toc, tof, eta, _ccm, cin, _eadj = r
        op = f"{EXPERIMENT_ID}-tp{pt:02d}"
        edb.add_operating_point(conn, op, EXPERIMENT_ID, f"test point {pt} (phase {phase})")
        vals = [("PR_reported", pr, "1", "input"), ("speed_param", sp, "rpm/degR^0.5", "input"),
                ("Pt_in", psia, "psia", "input"), ("Tt_in", tf, "degF", "input"),
                ("tip_clearance", cin, "in", "input"),
                ("flow_function", ff, "lbm*degR^0.5/(s*psia)", "output"),
                ("reaction_p", reac, "percent", "output"),
                ("Tt_out", tof, "degF", "output"), ("eta_thermo_cooled", eta, "percent", "output")]
        for q, v, unit, role in vals:
            if v is None:
                continue          # blank in the table: no row
            loc = TABLE + (" [scan-unclear]" if (pt, q) in SCAN_UNCLEAR else "")
            if (pt, q.replace("_in", "")) in KNOWN_SOURCE_INCONSISTENCY:
                loc += " [source: SI column 0.459 MPa disagrees]"
            if q == "flow_function":
                sig, kind = ff * FLOW_REL_SIGMA, "derived"
            else:
                sig, kind = SIGMA.get(q, (None, None))
            edb.add_observation(conn, op, q, v, unit, role, loc, sigma=sig, sigma_kind=kind)
