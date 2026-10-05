"""Measured GE E3 core combustor line, NASA CR-168069 (Stearns et al., 1982).

NTRS 19900019243, Table XIX (report p. 239; PDF p. 265). The table's raw
"Measured" EI columns are entered; FPS-cycle-adjusted Table XX is excluded.
DMS 195 has emissions but no station conditions; 198/199 have station
conditions but no emissions; 207 is a post-test zero. We enter 187-194
and 196-206 with the 198/199 target cells absent.
The source's W36 metric and parenthesised imperial columns differ by roughly
4 %, so W36 is left out until the source inconsistency is resolved.
"""

SOURCE_ID = "NASA-CR-168069"
PDF = "19900019243.pdf"
PDF_SHA256 = "ee3abbf95000caa003f263735195fb4b519204b3debbb2364dd6ef2b74ceaf8a"
EXPERIMENT_ID = "GE-E3-core-combustor-1982"
TABLE = "Table XIX, measured columns (report p. 239; PDF p. 265)"

# DMS: P3 kPa, T3 degC, f/a, P4 kPa, CO/HC/NOx g/kg, chemical efficiency.
# None = blank in the printed table. Every value below was checked against
# the rotated page image, rather than trusted to the scan's OCR text.
ROWS = {
    187: (409.5, 219, .0156, 386.1, 42.8, 2.84, 5.5, .987),
    188: (320.6, 187, .0173, 304.7, 62.1, .86, 4.6, .985),
    189: (584.0, 268, .0142, 550.9, 27.5, .14, 7.1, .993),
    190: (577.1, 266, .0139, 554.7, 91.7, 52.2, 3.7, .926),
    191: (790.8, 309, .0144, 743.3, 67.8, 24.5, 5.2, .960),
    192: (1067.3, 358, .0168, 1007.3, 30.6, 3.37, 7.9, .989),
    193: (1196.9, 383, .0192, 1132.8, 16.2, 1.20, 10.3, .995),
    194: (1811.3, 401, .0184, 1712.7, 9.6, .23, 11.7, .998),
    196: (2489.0, 459, .0234, 2360.1, .7, .09, 18.4, .999),
    197: (2826.2, 484, .0252, 2689.0, .5, .16, 23.7, .999),
    198: (2842.0, 484, .0256, 2695.9, None, None, None, None),
    199: (2857.2, 483, .0255, 2714.5, None, None, None, None),
    200: (1774.0, 398, .0194, 1680.3, 11.9, .84, 10.5, .996),
    201: (579.2, 278, .0159, 546.1, 29.5, .81, 7.1, .992),
    202: (586.7, 276, .0157, 553.0, 28.8, 2.41, 7.2, .991),
    203: (399.2, 222, .0164, 378.5, 45.6, .28, 5.4, .989),
    204: (320.6, 189, .0174, 304.7, 62.6, .32, 4.1, .985),
    205: (281.3, 168, .0176, 268.2, 73.2, .23, 4.0, .983),
    206: (206.8, 122, .0201, 198.6, 89.1, 2.96, 3.13, .976),
}


def enter(conn, edb, pdf_dir) -> None:
    edb.add_source(
        conn, SOURCE_ID,
        "Stearns, E. M., et al. (1982). Energy Efficient Engine core design and performance "
        "report. NASA CR-168069, General Electric.",
        "nasa_report", "2026-09-29", identifier="NTRS 19900019243",
        pdf_path=(pdf_dir / PDF) if (pdf_dir / PDF).exists() else None,
        pdf_sha256=PDF_SHA256,
        notes="Table XIX raw core-test observations only; modeled FPS corrections in Table XX excluded.")
    edb.add_experiment(
        conn, EXPERIMENT_ID, SOURCE_ID, "GE-E3-core-test-1982", "GE E3 core engine combustor",
        "combustor", "combustion_products", "experiment", 4, "B",
        facility="General Electric E3 core test", geometry={"double_annular": True},
        notes="Chemical combustion efficiency is from gas analysis, not a cycle enthalpy "
        "efficiency. Pilot-only and staged operation are mixed in this table; DMS numbers "
        "and source notes identify the mode. W36 withheld: source kg/s and lbm/s columns "
        "are systematically inconsistent by about 4 %. DMS 195 lacks station conditions; "
        "DMS 207 is a post-test zero. Missing emissions at DMS 198/199 remain absent rows.")
    for dms, row in ROWS.items():
        p3, t3, far, p4, co, hc, nox, eta = row
        assert p3 > p4 > 0 and 0 < far < .1 and 0 < t3 < 1000
        op = f"{EXPERIMENT_ID}-DMS{dms}"
        edb.add_operating_point(conn, op, EXPERIMENT_ID, f"DMS {dms}")
        vals = [("Pt_in", p3, "kPa", "input"), ("Tt_in", t3, "degC", "input"),
                ("FAR", far, "1", "input"), ("Pt_out", p4, "kPa", "output"),
                ("EI_CO", co, "g/kg", "output"), ("EI_HC", hc, "g/kg", "output"),
                ("EI_NOx", nox, "g/kg", "output"),
                ("eta_chemical", eta, "1", "output")]
        for quantity, value, unit, role in vals:
            if value is not None:
                edb.add_observation(conn, op, quantity, value, unit, role, TABLE)
