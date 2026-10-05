"""GE E3 HPT air-turbine rig, NASA CR-168289 (Timko, 1984).

NTRS 19900019237, PDF page 92/report page 82, Table XIII: only the TEST
column is entered. ICLS/FPS targets and PREDICTION are not observations.
The 0.900 thermodynamic efficiency is stated in the report abstract and is
different from Table XIII's GE-cycle efficiency 0.925, which is not mapped
to the database's isentropic-efficiency quantity. Detailed maps in this report
are plotted and await three-repeat digitisation.
"""

SOURCE_ID = "NASA-CR-168289"
PDF = "19900019237.pdf"
PDF_SHA256 = "2e6f8a691363368dcffe1d8dcedea50e3fe7933f7d303f4c09e335833c2e2d6d"
EXPERIMENT_ID = "GE-E3-HPT-air-rig-1984"


def enter(conn, edb, pdf_dir) -> None:
    edb.add_source(
        conn, SOURCE_ID,
        "Timko, L. P. (1984). Energy Efficient Engine high pressure turbine component test "
        "performance report. NASA CR-168289, General Electric.",
        "nasa_report", "2026-09-29", identifier="NTRS 19900019237",
        pdf_path=(pdf_dir / PDF) if (pdf_dir / PDF).exists() else None,
        pdf_sha256=PDF_SHA256,
        notes="Measured design point entered; map curves and Appendix E cascade readings not entered.")
    edb.add_experiment(
        conn, EXPERIMENT_ID, SOURCE_ID, "GE-E3-HPT-air-rig", "GE E3 two-stage HPT warm-air rig",
        "turbine", "heated_air", "experiment", 4, "B",
        facility="General Electric air-turbine test rig", geometry={"stages": 2, "cooled": True},
        notes="Table XIII gives Pt4/Pt42 (total-to-total) and W41*sqrt(Tt41)/Pt4. "
        "Thermodynamic efficiency 0.900 is stated in the abstract; GE-cycle efficiency "
        "0.925 from Table XIII is a different definition and is not stored as eta_tt.")
    op = f"{EXPERIMENT_ID}-design"
    edb.add_operating_point(conn, op, EXPERIMENT_ID, "measured design operating point")
    edb.add_observation(conn, op, "PR", 5.01, "1", "input",
                        "Table XIII, TEST column (report p. 82; PDF p. 92)")
    edb.add_observation(conn, op, "flow_function", 18.19,
                        "lbm*degR^0.5/(s*psia)", "output",
                        "Table XIII, TEST column (report p. 82; PDF p. 92)")
    edb.add_observation(conn, op, "eta_thermo_cooled", 90.0, "percent", "output",
                        "Abstract: thermodynamic definition (PDF p. 2)")
