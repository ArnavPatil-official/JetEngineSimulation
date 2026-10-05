"""Measured five-stage GE E3 LPT rig points, NASA CR-168290 (1983).

NTRS 19900019247, Appendix H, Configuration 5, report p. 236/PDF p. 251.
Only four clearly legible Block II runs spanning low and high pressure ratio
are transcribed. Appendix H contains many more runs; this is a partial entry
and must not be treated as the complete map. The scan itself marks these pages
poor quality, so the experiment is class C and has no invented uncertainties.
Table XI contains intended design parameters and is not entered as test data.
"""

SOURCE_ID = "NASA-CR-168290"
PDF = "19900019247.pdf"
PDF_SHA256 = "2a9003050b91ae5c7591843b2cb19627e1bad42b3f8f30df5b17e9939c63ddec"
EXPERIMENT_ID = "GE-E3-LPT-five-stage-rig-1983"
TABLE = "Appendix H, Configuration 5 (report p. 236; PDF p. 251)"

# Run: Pt/Ps, Tt39 degR, Pt42 psia, N/sqrt(Tt39) rpm/sqrt(degR),
# W*sqrt(Tt39)/Pt42 lbm*sqrt(degR)/(s*psia), Pt42/Pt55, Tt55 degR,
# eta total-to-total, eta total-to-static. All values are printed test values.
ROWS = {
    539: (5.637, 748.3, 45.046, 121.0, 37.83, 5.027, 492.3, .9261, .8777),
    540: (5.636, 748.7, 45.037, 121.7, 37.86, 5.029, 492.5, .9250, .8767),
    550: (2.700, 752.9, 45.029, 97.3, 37.53, 2.623, 587.8, .8998, .8771),
    553: (2.701, 749.5, 45.014, 106.5, 36.76, 2.623, 585.6, .9094, .8861),
}


def enter(conn, edb, pdf_dir) -> None:
    edb.add_source(
        conn, SOURCE_ID,
        "Bridgeman, M. J., Cherry, D. G., and Pedersen, J. (1983). NASA/GE Energy Efficient "
        "Engine low pressure turbine scaled test vehicle performance report. NASA CR-168290.",
        "nasa_report", "2026-09-29", identifier="NTRS 19900019247",
        pdf_path=(pdf_dir / PDF) if (pdf_dir / PDF).exists() else None,
        pdf_sha256=PDF_SHA256,
        notes="Appendix H contains a larger measured map; four clear rows entered pending full scan review.")
    edb.add_experiment(
        conn, EXPERIMENT_ID, SOURCE_ID, "GE-E3-LPT-Block-II-five-stage-rig",
        "GE E3 Block II five-stage scaled air turbine", "turbine", "cold_air",
        "experiment", 4, "C", facility="General Electric scaled air-turbine test rig",
        geometry={"stages": 5, "scale_factor": .67},
        notes="Appendix H measured Block II Configuration 5, four selected legible runs; "
        "values are not the ICLS design predictions in Table XI. PR is Pt42/Pt55; "
        "PR_ts is Pt42/Ps55. The report defines flow function as W*sqrt(Tt39)/Pt42.")
    for run, row in ROWS.items():
        pr_ts, tt_in, pt_in, speed_param, flow_fn, pr, tt_out, eta_tt, eta_ts = row
        assert pr_ts >= pr > 1 and tt_in > tt_out and 0 < eta_ts <= eta_tt <= 1
        op = f"{EXPERIMENT_ID}-run{run}"
        edb.add_operating_point(conn, op, EXPERIMENT_ID, f"run {run}")
        vals = [("PR_ts", pr_ts, "1", "input"), ("Tt_in", tt_in, "degR", "input"),
                ("Pt_in", pt_in, "psia", "input"),
                ("speed_param", speed_param, "rpm/degR^0.5", "input"),
                ("flow_function", flow_fn, "lbm*degR^0.5/(s*psia)", "output"),
                ("PR", pr, "1", "output"), ("Tt_out", tt_out, "degR", "output"),
                ("eta_tt", eta_tt, "1", "output"), ("eta_ts", eta_ts, "1", "output")]
        for q, v, unit, role in vals:
            edb.add_observation(conn, op, q, v, unit, role, TABLE)
