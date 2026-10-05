"""Actual 640k screening plus selected-only top-ten simulator verification."""
from __future__ import annotations

import csv
import math
import os
import resource
import time
from pathlib import Path

from .models import load_product, summarize_draws
from .postprocess import lifecycle_scenarios
from .registration import read_json, sha256_file, write_once


def screen(root,output,reg,reg_sha,context,run):
    from .run import command_spec,csv_once
    from .teacher import ParallelTeacher
    output=Path(output);run.assert_current()
    product=load_product(output/"product.json",require_deployment=False)
    queries=read_json(output/"splits/study.json")
    if len(queries)!=640000:
        raise RuntimeError("Registered study budget changed")
    prediction_path=output/"study_predictions.csv"
    columns=("design_id","draw_id","input_sha256","f_JetA","f_HEFA","f_FT","f_ATJ","thrust_fraction",
        "status","ff_kg_s","T4_K","EI_CO2_kg_kg","CO2_g_s","lifecycle_g_s","nvpm_dEI_number_pct",
        "nvpm_status","nvpm_reason","seed_sd_ff_kg_s","seed_sd_T4_K","diagnostic_unsafe")
    groups=[];invalid=0;started=time.perf_counter_ns()
    with prediction_path.open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for start in range(0,640000,4096):
            run.assert_current();batch=queries[start:start+4096];predictions=product.predict(batch)
            for query,prediction in zip(batch,predictions):
                row={**query,**prediction};writer.writerow({key:row.get(key) for key in columns})
                invalid+=int(prediction["status"]!="predicted")
            groups.extend(summarize_draws(batch,predictions))
            stream.flush();os.fsync(stream.fileno())
    groups.sort(key=lambda g:(g.get("ranking_q95_lifecycle_g_s") if g.get("ranking_q95_lifecycle_g_s") is not None else math.inf,g["candidate_id"]))
    for rank,g in enumerate(groups,start=1):g["conditional_lifecycle_rank"]=rank
    elapsed=(time.perf_counter_ns()-started)/1e9
    bands=[]
    for group in groups:
        row={key:group.get(key) for key in ("candidate_id","draw_count","status","conditional_lifecycle_rank","ranking_q95_lifecycle_g_s","diagnostic_unsafe")}
        for field in ("ff_kg_s","T4_K","CO2_g_s","lifecycle_g_s","nvpm_dEI_number_pct"):
            values=group.get(field,{})
            row.update({field+"_"+stat:values.get(stat) for stat in ("mean","q025","q50","q975","min","max","available_draws")})
        bands.append(row)
    csv_once(output/"study_bands.csv",bands)
    selected_ids=[g["candidate_id"] for g in groups[:10]]
    selected_queries=[q for q in queries if q["design_id"] in selected_ids]
    if len(selected_queries)!=640:
        raise RuntimeError("Fixed top-ten common-draw verification coverage changed")
    write_once(output/"study_topten.json",{"registration_sha256":reg_sha,"ranking_scope":"within_full_surrogate_screen_only",
        "prediction_sha256":sha256_file(prediction_path),"selected_ids":selected_ids,
        "queries":selected_queries,"global_simulator_overlap":None,"global_simulator_regret":None})
    spec=command_spec(root,output,context,reg_sha,"study",[__import__('sys').executable,"P8-S","selected640verification"])
    write_once(root/spec["owner_lease"]["snapshot_path"],(root/spec["owner_lease"]["path"]).read_bytes())
    write_once(root/spec["command_spec_path"],spec)
    teacher=ParallelTeacher(spec,product.properties,product.public,product.draws,6,run=run)
    audit_started=time.perf_counter()
    try:actual=teacher(selected_queries)
    finally:teacher.close()
    actual_bands=summarize_draws(selected_queries,actual)
    predicted_selected=product.predict(selected_queries)
    rows=[];fferrors=[];Terrors=[]
    for query,prediction,reference in zip(selected_queries,predicted_selected,actual):
        row={**query,**reference,"predicted_ff_kg_s":prediction["ff_kg_s"],"predicted_T4_K":prediction["T4_K"],
            "predicted_lifecycle_g_s":prediction["lifecycle_g_s"],"verification_scope":"selected_candidates_only"}
        rows.append(row)
        if reference["status"]=="converged" and prediction["status"]=="predicted":
            fferrors.append(abs(prediction["ff_kg_s"]-reference["ff_kg_s"]));Terrors.append(abs(prediction["T4_K"]-reference["T4_K"]))
    csv_once(output/"study_topten_teacher.csv",rows)
    import numpy as np
    scenarios=lifecycle_scenarios(product.properties);scenario_rows=[]
    for design_id in selected_ids:
        q=next(q for q in selected_queries if q["design_id"]==design_id)
        ff=[r["ff_kg_s"] for r in actual if r["design_id"]==design_id]
        rates=[]
        if len(ff)==64 and all(value is not None and math.isfinite(value) for value in ff):
            for scenario in scenarios:
                factor=sum(q[f"f_{fuel}"]*product.properties["fuels"][product.properties["surrogates"][fuel]]["lhv_liquid_MJ_kg"]*scenario[fuel]
                           for fuel in ("JetA","HEFA","FT","ATJ"))
                rates.append(float(np.quantile(np.asarray(ff)*factor,.95)))
        scenario_rows.append({"design_id":design_id,"common_scenarios":1000,"state":"COMPLETE" if rates else "INVALID_REFERENCE",
            "selected_only_lifecycle_q95_common_LCEF_q025_q50_q975":np.quantile(rates,[.025,.5,.975]).tolist() if rates else None})
    result={"state":"COMPLETE","queries":640000,"compositions":10000,"cpu64_total_seconds":elapsed,
        "peak_rss_native_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"invalid_prediction_rows":invalid,
        "selected_teacher_requests":640,"selected_verification_seconds":time.perf_counter()-audit_started,
        "selected_reference_converged":sum(r["status"]=="converged" for r in actual),
        "selected_ff_MAE_kg_s":float(np.mean(fferrors)) if fferrors else None,"selected_ff_max_kg_s":max(fferrors) if fferrors else None,
        "selected_T4_MAE_K":float(np.mean(Terrors)) if Terrors else None,"selected_T4_max_K":max(Terrors) if Terrors else None,
        "selected_actual_bands":actual_bands,"common_LCEF_scenarios":scenario_rows,
        "global_simulator_overlap":None,"global_simulator_regret":None,"verification_scope":"selected_candidates_only",
        "diagnostic_unsafe":True}
    write_once(output/"study_metrics.json",result)
    run.assert_current()
    return result
