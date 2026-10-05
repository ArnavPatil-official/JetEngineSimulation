"""Freeze every candidate before one locked score pass; no target-driven choice."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path

from .inputs import FUELS, MODES, canonical_query, feature_rows, named_queries
from .models import ensemble_prediction, forward_cpu64, load_product, training_envelope
from .postprocess import derived_outputs
from .registration import artifact_hashes, read_json, sha256_file, verify_artifacts, write_once
from .thermo import Thermo
from .train import prediction_metrics, read_dataset, validation_pass


def named_product_queries(draws, public):
    rows = []
    for central in named_queries(draws, public):
        if not central["in_product_API"]:
            continue
        for draw_id in ("central", *[f"draw_{i:02d}" for i in range(64)]):
            row = canonical_query(dict(central, draw_id=draw_id), draws, public)
            row.update(fuel=central["fuel"], op=central["op"], named_case_id=central["named_case_id"],
                       design_id=central["named_case_id"], prefix_index=len(rows))
            rows.append(row)
    return rows


def _params(output, member):
    import numpy as np
    with np.load(output / member["weight_path"], allow_pickle=False) as archive:
        return {key: archive[key].astype(np.float64) for key in archive.files if key.startswith("layers.")}


def seal_predictions(output, reg, reg_sha, context, run, *, backend=None):
    import numpy as np
    output = Path(output)
    validation = read_json(output / "validation.json")
    from simulation.ml_backend import resolve_backend
    backend = resolve_backend(backend)
    selection = read_json(output / "selection.json")["selection"]
    public, draws = read_json(output / "public_inputs.json"), read_json(output / "fixed_draws.json")
    named = named_product_queries(draws, public)
    write_once(output / "splits/named_product.json", named)
    query_sets = {name: read_json(output / f"splits/{name}.json") for name in ("test", "ranking_test")}
    query_sets["named"] = named
    paths = []
    for member in validation["members"]:
        run.assert_current()
        scaler = member["feature_scaler"]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        for name, queries in query_sets.items():
            values = forward_cpu64(_params(output, member), (feature_rows(queries, draws, public)-mean)/scale, backend)
            stream = io.BytesIO()
            np.savez(stream, ff=values[0], T4=values[1], Y4=values[2],
                     input_sha256=np.asarray([q["input_sha256"] for q in queries]))
            path = f"sealed_predictions/{member['arm']}/N{member['N']}/seed{member['seed']}_{name}.npz"
            write_once(output / path, stream.getvalue()); paths.append(path)
    frozen = {"registration_sha256": reg_sha, "selection_sha256": sha256_file(output / "selection.json"),
              "validation_sha256": sha256_file(output / "validation.json"),
              "predictions": artifact_hashes(output, paths),
              "weights": artifact_hashes(output, [m["weight_path"] for m in validation["members"]]),
              "inputs": artifact_hashes(output, ["splits/test.json", "splits/ranking_test.json", "splits/named_product.json"])}
    write_once(output / "predictions_freeze.json", frozen)
    N = selection["Mphys"]["N"]
    members = [m for m in validation["members"] if m["arm"] == "Mphys" and m["N"] == N]
    train = read_json(output / "splits/train.json")
    manifest = read_json(output / "manifest.json")
    artifacts = artifact_hashes(output, ["frozen_properties.json", "public_inputs.json", "fixed_draws.json", "selection.json",
        "predictions_freeze.json"] + [m["weight_path"] for m in members])
    bundle = {"schema_version": 1, "registration_id": reg["id"], "registration_sha256": reg_sha,
        "binary_sha256": context.binary_sha256, "consumer_identity": context.identity,
        "training_backend":validation.get("training_backend", {}).get("backend", backend),
        "score_backend":"torch" if backend == "torch" else "numpy",
        "score_device":"cpu", "score_dtype":"float64",
        "scientific_sources": manifest["scientific_sources"], "selected_N": N,
        "model": {"seeds": [42,43,44], "output_dimension": 494, "hidden_widths": [128]*4, "activation": "silu"},
        "properties_path": "frozen_properties.json", "public_inputs_path": "public_inputs.json",
        "fixed_draws_path": "fixed_draws.json", "artifacts": artifacts,
        "members": [{"seed": m["seed"], "weights_path": m["weight_path"], "feature_scaler": m["feature_scaler"]} for m in members],
        "training_envelope": training_envelope(train), "selected_training_envelope": training_envelope(train[:N]),
        "nominal_domain": {"simplex": list(FUELS), "thrust_fraction": [.07,1], "draw_ids": [f"draw_{i:02d}" for i in range(64)]}}
    if getattr(context, "execution_profile", None) == "pc":
        bundle["execution_profile"] = "pc"
    if getattr(context, "simulator_backend", None) == "python":
        bundle.update(simulator=context.simulator_identity,
                      simulator_identity_sha256=context.simulator_identity_sha256)
    write_once(output / "product.json", bundle)
    run.assert_current()


def effect_floor(rows, *, tolerance=1e-12):
    """Fixed central 48 contrasts; undefined contrasts deny the floor."""
    lookup = {(row["fuel"], row["op"]): row for row in rows}
    effects, zero, invalid = [], [], []
    for pathway in ("HEFA", "FT", "ATJ"):
        for percent in (10,20,30,50):
            for op, _ in MODES:
                key = (f"{pathway}-{percent}", op)
                a, b = lookup.get(key), lookup.get(("JetA", op))
                try:
                    if a is None or b is None or a.get("status") != "converged" or b.get("status") != "converged":
                        raise ValueError("invalid reference")
                    value = abs(float(a["ff"])-float(b["ff"]))
                    if not math.isfinite(value):
                        raise ValueError("nonfinite reference")
                except (ValueError, TypeError, KeyError):
                    invalid.append(list(key)); continue
                item = {"fuel": key[0], "op": op, "absolute_effect_kg_s": value}
                (effects if value > tolerance else zero).append(item)
    floor = min(row["absolute_effect_kg_s"] for row in effects) if effects else None
    return {"state": "DEFINED" if floor is not None and not invalid else "UNDEFINED",
            "effect_floor_kg_s": floor, "threshold_kg_s": .1*floor if floor is not None and not invalid else None,
            "numeric_zero_tolerance_kg_s": tolerance, "eligible": effects, "excluded_zero": zero, "invalid": invalid}


def ranking_metrics(queries, predicted_ff, true_ff, properties):
    import numpy as np
    from scipy.stats import kendalltau
    if len(queries) != 4096 or not np.isfinite(predicted_ff).all() or not np.isfinite(true_ff).all():
        return {"state": "INVALID", "pass": False}
    values = []
    for start in range(0,4096,64):
        group = queries[start:start+64]
        if [q["draw_id"] for q in group] != [f"draw_{i:02d}" for i in range(64)]:
            raise ValueError("Ranking paired draws are misordered")
        factor = derived_outputs(group[0], 1.0, properties)["lifecycle_g_s"]
        values.append((group[0]["design_id"], float(np.quantile(predicted_ff[start:start+64]*factor,.95)),
                       float(np.quantile(true_ff[start:start+64]*factor,.95))))
    pred, true = np.asarray([v[1] for v in values]), np.asarray([v[2] for v in values])
    tau = float(kendalltau(pred,true,variant="b").statistic)
    eligible, agreement = 0, 0
    for a in range(64):
        for b in range(a+1,64):
            if abs(true[a]-true[b]) <= 1e-12:
                continue
            eligible += 1; agreement += int(np.sign(pred[a]-pred[b]) == np.sign(true[a]-true[b]))
    pred_order = sorted(range(64),key=lambda i:(pred[i],values[i][0]))
    true_order = sorted(range(64),key=lambda i:(true[i],values[i][0]))
    overlap = len(set(pred_order[:10]) & set(true_order[:10]))
    fraction = agreement/eligible if eligible else None
    return {"state": "SCORED", "pass": math.isfinite(tau) and tau>=.98 and overlap>=9 and fraction is not None and fraction>=.99,
        "kendall_tau_b": tau if math.isfinite(tau) else None, "top10_overlap_count": overlap,
        "paired_ordering_agreement": fraction, "eligible_pairs": eligible,
        "rows": [{"design_id": q[0], "predicted_q95_g_s": q[1], "reference_q95_g_s": q[2]} for q in values]}


def _prediction(output, member, name):
    import numpy as np
    path = output / f"sealed_predictions/{member['arm']}/N{member['N']}/seed{member['seed']}_{name}.npz"
    with np.load(path, allow_pickle=False) as archive:
        query_name="named_product" if name=="named" else name
        queries=read_json(output/f"splits/{query_name}.json")
        if archive["input_sha256"].tolist()!=[q["input_sha256"] for q in queries]:
            raise ValueError("Frozen prediction/input identity mismatch")
        return tuple(archive[key].astype(np.float64) for key in ("ff","T4","Y4"))


def _read_named_auxiliary(output):
    import numpy as np
    with (output/"sealed/named_central_full_state.csv").open(newline="") as stream:rows=list(csv.DictReader(stream))
    with np.load(output/"sealed/named_central_species.npz",allow_pickle=False) as archive:
        return rows,{key:archive[key].copy() for key in archive.files}


def _named_auxiliary(output,data,central,named_queries,predictions,thermo):
    import numpy as np
    rows,archive=data
    Y=archive["Y4"];ids=archive["named_case_id"].tolist();hashes=archive["input_sha256"].tolist()
    if archive["species_order"].tolist()!=thermo.names:raise ValueError("Named species order drift")
    if len(rows)!=68 or len({r["named_case_id"] for r in rows})!=68:
        raise ValueError("Named central auxiliary coverage changed")
    expected=read_json(output/"splits/named_central.json")
    reference={(r["fuel"],r["op"]):r for r in central}
    finite=True;parity=True;true_ff=[];true_T=[];true_Y=[];pred_indices=[]
    indices={q["named_case_id"]:i for i,q in enumerate(named_queries) if q["draw_id"]=="central"}
    for row,query in zip(rows,expected):
        if row["named_case_id"]!=query["named_case_id"] or row["input_sha256"]!=query["input_sha256"]:
            raise ValueError("Named full-state input identity mismatch")
        source=reference[(row["fuel"],row["op"])]
        if row["status"]!="converged" or source["status"]!="converged":finite=False;continue
        parity &= abs(float(row["ff_kg_s"])-float(source["ff"]))<=1e-9 and abs(float(row["T4_K"])-float(source["T4"]))<=1e-6
        j=int(row["species_row_index"])
        if ids[j]!=query["named_case_id"] or hashes[j]!=query["input_sha256"]:raise ValueError("Named species proof mismatch")
        if not query["in_product_API"]:continue
        pred_indices.append(indices[query["named_case_id"]]);true_ff.append(float(row["ff_kg_s"]));true_T.append(float(row["T4_K"]));true_Y.append(Y[j])
    if not finite or len(pred_indices)!=64:
        return {"complete":False,"state":"INVALID_REFERENCE","source_parity_pass":parity,"auxiliary_pass":False}
    try:metrics=prediction_metrics(*(p[pred_indices] for p in predictions),np.asarray(true_ff),np.asarray(true_T),np.asarray(true_Y),thermo)
    except ValueError:metrics={"complete":False,"state":"INVALID_THERMO"}
    metrics.update(source_parity_pass=bool(parity),auxiliary_pass=bool(parity and _auxiliary_pass(metrics)),
        in_API_auxiliary_rows=64,alternative_context_rows=4,alternative_context_prediction="unavailable_outside_four_fraction_API")
    return metrics


def _publish_predictions(output,name,queries,predictions,true_ff,true_T,properties):
    from .run import csv_once
    rows=[]
    for i,q in enumerate(queries):
        ff,T=predictions[0][i],predictions[1][i]
        valid=math.isfinite(ff) and ff>0 and math.isfinite(T) and T>0
        row={key:q[key] for key in ("draw_id","input_sha256","f_JetA","f_HEFA","f_FT","f_ATJ","thrust_fraction")}
        row.update({key:q[key] for key in ("design_id","fuel","op","named_case_id") if key in q})
        row.update(ff_kg_s=float(ff) if math.isfinite(ff) else None,T4_K=float(T) if math.isfinite(T) else None,
            reference_ff_kg_s=float(true_ff[i]) if math.isfinite(true_ff[i]) else None,
            reference_T4_K=float(true_T[i]) if math.isfinite(true_T[i]) else None,
            status="predicted" if valid else "invalid_prediction",diagnostic_unsafe=True)
        if valid:row.update(derived_outputs(q,ff,properties))
        rows.append(row)
    filename="ranking_predictions.csv" if name=="ranking_test" else f"{name}_predictions.csv"
    csv_once(output/filename,rows)


def claim_concordance(claims,queries,predictions,properties):
    """Same unchanged rule for the three requested quantities represented here."""
    import numpy as np
    from .postprocess import lifecycle_scenarios
    lookup={(q["fuel"],q["op"],q["draw_id"]):(q,i) for i,q in enumerate(queries)}
    scenarios=lifecycle_scenarios(properties);rows=[]
    for claim in claims:
        quantity=claim["quantity"]
        if quantity not in ("ff","T4","lifecycle_g_s"):continue
        a,b,op=claim["a"],claim["b"],claim["op"]
        keys=[(fuel,op,draw) for fuel in (a,b) for draw in ("central",*[f"draw_{i:02d}" for i in range(64)])]
        if not all(key in lookup for key in keys):
            rows.append({"a":a,"b":b,"op":op,"quantity":quantity,"state":"outside_product_API","claimed":None});continue
        def value(key):
            q,i=lookup[key]
            if quantity=="ff":return predictions[0][i]
            if quantity=="T4":return predictions[1][i]
            return derived_outputs(q,predictions[0][i],properties)["lifecycle_g_s"]
        delta=float(value((a,op,"central"))-value((b,op,"central")))
        paired=np.asarray([value((a,op,f"draw_{i:02d}"))-value((b,op,f"draw_{i:02d}")) for i in range(64)])
        finite=math.isfinite(delta) and np.isfinite(paired).all()
        sign=float(np.mean(np.sign(paired)==np.sign(delta))) if finite else None
        extra=None
        if quantity=="lifecycle_g_s" and finite:
            qa,ia=lookup[(a,op,"central")];qb,ib=lookup[(b,op,"central")]
            differences=[]
            for scenario in scenarios:
                def factor(q):return sum(q[f"f_{fuel}"]*properties["fuels"][properties["surrogates"][fuel]]["lhv_liquid_MJ_kg"]*scenario[fuel] for fuel in FUELS)
                differences.append(predictions[0][ia]*factor(qa)-predictions[0][ib]*factor(qb))
            extra=float(np.mean(np.sign(differences)==np.sign(delta)))
        # The alternative reference is outside this product API; retain the
        # independently scored unchanged simulator representation spread.
        spread=float(claim["spread_S_dooley2012_vs_2010"])
        domain=claim["in_domain"].lower()=="true"
        claimed=finite and domain and delta!=0 and sign>=.95 and abs(delta)>spread and (extra is None or extra>=.95)
        original=claim["claimed"].lower()=="true"
        rows.append({"a":a,"b":b,"op":op,"quantity":quantity,"state":"SCORED","delta_central":delta if math.isfinite(delta) else None,
            "paired_sign_agreement":sign,"corsia_sign_agreement":extra,"unchanged_reference_spread":spread,
            "claimed":bool(claimed),"reference_claimed":original,"claim_concordant":bool(claimed)==original,
            "central_sign_concordant":finite and np.sign(delta)==np.sign(float(claim["delta_central"]))})
    return rows


def _distribution(values):
    import numpy as np
    values=np.asarray(values,dtype=float).ravel();valid=np.isfinite(values);finite=values[valid]
    return {"rows":int(len(values)),"invalid":int((~valid).sum()),"mean":float(finite.mean()) if len(finite) else None,
        "RMS":float(np.sqrt(np.mean(finite**2))) if len(finite) and np.isfinite(finite**2).all() else None,
        "min":float(finite.min()) if len(finite) else None,"max":float(finite.max()) if len(finite) else None,
        "q025_q50_q975":np.quantile(finite,[.025,.5,.975]).tolist() if len(finite) else None}


def physics_diagnostics(output,validation,N,datasets,queries,named_q,named_aux_data,central,drawn,thermo,public,draws,run,backend=None):
    import numpy as np
    from .train import monotonic_endpoints
    data=dict(datasets);qsets=dict(queries)
    for name in ("train","validation"):
        data[name]=read_dataset(output,name);qsets[name]=read_json(output/f"splits/{name}.json")
    result={"sets":{},"monotonic":{},"teacher_anchor_monotonic":[],"scope":"paired arms at selected Mphys N"}
    named_rows,archive=named_aux_data
    full_named=read_json(output/"splits/named_central.json")
    named_ff=np.full(68,np.nan);named_T=np.full(68,np.nan);named_Y=np.full((68,492),np.nan)
    for i,row in enumerate(named_rows):
        if row["status"]=="converged":
            named_ff[i]=float(row["ff_kg_s"]);named_T[i]=float(row["T4_K"]);named_Y[i]=archive["Y4"][int(row["species_row_index"])]
    data["named_central"]=(named_rows,named_ff,named_T,named_Y,None);qsets["named_central"]=full_named
    def residuals(ff,T,Y,states):
        result={key:np.full(len(ff),np.nan) for key in ("rE","rE_LHV")} | {"elements":np.full((len(ff),6),np.nan)}
        valid=np.isfinite(ff)&(ff>0)&np.isfinite(T)&np.isfinite(Y).all(axis=1)
        valid &= (T>=thermo.bounds[:,0].max())&(T<=thermo.bounds[:,2].min())
        if valid.any():
            selected_states={key:value[valid] for key,value in states.items()}
            values=thermo.energy_diagnostics(ff[valid],T[valid],Y[valid],selected_states)
            for key,value in values.items():result[key][valid]=value
        return result
    for name,dataset in data.items():
        run.assert_current();case_queries=qsets[name];states=thermo.input_states(case_queries,public,draws)
        teacher=residuals(*dataset[1:4],states)
        entry={"teacher":{key:_distribution(value) for key,value in teacher.items()},"models":{}}
        if name=="named_central":
            keep=[i for i,q in enumerate(case_queries) if q["in_product_API"]]
            model_queries=[case_queries[i] for i in keep]
            source_indices={q["named_case_id"]:i for i,q in enumerate(named_q) if q["draw_id"]=="central"}
            prediction_indices=[source_indices[q["named_case_id"]] for q in model_queries]
            model_states={key:value[keep] for key,value in states.items()}
            matching_teacher={key:value[keep] for key,value in teacher.items()}
            entry["outside_product_API_reference_rows"]=4
        else:model_queries=case_queries;model_states=states;matching_teacher=teacher
        for arm in ("Mdata","Mphys"):
            members=[m for m in validation if m["arm"]==arm and m["N"]==N]
            predictions=[];arm_rows={}
            for member in members:
                if name in ("test","ranking_test"):
                    pred=_prediction(output,member,name)
                elif name=="named_central":pred=tuple(v[prediction_indices] for v in _prediction(output,member,"named"))
                else:
                    scaler=member["feature_scaler"];features=feature_rows(model_queries,draws,public)
                    pred=forward_cpu64(_params(output,member),(features-np.asarray(scaler["mean"]))/np.asarray(scaler["scale"]),backend)
                predictions.append(pred);values=residuals(*pred,model_states)
                arm_rows[str(member["seed"])]= {key:_distribution(value) for key,value in values.items()} | {
                    "signed_model_minus_teacher":{key:_distribution(values[key]-matching_teacher[key]) for key in ("rE","rE_LHV")}}
            ensemble=ensemble_prediction(predictions)
            values=residuals(*ensemble,model_states)
            arm_rows["ensemble"]={key:_distribution(value) for key,value in values.items()} | {
                "signed_model_minus_teacher":{key:_distribution(values[key]-matching_teacher[key]) for key in ("rE","rE_LHV")},
                "seed_RMS_sd_ddof1":float(np.std([arm_rows[str(seed)]["rE"]["RMS"] for seed in (42,43,44)],ddof=1))
                    if all(arm_rows[str(seed)]["rE"]["RMS"] is not None for seed in (42,43,44)) else None}
            entry["models"][arm]=arm_rows
        result["sets"][name]=entry
    physics=read_json(output/"splits/physics.json");minus,plus,width=monotonic_endpoints(physics);width=np.asarray(width)
    for arm in ("Mdata","Mphys"):
        members=[m for m in validation if m["arm"]==arm and m["N"]==N];slopes=[];records={}
        for member in members:
            run.assert_current();mean=np.asarray(member["feature_scaler"]["mean"]);scale=np.asarray(member["feature_scaler"]["scale"])
            low=forward_cpu64(_params(output,member),(feature_rows(minus,draws,public)-mean)/scale,backend)[0]
            high=forward_cpu64(_params(output,member),(feature_rows(plus,draws,public)-mean)/scale,backend)[0]
            slope=(high-low)/width;slopes.append(slope)
            records[str(member["seed"])]= {"negative_slope_count":int((slope<0).sum()),"negative_slope_penalty":_distribution(np.maximum(-slope,0))}
        mean=np.mean(slopes,axis=0);records["ensemble"]={"negative_slope_count":int((mean<0).sum()),"negative_slope_penalty":_distribution(np.maximum(-mean,0))}
        result["monotonic"][arm]=records
    lookup={(r["fuel"],r["op"],r.get("draw","central")):r for r in central+drawn}
    order=("IDLE","APPROACH","CLIMB85","TAKE-OFF");xs=(.07,.30,.85,1.0)
    for fuel in sorted({q["fuel"] for q in named_q}):
        for draw in ("central",*[f"draw_{i:02d}" for i in range(64)]):
            rows=[lookup[(fuel,op,draw)] for op in order]
            valid=all(r["status"]=="converged" and math.isfinite(float(r["ff"])) for r in rows)
            result["teacher_anchor_monotonic"].append({"fuel":fuel,"draw_id":draw,"valid":valid,
                "slopes_kg_s_per_thrust_fraction":[(float(rows[i+1]["ff"])-float(rows[i]["ff"]))/ (xs[i+1]-xs[i]) for i in range(3)] if valid else None})
    return result


def _scalar_metrics(pred, true):
    import numpy as np
    ff,T4,true_ff,true_T4=[np.asarray(value,dtype=np.float64) for value in (*pred[:2],*true[:2])]
    valid=np.isfinite(ff)&np.isfinite(T4)&np.isfinite(true_ff)&np.isfinite(true_T4)&(ff>0)&(true_ff>0)&(T4>0)&(true_T4>0)
    summary={"complete":bool(valid.all()),"state":"SCORED" if valid.all() else "INVALID",
             "requested_rows":len(valid),"conditional_valid_rows":int(valid.sum()),"invalid_rows":int((~valid).sum())}
    if not valid.any():
        return summary
    a,b = np.abs(ff[valid]-true_ff[valid]), np.abs(T4[valid]-true_T4[valid])
    return {**summary, "ff_MAE_kg_s":float(a.mean()), "ff_max_kg_s":float(a.max()),
            "T4_MAE_K":float(b.mean()), "T4_max_K":float(b.max())}


def _fidelity(metrics, floor):
    threshold = floor["threshold_kg_s"]
    return metrics.get("complete",False) and threshold is not None and metrics["ff_MAE_kg_s"]<=threshold and metrics["ff_max_kg_s"]<=threshold and metrics["T4_MAE_K"]<=.5 and metrics["T4_max_K"]<=2


def _auxiliary_pass(metrics):
    return metrics.get("complete",False) and all(metrics[key]<=limit for key,limit in
        (("composition_mean_L1",.001),("composition_max_L1",.005),("cp_relative_max",.005),
         ("R_relative_max",.001),("gamma_absolute_max",.002)))


def score_all(root, output, reg, reg_sha, context, run, *, backend=None):
    import numpy as np
    from .run import csv_once
    output, root = Path(output), Path(root)
    from simulation.ml_backend import resolve_backend
    backend = resolve_backend(backend)
    run.assert_current()
    frozen = read_json(output / "predictions_freeze.json")
    for key in ("predictions","weights","inputs"):
        verify_artifacts(output,frozen[key])
    p73_paths = reg["sole_score"]["named_paths"]
    targets = {str((output/path).relative_to(root)):sha256_file(output/path) for path in
        ("sealed/test_teacher_rows.csv","sealed/test_teacher_species.npz","sealed/ranking_teacher_rows.csv","sealed/ranking_teacher_species.npz",
         "sealed/named_central_full_state.csv","sealed/named_central_species.npz")}
    targets.update({path:sha256_file(root/path) for path in p73_paths})
    reservation = {"schema_version":1,"registration_sha256":reg_sha,"identity":context.identity,
        "binary_sha256":context.binary_sha256,"predictions_freeze_sha256":sha256_file(output/"predictions_freeze.json"),
        "selection_sha256":sha256_file(output/"selection.json"),"targets":targets}
    write_once(output/"score_reservation.json",reservation)
    # Numeric targets are opened only below this sole write-once reservation.
    central_path, draws_path = root/p73_paths[0],root/p73_paths[1]
    with central_path.open(newline="") as stream: central = list(csv.DictReader(stream))
    with draws_path.open(newline="") as stream: drawn = list(csv.DictReader(stream))
    floor = effect_floor(central)
    draw_floors = []
    for draw in [f"draw_{i:02d}" for i in range(64)]:
        draw_floors.append({"draw_id":draw,**effect_floor([r for r in drawn if r["draw"]==draw])})
    public,draws,properties = (read_json(output/name) for name in ("public_inputs.json","fixed_draws.json","frozen_properties.json"))
    thermo = Thermo(properties)
    named_aux_data=_read_named_auxiliary(output)
    named_q = read_json(output/"splits/named_product.json")
    lookup = {(r["fuel"],r["op"],r.get("draw","central")):r for r in central+drawn}
    expected_keys={(q["fuel"],q["op"],q["draw_id"]) for q in named_q}
    if not expected_keys.issubset(lookup):
        raise ValueError("Named paired input keys incomplete")
    named_true=[np.asarray([float(lookup[(q["fuel"],q["op"],q["draw_id"])][key])
              if lookup[(q["fuel"],q["op"],q["draw_id"])]["status"]=="converged" else np.nan for q in named_q]) for key in ("ff","T4")]
    validation=read_json(output/"validation.json")["members"]
    selection=read_json(output/"selection.json")["selection"]
    datasets={name:read_dataset(output,name,score_reservation=reservation) for name in ("test","ranking_test")}
    queries={name:read_json(output/f"splits/{name}.json") for name in datasets}
    candidate_rows=[]; metrics_by_member={}; selected={}
    for member in validation:
        run.assert_current()
        entry={}
        for name,dataset in datasets.items():
            pred=_prediction(output,member,name)
            try: metrics=prediction_metrics(*pred,*dataset[1:4],thermo)
            except ValueError: metrics={"complete":False,"state":"INVALID_THERMO"}
            metrics["fidelity_pass"]=_fidelity(metrics,floor)
            metrics["auxiliary_pass"]=_auxiliary_pass(metrics)
            entry[name]=metrics
            candidate_rows.append({"arm":member["arm"],"N":member["N"],"seed":member["seed"],"dataset":name,**metrics})
        metrics=_scalar_metrics(_prediction(output,member,"named"),named_true)
        auxiliary=_named_auxiliary(output,named_aux_data,central,named_q,_prediction(output,member,"named"),thermo)
        metrics["fidelity_pass"]=_fidelity(metrics,floor);metrics["auxiliary_pass"]=auxiliary["auxiliary_pass"];entry["named"]=metrics
        candidate_rows.append({"arm":member["arm"],"N":member["N"],"seed":member["seed"],"dataset":"named",**metrics})
        metrics_by_member[(member["arm"],member["N"],member["seed"])]=entry
    N=selection["Mphys"]["N"]
    for arm in ("Mphys","Mdata"):
        members=[m for m in validation if m["arm"]==arm and m["N"]==N]
        selected[arm]={name:ensemble_prediction([_prediction(output,m,name) for m in members])
                       for name in ("test","ranking_test","named")}
    ensemble_metrics={}
    for name,dataset in datasets.items():
        try: metrics=prediction_metrics(*selected["Mphys"][name],*dataset[1:4],thermo)
        except ValueError: metrics={"complete":False,"state":"INVALID_THERMO"}
        metrics["fidelity_pass"]=_fidelity(metrics,floor); metrics["auxiliary_pass"]=_auxiliary_pass(metrics)
        ensemble_metrics[name]=metrics
    ensemble_metrics["named"]=_scalar_metrics(selected["Mphys"]["named"],named_true)
    ensemble_metrics["named"]["fidelity_pass"]=_fidelity(ensemble_metrics["named"],floor)
    named_auxiliary=_named_auxiliary(output,named_aux_data,central,named_q,selected["Mphys"]["named"],thermo)
    ensemble_metrics["named"]["auxiliary_pass"]=named_auxiliary["auxiliary_pass"]
    ranking=ranking_metrics(queries["ranking_test"],selected["Mphys"]["ranking_test"][0],datasets["ranking_test"][1],properties)
    all_seeds=all(metrics_by_member[("Mphys",N,seed)][name]["fidelity_pass"] and
                  metrics_by_member[("Mphys",N,seed)][name]["auxiliary_pass"]
                  for seed in (42,43,44) for name in ("test","ranking_test","named"))
    fidelity=selection["Mphys"]["validation_pass"] and all_seeds and all(row["fidelity_pass"] and row["auxiliary_pass"] for row in ensemble_metrics.values())
    physics=physics_diagnostics(output,validation,N,datasets,queries,named_q,named_aux_data,central,drawn,thermo,public,draws,run,backend)
    baseline=physics["sets"]["test"]["models"]["Mdata"]["ensemble"]["rE"]["RMS"]
    physical=physics["sets"]["test"]["models"]["Mphys"]["ensemble"]["rE"]["RMS"]
    comparator_pass=all(metrics_by_member[("Mdata",N,seed)][name]["fidelity_pass"] and metrics_by_member[("Mdata",N,seed)][name]["auxiliary_pass"]
        for seed in (42,43,44) for name in ("test","ranking_test","named"))
    comparator_pass &= all(m["validation_pass"] for m in validation if m["arm"]=="Mdata" and m["N"]==N)
    comparator_ensemble={}
    for name,dataset in datasets.items():
        try:metric=prediction_metrics(*selected["Mdata"][name],*dataset[1:4],thermo)
        except ValueError:metric={"complete":False,"state":"INVALID_THERMO"}
        metric["fidelity_pass"]=_fidelity(metric,floor);metric["auxiliary_pass"]=_auxiliary_pass(metric)
        comparator_ensemble[name]=metric
    metric=_scalar_metrics(selected["Mdata"]["named"],named_true)
    metric["fidelity_pass"]=_fidelity(metric,floor)
    metric["auxiliary_pass"]=_named_auxiliary(output,named_aux_data,central,named_q,selected["Mdata"]["named"],thermo)["auxiliary_pass"]
    comparator_ensemble["named"]=metric
    comparator_pass &= all(metric["fidelity_pass"] and metric["auxiliary_pass"] for metric in comparator_ensemble.values())
    physics["comparator_ensemble_metrics"]=comparator_ensemble
    physics["comparator_fidelity_pass"]=bool(comparator_pass)
    physics["benefit_pass"]=bool(fidelity and comparator_pass and baseline is not None and math.isfinite(baseline) and baseline>0
        and physical is not None and math.isfinite(physical) and physical<=.8*baseline)
    # CPU64 scoring preserves native Torch64 weights and promotes optional
    # MLX32 exports; the source/export diagnostic records the actual precision.
    precision=read_json(output/"precision.json")
    write_once(output/"test_metrics.json",{"ensemble":ensemble_metrics,"physics":physics,"fidelity_pass":fidelity})
    write_once(output/"ranking_metrics.json",ranking)
    with (root/p73_paths[2]).open(newline="") as stream:claims=list(csv.DictReader(stream))
    write_once(output/"named_metrics.json",{"effect_floor":floor,"draw_floors":draw_floors,
        "ensemble":ensemble_metrics["named"],"central_auxiliary":named_auxiliary,
        "claim_concordance":claim_concordance(claims,named_q,selected["Mphys"]["named"],properties),
        "auxiliary_draw_status":"unavailable_no_fullY_in_A1"})
    for name,data in datasets.items():_publish_predictions(output,name,queries[name],selected["Mphys"][name],data[1],data[2],properties)
    _publish_predictions(output,"named",named_q,selected["Mphys"]["named"],named_true[0],named_true[1],properties)
    csv_once(output/"all_candidate_diagnostic_metrics.csv",candidate_rows)
    csv_once(output/"per_seed_metrics.csv",[r for r in candidate_rows if r["arm"]=="Mphys" and r["N"]==N])
    run.assert_current()
    return {"fidelity_pass":bool(fidelity),"ranking_pass":bool(ranking["pass"]),"precision_pass":precision["state"]=="PASS",
        "physics_benefit_pass":physics["benefit_pass"],"effect_threshold_kg_s":floor["threshold_kg_s"]}


def deployment_receipt(output,reg,reg_sha,scored,timing):
    gates={"fidelity":"PASS" if scored["fidelity_pass"] else "FAIL",
        "ranking":"PASS" if scored["ranking_pass"] else "FAIL",
        "precision":"PASS" if scored["precision_pass"] else "FAIL", "provenance":"PASS",
        "operational":"PASS" if timing["operational_pass"] else "FAIL"}
    write_once(output/"deployment_receipt.json",{"registration_id":reg["id"],"registration_sha256":reg_sha,
        "state":"PASS" if all(v=="PASS" for v in gates.values()) else "FAIL", "gates":gates,
        "product_sha256":sha256_file(output/"product.json"),
        "evidence_sha256":artifact_hashes(output,["test_metrics.json","ranking_metrics.json","named_metrics.json",
                                                  "precision.json","timing.json","mlx_gpu_timing.json","report.json"])})
