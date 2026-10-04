"""Registered source diagnostics and honest CPU64/all-core/GPU measurements."""
from __future__ import annotations

import math
import os
import statistics
import time
from pathlib import Path

from .models import forward64, load_product, make_model, output_map, summarize_draws
from .registration import read_json, sha256_file, write_once
from .thermo import Thermo,backend_array


def break_even(setup_seconds, cpp_per_query, product_per_query):
    difference=cpp_per_query-product_per_query
    if not all(math.isfinite(v) and v>=0 for v in (setup_seconds,cpp_per_query,product_per_query)) or difference<=0:
        return None
    return math.ceil(setup_seconds/difference)


def source_diagnostics(root,output,reg,context,run):
    """No HP equilibrium or project label reads; fixed prospective checks."""
    import cantera as ct
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from .inputs import canonical_query,state_for
    from .teacher import load_selected_core
    run.assert_current()
    started=time.perf_counter()
    properties,public,draws=(read_json(output/name) for name in ("frozen_properties.json","public_inputs.json","fixed_draws.json"))
    thermo=Thermo(properties)
    gas=ct.Solution(str(root/"data/creck_c1c16_full.yaml"))
    maxima={key:0.0 for key in ("h","cp","s")}; passed=True;cases=0
    for i,bounds in enumerate(thermo.bounds):
        for temperature in (300,600,1000,1500,2000,bounds[1]-.001,bounds[1]+.001):
            run.assert_current() if cases%128==0 else None
            gas.TP=float(temperature),101325
            refs={"h":gas.standard_enthalpies_RT[i]*ct.gas_constant*temperature/gas.molecular_weights[i],
                  "cp":gas.standard_cp_R[i]*ct.gas_constant/gas.molecular_weights[i],
                  "s":gas.standard_entropies_R[i]*ct.gas_constant/gas.molecular_weights[i]}
            for field in maxima:
                value=float(thermo._species(float(temperature),field,species_index=i)[0]);error=abs(value-refs[field])
                bound=(1e-5 if field=="h" else 1e-8)+1e-10*abs(refs[field])
                maxima[field]=max(maxima[field],error);passed &= error<=bound
            cases+=1
    if cases!=3444:
        raise RuntimeError("Fixed source-only thermochemistry case budget changed")
    core=load_selected_core(context.binary_path,context.binary_sha256)
    engine=core.V6Engine(str(root/"data/creck_c1c16_full.yaml"));T3error=0.0;compressor_calls=0
    for draw in ("central","draw_00","draw_63"):
        for x in (.07,.15,.30,.55,.70,.85,.925,1):
            run.assert_current()
            query={"f_JetA":1,"f_HEFA":0,"f_FT":0,"f_ATJ":0,"thrust_fraction":x,"draw_id":draw}
            state=state_for(query,public,draws);config=engine.config
            config.pi_c,config.eta_c=state["pi_c"],state["eta_c"];engine.config=config
            actual=engine.run_compressor(288.15,101325)
            T3error=max(T3error,abs(float(actual["T_out"])-thermo.compressor_temperature(state["pi_c"],state["eta_c"])))
            compressor_calls+=1
    passed &= T3error<=1e-6
    old=mx.default_device();mx.set_default_device(mx.cpu)
    try:
        # Fixed synthetic physical output parameters; these do not fit labels.
        query=canonical_query({"f_JetA":.7,"f_HEFA":.1,"f_FT":.1,"f_ATJ":.1,"thrust_fraction":.55,"draw_id":"draw_00"},draws,public)
        states=thermo.input_states([query],public,draws)
        indices=[thermo.names.index("CO2"),thermo.names.index("H2O")]
        logits=np.full(492,-12.0);logits[thermo.names.index("N2")]=0;logits[indices]=[-1,-2]
        theta=np.asarray([math.log(.6),math.log(1.5),.2,-.1],dtype=np.float32)
        def physical_loss(value,xp,control=None):
            active=value if control!="constant_output" else backend_array(theta,xp)
            z=backend_array(logits,xp)
            mask=backend_array(np.eye(492)[indices],xp)
            z=z+xp.sum(active[2:,None]*mask,axis=0)
            z=z-xp.max(z);Y=xp.exp(z);Y=Y/xp.sum(Y)
            ff=xp.exp(active[0]);T4=1000*xp.exp(active[1])
            energy,element=thermo.residuals(ff[None],T4[None],Y[None,:],states,xp)
            if control=="wrong_sign":
                href=backend_array(thermo.species_h(298.15),xp);air=backend_array(properties["burner_air_Y"],xp)
                ma=backend_array(states["ma"],xp);eta=backend_array(states["eta_b"],xp);fuel=backend_array(states["fuel_Y"],xp)
                Qchem=ma*xp.sum(air*href)+ff*xp.sum(fuel*href,axis=-1)-(ma+ff)*xp.sum(Y*href)
                energy=energy+2*eta*Qchem/(ma*1e6)
            if control=="wrong_units":energy=energy*1000
            if control=="wrong_reference":energy=energy+xp.sum(Y*backend_array(thermo.species_h(298.15),xp))/1e6
            return xp.sum(energy**2)+xp.sum(element**2)
        analytic=np.asarray(mx.grad(lambda t:physical_loss(t,mx))(mx.array(theta)),dtype=np.float64)
        derivative_errors={};controls={}
        for h in (.001,.01):
            finite=[]
            for i in range(4):
                delta=np.zeros(4);delta[i]=h
                finite.append((float(physical_loss(theta.astype(float)+delta,np))-float(physical_loss(theta.astype(float)-delta,np)))/(2*h))
            finite=np.asarray(finite)
            denom=max(1e-5,float(np.max(np.abs(analytic))),float(np.max(np.abs(finite))))
            error=float(np.max(np.abs(analytic-finite))/denom);derivative_errors[str(h)]=error;passed &= error<=.02
        for control in ("wrong_sign","wrong_units","wrong_reference","constant_output"):
            wrong=np.asarray(mx.grad(lambda t:physical_loss(t,mx,control))(mx.array(theta)),dtype=np.float64)
            error=float(np.max(np.abs(wrong-analytic))/max(1e-5,float(np.max(np.abs(wrong))),float(np.max(np.abs(analytic)))))
            controls[control]=error;passed &= error>.02
        model=make_model(42);features=np.random.default_rng(39001).standard_normal((33,12)).astype(np.float32)
        actual=[np.asarray(v,dtype=np.float64) for v in output_map(model(mx.array(features)),mx)]
        params={key:np.asarray(v,dtype=np.float32).astype(float) for key,v in tree_flatten(model.parameters())}
        expected=forward64(params,features.astype(float))
        relative=[float(np.max(np.abs(a-b))/max(1e-30,float(np.max(np.abs(b))))) for a,b in zip(actual[:2],expected[:2])]
        species_error=float(np.max(np.abs(actual[2]-expected[2]).sum(axis=1)))
        passed &= max(relative)<=5e-5 and species_error<=1e-4
    finally:
        mx.set_default_device(old)
    run.assert_current()
    receipt={"state":"PASS" if passed else "FAIL","thermo_cases":cases,"compressor_calls":compressor_calls,
        "max_abs_thermo_errors":maxima,"T3_max_abs_K":T3error,"gradient_normalized_errors":derivative_errors,
        "negative_control_disagreements":controls,"export_ff_T4_max_normalized":relative,
        "export_species_L1_max":species_error,"setup_seconds":time.perf_counter()-started,
        "binary_sha256":context.binary_sha256,"physical_energy_floor_is_not_zero_guarantee":True}
    write_once(output/"precision.json",receipt)
    if not passed:
        raise RuntimeError("Registered source/graph/precision checks failed before label acquisition")


def _timed(function,queries):
    started=time.perf_counter_ns();predictions=function(queries)
    # Product API and published timing use the identical public aggregation.
    summarize_draws(queries,predictions)
    return (time.perf_counter_ns()-started)/1e9


def _gpu_measure(product,queries,models,run):
    import numpy as np
    import mlx.core as mx
    from .inputs import feature_rows,canonical_query
    from .postprocess import derived_outputs
    from .models import ensemble_outputs
    chunks=[];gpu_seconds=transfer_seconds=post_seconds=0.0
    started=time.perf_counter_ns()
    for start in range(0,len(queries),4096):
        run.assert_current()
        batch=product.canonical_queries(queries[start:start+4096]);X=feature_rows(batch,product.draws,product.public)
        members=[]
        for model,(mean,scale) in zip(models,product.scalers):
            transfer_start=time.perf_counter_ns();x=mx.array(((X-mean)/scale).astype(np.float32));mx.eval(x)
            transfer_seconds+=(time.perf_counter_ns()-transfer_start)/1e9
            gpu_start=time.perf_counter_ns();values=output_map(model(x),mx);mx.eval(*values);mx.synchronize()
            gpu_seconds+=(time.perf_counter_ns()-gpu_start)/1e9
            transfer_start=time.perf_counter_ns();members.append([np.asarray(value,dtype=float) for value in values])
            transfer_seconds+=(time.perf_counter_ns()-transfer_start)/1e9
        post_start=time.perf_counter_ns()
        arrays=ensemble_outputs(members)
        rows=product.postprocess(batch,arrays)
        chunks.extend(summarize_draws(batch,rows));post_seconds+=(time.perf_counter_ns()-post_start)/1e9
    chunks.sort(key=lambda row:(row.get("ranking_q95_lifecycle_g_s") if row.get("ranking_q95_lifecycle_g_s") is not None else math.inf,row["candidate_id"]))
    return {"total_seconds":(time.perf_counter_ns()-started)/1e9,"gpu_seconds":gpu_seconds,
        "transfers_seconds":transfer_seconds,"cpu64_exact_postprocess_and_aggregation_seconds":post_seconds}


def gpu_measurements(output,product,study,run):
    from scripts.phase8.scientific_workflow_gate import GateError
    started=time.perf_counter();record={"mandatory_attempt":True,"available":False,"state":"UNAVAILABLE"}
    try:
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal GPU is unavailable")
        old=mx.default_device();mx.set_default_device(mx.gpu)
        try:
            setup_started=time.perf_counter();models=[]
            for member in product.bundle["members"]:
                model=make_model(member["seed"]);model.load_weights(str(product.output/member["weights_path"].replace(".npz",".safetensors")))
                mx.eval(model.parameters());models.append(model)
            record["cold_model_setup_seconds"]=time.perf_counter()-setup_started
            rows=[]
            for count in (1,64,4096,640000):
                run.assert_current();queries=study[:count]
                cold=_gpu_measure(product,queries,models,run)
                if count==640000:
                    timed=[cold];warm=None
                else:
                    warm=_gpu_measure(product,queries,models,run);timed=[_gpu_measure(product,queries,models,run) for _ in range(7)]
                rows.append({"rows":count,"cold":cold,"warmup":warm,"timed":timed})
            record.update(available=True,state="COMPLETE",batches=rows,device="gpu",precision="float32")
        finally:mx.set_default_device(old)
    except GateError:
        raise
    except (ImportError,RuntimeError,ValueError) as error:
        record.update(reason=str(error),exception_type=type(error).__name__)
    run.assert_current()
    record["attempt_seconds"]=time.perf_counter()-started
    write_once(output/"mlx_gpu_timing.json",record)
    return record


def measure(root,output,reg,reg_sha,context,run):
    from .run import command_spec
    from .teacher import ParallelTeacher
    output=Path(output);run.assert_current()
    load_started=time.perf_counter();product=load_product(output/"product.json",require_deployment=False)
    product_load_seconds=time.perf_counter()-load_started
    study=read_json(output/"splits/study.json")
    workers=read_json(output/"environment.json")["physical_cores"]
    spec=command_spec(root,output,context,reg_sha,"timing",[os.fsdecode(os.fsencode(__import__('sys').executable)),"P8-S","timing"])
    write_once(root/spec["owner_lease"]["snapshot_path"],(root/spec["owner_lease"]["path"]).read_bytes())
    write_once(root/spec["command_spec_path"],spec)
    teacher=ParallelTeacher(spec,product.properties,product.public,product.draws,workers,run=run)
    batches=[];requests=0;started=time.perf_counter()
    try:
        for count in (1,64,4096):
            run.assert_current();queries=study[:count]
            cpp_cold=_timed(teacher,queries);product_cold=_timed(product.predict,queries);requests+=count
            cpp_warm=_timed(teacher,queries);product_warm=_timed(product.predict,queries);requests+=count
            cpp=[];surrogate=[]
            for _ in range(7):
                run.assert_current();cpp.append(_timed(teacher,queries));requests+=count
                surrogate.append(_timed(product.predict,queries))
            batches.append({"rows":count,"cpp_cold_seconds":cpp_cold,"product_cold_seconds":product_cold,
                "cpp_warmup_seconds":cpp_warm,"product_warmup_seconds":product_warm,
                "cpp_seconds":cpp,"product_seconds":surrogate,"cpp_median_seconds":statistics.median(cpp),
                "product_median_seconds":statistics.median(surrogate),"speedup":statistics.median(cpp)/statistics.median(surrogate),
                "cpp_min_max_seconds":[min(cpp),max(cpp)],"product_min_max_seconds":[min(surrogate),max(surrogate)]})
    finally:teacher.close()
    if requests!=37449:raise RuntimeError("Registered timing request budget changed")
    gpu=gpu_measurements(output,product,study,run)
    result={"state":"COMPLETE","cpu64_end_to_end":True,"all_physical_cpp_workers":workers,
        "teacher_full_cycle_requests":requests,"batches":batches,"product_cold_model_setup_seconds":product_load_seconds,"measurement_seconds":time.perf_counter()-started,
        "cpu64_bulk_speedup":batches[-1]["speedup"],"gpu_available":gpu["available"],
        "operational_pass":False,"break_even_queries":None,
        "break_even_pending_total_setup_and_selected_study_audit":True}
    write_once(output/"timing_measurements.json",result)
    run.assert_current()
    return result


def finalize_timing(root,output,pipeline_started,timing,study,scored):
    from datetime import datetime
    prerequisite=read_json(output/"prerequisite.json")
    exit_record=read_json(Path(prerequisite["output"])/"command.exit.json")
    # Whole prospective A1 acquisition/provenance command is charged once.
    first,last=exit_record["started"],exit_record["finished"]
    prerequisite_seconds=(datetime.fromisoformat(last)-datetime.fromisoformat(first)).total_seconds()
    if prerequisite_seconds<0:raise ValueError("Prerequisite timing evidence is invalid")
    setup=time.perf_counter()-pipeline_started+prerequisite_seconds
    bulk=timing["batches"][-1];cpp=bulk["cpp_median_seconds"]/4096;product=bulk["product_median_seconds"]/4096
    crossover=break_even(setup,cpp,product)
    timing=dict(timing,total_setup_seconds=setup,named_prerequisite_seconds=prerequisite_seconds,
        named_prerequisite_cost_scope="whole registered acquisition/parity/provenance command, charged once",
        break_even_queries=crossover,break_even_pending_total_setup_and_selected_study_audit=False,
        projected_simulator_640k_seconds=640000*cpp,measured_product_640k_seconds=study["cpu64_total_seconds"],
        operational_pass=bool(scored["fidelity_pass"] and scored["ranking_pass"] and scored["precision_pass"]
            and timing["gpu_available"] and timing["cpu64_bulk_speedup"]>1 and crossover is not None and crossover<=640000
            and study["invalid_prediction_rows"]==0 and study["selected_reference_converged"]==640))
    write_once(output/"timing.json",timing)
    return timing
