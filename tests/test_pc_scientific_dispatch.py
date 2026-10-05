"""Actual PC stage dispatch with manufactured records and mocked computations."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.phase8 import python_pc as pc


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Run:
    def __init__(self, root, out, identity):
        self.out, self.identity, self._released = out, identity, False
        out.mkdir(parents=True)
        self.lease_path = root / 'outputs/stage_owner.json'
        self.lease_path.write_text('{}')
        self.children = []
    def assert_current(self):
        assert not self._released
    def record_children(self, children):
        self.children = children
    def release(self, terminal):
        self._released = True
        terminal = dict(terminal, identity=self.identity)
        dump(self.out/'terminal.json', terminal)
        return terminal


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    w = pc.ScientificWorkflow.__new__(pc.ScientificWorkflow)
    w.root, w.metadata = tmp_path, tmp_path/'outputs/pc'
    w.metadata.mkdir(parents=True)
    w.workers, w.backend, w.device = 10, 'mlx', 'cpu'
    w.sources, w.parity = {}, {'status':'PASS', 'rows':20}
    dump(w.metadata/'scientific_sources.json',{})
    w.saf, w.p73, w.nozzle = [tmp_path/'outputs'/name for name in ('saf','p73','nozzle')]
    w.reg = {'saf':{'id':'P8-S-20261004', 'artifact_root':'outputs/saf'}, 'p73':{}, 'nozzle':{}}
    identity = {'schema':'pc-python-v6-v1', 'registration_sha256':'registered',
                'simulator_identity_sha256':'python-source', 'source_hashes':{},
                'simulator':{'name':'python-v6'}}
    ctx = SimpleNamespace(root=tmp_path, identity=identity, simulator_backend='python',
        simulator_identity=identity['simulator'], simulator_identity_sha256='python-source', workers=10,
        binary_path=None, binary_sha256=None, pc_lease_path=w.metadata/'stage_owner.json')
    ctx.assert_current=lambda:identity
    acquired = []
    def acquire(output, reg_sha, **kwargs):
        assert reg_sha=='registered'
        acquired.append(output)
        return Run(tmp_path, Path(output), identity)
    ctx.acquire_run=acquire
    monkeypatch.setattr(w, 'context', lambda registration, **kwargs:ctx)
    def owned(letter, registration):
        return ctx, acquire(w.metadata/'runs'/letter, 'registered')
    monkeypatch.setattr(w, 'owned', owned)
    w._context, w._acquired = ctx, acquired
    return w


def test_b_uses_exact20_python_rows_and_closes_model(workflow, monkeypatch):
    w=workflow
    rows=pd.DataFrame({'identity':range(25), 'CO (g/kg)':1., 'HC (g/kg)':2., 'NOx (g/kg)':3.})
    seen=[]
    model=SimpleNamespace(predict=lambda params, frame:(seen.append(('predict',params,len(frame))) or
        pd.DataFrame({'predicted':np.arange(len(frame))},index=frame.index)), close=lambda:seen.append(('close',)))
    v5=ModuleType('lto_v5');v5.load_split=lambda:{'heldout_models':[]};v5.calibration_rows=lambda split:rows
    v6=ModuleType('lto_v6');v6.load_registration_v6=lambda:{}
    backend=ModuleType('v6_backend')
    backend.make_model_v6=lambda name, workers:(seen.append(('model',name,workers)) or model)
    parity=ModuleType('g0_parity')
    def compare(frozen, actual):
        actual=pd.read_csv(actual)
        assert len(actual)==20 and list(actual.columns)==['identity','predicted']
        assert len(pd.read_csv(frozen))==20
        return {'match':True}
    parity.compare=compare
    for name,module in [('lto_v5',v5),('lto_v6',v6),('v6_backend',backend),('g0_parity',parity)]:
        monkeypatch.setitem(sys.modules,name,module)
    dump(w.root/'outputs/phase7/calibration_v6.json',{'params':{'toy':1}})
    (w.root/'outputs/phase7/calibration_v6_rows.csv').write_text(rows.to_csv(index=False))
    result=w.b()
    assert result['scientific_verdict']=='PASS' and w.parity['rows']==20
    assert seen==[('model','python',10),('predict',{'toy':1},20),('close',)]


def test_c_dispatches_python_context_without_cpp_parity(workflow, monkeypatch):
    from scripts.phase8 import p73_a1_cpp as p73
    w=workflow;seen=[]
    contract={key:f'{key}.json' for key in ('fit_path','profile_path','gate_path','historical_closed_gate_path')}
    for path in contract.values():dump(w.root/path,{})
    w.reg['p73']={'implementation_contract':contract}
    monkeypatch.setattr(p73,'validate_contract',lambda root,reg:seen.append('contract'))
    monkeypatch.setattr(p73,'validate_exception',lambda profile,fit,hold,historical:seen.append('exception'))
    protocol=SimpleNamespace(v6=SimpleNamespace(load_registration_v6=lambda:{}),
        v5=SimpleNamespace(load_split=lambda:{},load_rows=lambda ids,with_targets:pd.DataFrame([{'toy':1}])),
        AE3_UID='AE3',fuel_parts=lambda:{'JetA':{}},
        load_draws=lambda:[(f'draw_{i:02d}',{}) for i in range(64)])
    monkeypatch.setattr(p73,'import_protocol',lambda root:(protocol,SimpleNamespace()))
    monkeypatch.setattr(p73,'load_selected_core',lambda *a:pytest.fail('compiled core requested'))
    def study(protocol,backend,reg,reg6,split,fit,ae3,fuels,draws,context,run,out,checkpoint):
        assert context.simulator_backend=='python' and context.workers==10 and len(draws)==64
        (out/'toy.csv').write_text('measured fixture\n')
        seen.append('study')
        return {'_sealed_outputs':{'toy.csv':pc.sha256_file(out/'toy.csv')}}
    monkeypatch.setattr(p73,'study_stage',study)
    monkeypatch.setattr(pc,'platform_info',lambda:{'platform':'Linux WSL'})
    result=w.c()
    assert result['status']=='COMPLETE' and seen==['contract','exception','study']
    assert pc.read_json(w.p73/'parity/parity.json')['checks']['frozen20']['status']=='PASS'
    assert not (w.p73/'parity/cpp.json').exists()


def test_d_preserves_freeze_diagnostic_generation_order(workflow, monkeypatch):
    from scripts.phase8.saf_surrogate import run as saf, timing
    w=workflow;seen=[]
    dump(w.root/pc.REGISTRATIONS['p73'],{})
    dump(w.p73/'terminal.json',{'status':'COMPLETE'})
    def producer_check(root,out,*,expected_simulator,selected):
        assert expected_simulator=='python-source'
        assert all('/sealed/' not in path and not path.endswith('.csv') for path in selected)
        seen.append('producer')
    monkeypatch.setattr(pc,'verify_stage_producer',producer_check)
    def freeze(root,output,reg,reg_sha,context,run,backend,*,platform_info,source_hashes,provenance_dependencies):
        assert backend=='mlx' and context.binary_path is None
        assert set(provenance_dependencies)=={'local_pc_origin','named_prerequisite_terminal'}
        seen.append('freeze')
    monkeypatch.setattr(saf,'freeze',freeze)
    monkeypatch.setattr(timing,'source_diagnostics',lambda root,out,reg,context,run,backend:seen.append('diagnostics'))
    monkeypatch.setattr(saf,'wait_generation',lambda root,out,context,run,sha:seen.append('generate'))
    assert w.d()['status']=='COMPLETE'
    assert seen==['producer','freeze','diagnostics','generate']


def test_consumed_d_f_are_refused_before_acquisition(workflow):
    w=workflow;w.saf.mkdir()
    with pytest.raises(FileExistsError):w.d()
    assert w._acquired==[]
    (w.saf/'score_reservation.json').write_text('{}')
    with pytest.raises(RuntimeError,match='consumed'):w.f()
    assert w._acquired==[]


def test_f_propagates_explicit_backend_and_keeps_scientific_failure(workflow, monkeypatch):
    from scripts.phase8.saf_surrogate import score
    w=workflow;w.saf.mkdir();seen=[]
    monkeypatch.setattr(score,'seal_predictions',lambda out,reg,sha,context,run,*,backend:seen.append(('seal',backend)))
    def scored(root,out,reg,sha,context,run,*,backend):
        seen.append(('score',backend))
        return {'fidelity_pass':False,'ranking_pass':True,'precision_pass':True}
    monkeypatch.setattr(score,'score_all',scored)
    result=w.f()
    assert result['status']=='COMPLETE' and result['scientific_verdict']=='FAIL'
    assert seen==[('seal','mlx'),('score','mlx')]


@pytest.mark.parametrize('status,completed',[('FAIL',True),('BLOCKED',False)])
def test_g_only_checkpoints_completed_nozzle_outputs(workflow, monkeypatch,status,completed):
    from scripts.phase8.nozzle_ode import run as nozzle
    w=workflow;seen=[]
    def execute(root,*,context_factory,source_loader,portable,backend):
        assert root==w.root and portable and backend=='mlx'
        assert context_factory.__self__ is w and source_loader is pc.load_source_properties
        dump(w.nozzle/'terminal.json',{'status':status,'execution_complete':completed,'outputs_complete':completed,
            'state':'COMPLETE' if completed else 'BLOCKED'})
        seen.append('nozzle')
    monkeypatch.setattr(nozzle,'run',execute)
    if completed:
        result=w.g()
        assert result['status']=='COMPLETE' and result['scientific_verdict']=='FAIL'
    else:
        with pytest.raises(RuntimeError):w.g()
    assert seen==['nozzle']


def test_h_measures_selected_cpu64_at_one_and_ten_workers_without_gpu_gate(workflow, monkeypatch):
    from scripts.phase8.saf_surrogate import models,run as saf,teacher,timing,study,score
    w=workflow;w.saf.mkdir();seen=[];clock=[0]
    class Product:
        properties=public=draws={}
        def predict(self, batch):clock[0]+=10**9;return batch
    def product(path,*,require_deployment,backend):
        assert backend=='mlx' and require_deployment is False
        seen.append(('product',backend));return Product()
    monkeypatch.setattr(models,'load_product',product)
    monkeypatch.setattr(models,'summarize_draws',lambda queries,predictions:None)
    monkeypatch.setattr(pc.time,'perf_counter_ns',lambda:clock[0])
    dump(w.saf/'splits/study.json',[{'fixture':i} for i in range(4096)])
    dump(w.saf/'sole_score_summary.json',{'fidelity_pass':True,'ranking_pass':True,'precision_pass':True})
    for letter in 'cdefg':dump(w.metadata/'checkpoints'/f'{letter}.json',{'elapsed_seconds':1})
    for letter in 'def':dump(w.metadata/'runs'/letter/'terminal.json',{'fixture':letter})
    def spec(root,out,context,sha,stage,argv):
        return {'simulator':context.simulator_identity,'output':str(out),
            'owner_lease':{'snapshot_path':f'outputs/{stage}_snapshot.json'},'command_spec_path':f'outputs/{stage}_spec.json'}
    monkeypatch.setattr(saf,'command_spec',spec)
    class Teacher:
        def __init__(self,spec,properties,public,draws,workers,*,run):
            assert spec['simulator']['name']=='python-v6' and spec['workers']==workers
            seen.append(('teacher',workers));self.workers=workers
        def __call__(self,batch):clock[0]+=2*10**9;return batch
        def close(self):seen.append(('close',self.workers))
    monkeypatch.setattr(teacher,'ParallelTeacher',Teacher)
    def screen(root,out,reg,sha,context,run,*,backend):
        assert backend=='mlx';seen.append(('screen',backend))
        return {'invalid_prediction_rows':0,'selected_reference_converged':640,'cpu64_total_seconds':.1}
    monkeypatch.setattr(study,'screen',screen)
    monkeypatch.setattr(timing,'gpu_measurements',lambda *args:{'available':False,'state':'FAIL'})
    def receipt(out,reg,sha,scored,timed):
        assert timed['operational_pass'] and timed['gpu_available'] is False
        dump(out/'deployment_receipt.json',{'state':'PASS'})
    monkeypatch.setattr(score,'deployment_receipt',receipt)
    result=w.h()
    measured=pc.read_json(w.saf/'timing.json')
    assert result['scientific_verdict']=='PASS' and measured['scoring_backend']=='numpy'
    assert measured['teacher_full_cycle_requests']==2*37449
    assert seen==[('product','mlx'),('teacher',1),('close',1),('teacher',10),('close',10),('screen','mlx')]
    assert pc.read_json(w.saf/'pc_pipeline_provenance.json')['stages'].keys()=={'d','e','f','h'}


def producer(root, output, selected):
    identity={'schema':'pc-python-v6-v1','simulator_identity_sha256':'python-source'}
    reservation={'identity':identity}
    dump(output/'reservation.json',reservation)
    terminal={'identity':identity,'status':'COMPLETE','exit_code':0,
        'reservation_sha256':pc.sha256_file(output/'reservation.json'),
        'artifact_hashes':{selected:pc.sha256_file(root/selected)}}
    dump(output/'terminal.json',terminal)
    dump(output/'released_lease.json',{'identity':identity,'state':'RELEASED',
        'terminal_sha256':pc.sha256_file(output/'terminal.json')})


def test_selected_producer_projection_rejects_byte_drift(tmp_path):
    target='outputs/train.csv';(tmp_path/'outputs').mkdir();(tmp_path/target).write_text('fixture')
    output=tmp_path/'outputs/producer';producer(tmp_path,output,target)
    assert pc.verify_stage_producer(tmp_path,output,expected_simulator='python-source',selected=[target])['status']=='COMPLETE'
    (tmp_path/target).write_text('changed')
    with pytest.raises(ValueError,match='artifact drift'):
        pc.verify_stage_producer(tmp_path,output,expected_simulator='python-source',selected=[target])
    with pytest.raises(ValueError,match='another Python'):
        pc.verify_stage_producer(tmp_path,output,expected_simulator='foreign')


def test_nozzle_source_loader_decodes_only_complete_train_named_projection(workflow, monkeypatch):
    from scripts.phase8.nozzle_ode import inputs
    w=workflow
    reg=json.loads((Path(__file__).resolve().parents[1]/pc.REGISTRATIONS['nozzle']).read_text())
    dump(w.root/pc.REGISTRATIONS['saf'],{'artifact_root':'outputs/saf'})
    w.saf.mkdir();w.nozzle.mkdir()
    producer_sha=pc.sha256_file(w.root/pc.REGISTRATIONS['saf'])
    train_ids,named_ids=inputs.expected_source_ids(reg)
    train=[{'design_id':name,'prefix_index':i,'draw_id':f'draw_{i%64:02d}','input_sha256':f'query{i}'}
           for i,name in enumerate(train_ids)]
    named=[{'named_case_id':name,'fuel':name.split('|')[0],'op':name.split('|')[1],
            'input_sha256':f'named{i}'} for i,name in enumerate(named_ids)]
    pre={'simulator_identity_sha256':'python-source','registration_sha256':producer_sha,
         'cases':{'train':train,'named_central':named}}
    dump(w.saf/'property_inputs_manifest.json',pre)
    pre_sha=pc.sha256_file(w.saf/'property_inputs_manifest.json')
    dump(w.saf/'splits/train.json',train);dump(w.saf/'splits/named_central.json',named)
    common={'status':'converged','gamma4':1.3,'R4_J_kg_K':287.,'cp4_J_kg_K':1.3*287/.3,
            'source_registration_sha256':producer_sha,'binary_sha256':'','property_manifest_sha256':pre_sha,
            'source_commit':'manufactured','f_JetA':1,'f_HEFA':0,'f_FT':0,'f_ATJ':0,'thrust_fraction':.55,
            'ff_kg_s':'forbidden_numeric_target','T4_K':'forbidden_numeric_target'}
    train_rows=[dict(common,**row,split='train',species_row_index=i) for i,row in enumerate(train)]
    named_rows=[]
    for row in named:
        value=dict(common,**row,in_product_API='True',fuel_parts='{"JetA":1}',
                   prerequisite_registration_sha256='named-source',full_state_sha256='full-state')
        if row['fuel']=='JetA_dooley2010':
            value.update(in_product_API='False',fuel_parts='{"JetA_dooley2010":1}',
                         f_JetA='',f_HEFA='',f_FT='',f_ATJ='')
        named_rows.append(value)
    def csv_rows(path,rows):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csv_rows(w.saf/'teacher_rows.csv',train_rows);csv_rows(w.saf/'named_central_properties.csv',named_rows)
    y=np.zeros((4096,492));y[:,0]=1
    species=[f'S{i}' for i in range(492)]
    np.savez(w.saf/'teacher_species.npz',Y4=y,species_order=np.asarray(species),
             **{key:np.asarray([row[key] for row in train]) for key in ('design_id','draw_id','prefix_index','input_sha256')})
    dump(w.saf/'frozen_properties.json',{'species_order':species})
    dump(w.saf/'generation_terminal.json',{'state':'COMPLETE','simulator_identity_sha256':'python-source'})
    allowed=('teacher_rows.csv','teacher_species.npz','named_central_properties.csv','frozen_properties.json')
    manifest={'registration_sha256':producer_sha,
        'property_inputs_manifest':{'path':'property_inputs_manifest.json','sha256':pre_sha},
        'producer_terminal':{'path':'generation_terminal.json','sha256':pc.sha256_file(w.saf/'generation_terminal.json')},
        'artifacts':{name:pc.sha256_file(w.saf/name) for name in allowed}}
    dump(w.saf/'property_manifest.json',manifest)
    projection=allowed+('property_inputs_manifest.json','property_manifest.json','generation_terminal.json',
                        'splits/train.json','splits/named_central.json')
    stage=w.metadata/'runs/d';producer(w.root,stage,'outputs/saf/teacher_rows.csv')
    terminal=pc.read_json(stage/'terminal.json')
    terminal['artifact_hashes']={f'outputs/saf/{name}':pc.sha256_file(w.saf/name) for name in projection}
    dump(stage/'terminal.json',terminal)
    released=pc.read_json(stage/'released_lease.json');released['terminal_sha256']=pc.sha256_file(stage/'terminal.json')
    dump(stage/'released_lease.json',released)
    path_open=Path.open
    opened=[]
    def allowed_open(path,*args,**kwargs):
        assert '/sealed/' not in str(path), f'sealed target accessed: {path}'
        opened.append(path)
        return path_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',allowed_open)
    properties,hashes=pc.load_source_properties(w.root,reg,w._context,w.nozzle)
    assert len(properties)==4164 and len(hashes)==9
    assert all('/sealed/' not in name for name in hashes)
    assert [p['source_id'] for p in properties[:4096]]==train_ids
    assert pc.read_json(w.nozzle/'source_manifest.json')['simulator_identity_sha256']=='python-source'


def test_generated_nozzle_readme_reports_actual_python_provenance(tmp_path):
    from scripts.phase8.nozzle_ode.run import technical_readme
    pc_out=tmp_path/'pc';mac_out=tmp_path/'mac';pc_out.mkdir();mac_out.mkdir()
    technical_readme(pc_out,portable=True,simulator_backend='python',backend='mlx')
    technical_readme(mac_out)
    pc_text=(pc_out/'README.md').read_text();mac_text=(mac_out/'README.md').read_text()
    assert 'frozen Python-v6 burner-state' in pc_text and 'frozen C++ burner-state' not in pc_text
    assert 'python scripts/pc_pipeline.py --run --resume --stop-after g --backend mlx' in pc_text
    assert 'inherited Track 4 evidence as available or incomplete' in pc_text
    assert 'caffeinate' not in pc_text
    assert 'caffeinate -i nice -n 15' in mac_text and 'frozen C++ burner-state' in mac_text
