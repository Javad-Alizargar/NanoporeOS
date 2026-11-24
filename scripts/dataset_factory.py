"""Failure-aware collection and frozen scenario-grouped neutral baseline exercise."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import sys
import time
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import Scenario,ohmic_current,sha256,write_json,source_identity
from nanoporeos.datasets import load_result
from nanoporeos.qualification import neutral_qualification,qualified_finest


def group_id(s):
    physical={k:v for k,v in s.items() if k not in ['nx','nr']}
    return hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()


def main():
    campaign=json.loads((ROOT/'results/matrix/campaign.json').read_text())
    protocol=ROOT/'config/learning_protocol.json'
    if campaign['status']!='FINISHED':raise RuntimeError('Matrix incomplete')
    groups=collections.defaultdict(list)
    for r in campaign['records']:groups[group_id(r['scenario'])].append(r)
    neutral=sorted(k for k,v in groups.items() if v[0]['scenario']['wall_offset_mV']==0)
    splits={k:'train' if i<7 else 'validation' if i<9 else 'test' for i,k in enumerate(neutral)}
    output=ROOT/'results/dataset';output.mkdir(parents=True,exist_ok=True)
    rows=[];failures=[];qualified=[];qualified_ids=set()
    for gid,records in groups.items():
        for rec in records:
            directory=ROOT/'results/matrix'/rec['case_id'];summary,arrays=load_result(directory)
            s=Scenario(**rec['scenario']);reference=ohmic_current(s)
            qualify=neutral_qualification(s,rec['status'],rec['diagnostics']['current_A'],arrays)
            if qualify:qualified_ids.add(rec['case_id'])
            rows.append(dict(case_id=rec['case_id'],scenario_group=gid,split=splits.get(gid,'excluded'),
                parent_directory=str(directory.relative_to(ROOT)),status=rec['status'],
                dataset_class='numerically_verified_simulation' if qualify else 'exploratory_numerical_simulation',
                training_qualified=qualify,qualification='Neutral analytical full-field and current checks' if qualify else 'HOLD: failed or outside analytically qualified domain',
                current_A=rec['diagnostics']['current_A'],legacy_reported_converged=rec['legacy_reported_converged'],
                result_sha256=sha256(directory/'result.json')))
            if not rec['legacy_reported_converged']:failures.append(dict(case_id=rec['case_id'],scenario=rec['scenario'],
                source='reexecuted supplied legacy kernel',status='FAIL',artifact=str((directory/'legacy.json').relative_to(ROOT)),
                historical_executed_source_identity='UNRESOLVED'))
        if gid in splits:
            fine=qualified_finest(records,qualified_ids)
            qualified.append(dict(group=gid,split=splits[gid],scenario=fine['scenario'],target_A=fine['diagnostics']['current_A']))
    with (output/'records.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(output/'failure_records.json',failures)
    write_json(output/'splits.json',dict(protocol_sha256=sha256(protocol),groups=splits,mesh_children_never_cross_split=True))
    write_json(output/'dataset_card.json',dict(schema_version='1.0',records=len(rows),physical_scenarios=len(groups),
        neutral_analytical_qualified_records=sum(r['training_qualified'] for r in rows),
        excluded_charged_wall_records=sum(r['scenario']['wall_offset_mV']!=0 for r in campaign['records']),
        excluded_unqualified_records=sum(not r['training_qualified'] for r in rows),legacy_failure_records=len(failures),
        data_mode='local_only',experimental_observations=0,simulator_truth_only=True,
        status='PARTIAL_DOMAIN_QUALIFICATION',source_sha256=source_identity(),
        explanation='Original exports retain conservative exploratory metadata. This collection adds traceable qualification for neutral analytic cases only; no source result was overwritten.',
        qualification_rule='Neutral wall, discrete PASS, analytical current and both full concentration/potential fields; no failed finest target admitted',
        sampling='Full Cartesian legacy matrix; no failed historical scenarios dropped; no resampling',
        license='UNRESOLVED_NOT_RELEASED',field_format='NPZ compressed arrays, allow_pickle=False',
        units='nm, V, mol/m3, A; fields [axial_cell, radial_cell]',
        round_trip='Each export loaded with SHA256 validation before collection',
        resources=dict(small='one bounded case <=648 cells',medium='108-case original matrix',large='not exposed; resource/domain gates required')))
    def features(items,physical=False):
        return np.array([[r['scenario']['radius_nm']**2*r['scenario']['salt_mM']*r['scenario']['voltage_mV']/r['scenario']['length_nm']]
            if physical else [1,r['scenario']['radius_nm'],r['scenario']['salt_mM'],r['scenario']['voltage_mV']] for r in items])
    train=[r for r in qualified if r['split']=='train'];test=[r for r in qualified if r['split']=='test']
    results=[]
    for method in ['analytical','raw_linear_regression','physical_feature_regression']:
        start=time.perf_counter();fit_seconds=0
        if method=='analytical':prediction=np.array([ohmic_current(Scenario(**r['scenario'])) for r in test]);coefficients=[]
        else:
            physical=method=='physical_feature_regression'
            coefficients=np.linalg.lstsq(features(train,physical),np.array([r['target_A'] for r in train]),rcond=None)[0]
            fit_seconds=time.perf_counter()-start;start=time.perf_counter()
            prediction=features(test,physical)@coefficients
            coefficients=coefficients.tolist()
        inference_seconds=time.perf_counter()-start
        truth=np.array([r['target_A'] for r in test]);error=abs(prediction-truth)
        results.append(dict(method=method,coefficients=coefficients,fit_seconds=fit_seconds,
            total_test_inference_seconds=inference_seconds,MAE_A=float(error.mean()),max_absolute_error_A=float(error.max()),
            normalized_MAE=float(error.mean()/max(1e-18,max(abs(truth)))),test_predictions_A=prediction.tolist()))
    write_json(ROOT/'results/learning.json',dict(status='LIMITED_NEUTRAL_BASELINE_DEMONSTRATION',protocol_sha256=sha256(protocol),
        train_groups=7,validation_groups=2,locked_test_groups=len(test),test=test,methods=results,
        charged_wall_surrogate='NOT_EVALUABLE_DOMAIN_HOLD',neural_PINN='NOT_IMPLEMENTED',
        OOD_policy='ABSTAIN for charged-wall or out-of-range cases; lookup inside tested neutral domain only',
        break_even_claim='NONE: analytical baseline already solves this limit directly',
        interpretation='Small deterministic numerical exercise, no performance/generalization/experimental claim'))
    # One real, self-contained, allowlisted dataset export, plus collection tables.
    example=next(r for r in rows if r['training_qualified'] and r['split']=='test')
    directory=ROOT/example['parent_directory']
    package_card=dict(scope='One self-contained case plus collection metadata',included_case_id=example['case_id'],
        case_directory='case/',collection_records=len(rows),other_case_fields='EXTERNAL_NOT_INCLUDED',
        note='The collection CSV references full-run workspace paths. Only the named case arrays/diagnostics are contained in this example archive.')
    with zipfile.ZipFile(output/'example_dataset.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['fields.npz','result.json','manifest.json','sections.csv']:z.write(directory/name,'case/'+name)
        for name in ['records.csv','failure_records.json','splits.json','dataset_card.json']:z.write(output/name,name)
        z.writestr('PACKAGE_CARD.json',json.dumps(package_card,indent=2))
    write_json(output/'manifest.json',dict(files={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(dict(records=len(rows),qualified=sum(r['training_qualified'] for r in rows),failures=len(failures),test_groups=len(test)))


if __name__=='__main__':main()
