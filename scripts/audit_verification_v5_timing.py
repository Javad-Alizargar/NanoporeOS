"""Audit independent live timing decisions/selected arrays; no new PDE solve."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks/manufactured_families')]
from nanoporeos.core import sha256,write_json
from independent_reference import fields,parameter_mapping,FARADAY

def main():
    pointer=json.loads((ROOT/'results/verification_v5_timing/latest.json').read_text());path=ROOT/pointer['summary']
    assert sha256(path)==pointer['sha256'];data=json.loads(path.read_text());assert data['status']=='FINISHED_RANDOMIZED_REPEATED_POLICY_EXECUTIONS'
    directory=path.parent;protocol=json.loads((directory/'provenance/verification_v5a_protocol.json').read_text())
    import hashlib
    assert hashlib.sha256(json.dumps(data['sources'],sort_keys=True).encode()).hexdigest()==data['source_identity']
    for key,name in [('driver_sha256','time_verification_v5.py'),('families_source_sha256','manufactured_families.py'),
            ('source_aware_audit_source_sha256','manufactured.py'),('matched_sha256','acceptance_matched.py'),('accuracy_sha256','accuracy.py')]:
        assert sha256(directory/'provenance'/name)==data['sources'][key]
    assert sha256(directory/'provenance/verification_v5a_protocol.json')==data['sources']['protocol_sha256']
    order=json.loads((directory/'execution_order.json').read_text())['order'];assert len(order)==len(data['records'])==72
    expected=[];rng=np.random.default_rng(protocol['timing_followup']['seed'])
    for rep in range(3):
        pairs=[(case,policy) for case in protocol['timing_followup']['cases'] for policy in protocol['policies']]
        expected.extend(dict(repetition=rep+1,case=pairs[i][0],policy=pairs[i][1]) for i in rng.permutation(len(pairs)))
    assert order==expected
    metric_discrepancy=0.;grid_consistency={}
    cases={r['name']:r for r in protocol['held_out_cases']};nodes,weights=np.polynomial.legendre.leggauss(96)
    for i,record in enumerate(data['records']):
        assert record['design']==order[i] and record['source_identity']==data['source_identity']
        decision_path=directory/record['decision_file'];assert sha256(decision_path)==record['decision_before_scoring_sha256']
        blind=json.loads(decision_path.read_text());assert 'errors' not in blind
        assert np.isclose(sum(a['seconds'] for a in blind['attempts']),blind['decision']['attempted_solve_seconds'],rtol=1e-12)
        assert record['wall_seconds']>=blind['decision']['attempted_solve_seconds']
        key=(record['design']['case'],record['design']['policy'])
        if key in grid_consistency:assert grid_consistency[key]==record['selected_grid']
        grid_consistency[key]=record['selected_grid']
        if record['selected_fields'] is None:assert record['exact_accuracy_pass'] is None;continue
        array_path=directory/record['selected_fields'];assert sha256(array_path)==record['selected_fields_sha256']
        with np.load(array_path,allow_pickle=False) as z:y=z['y'];x=z['x_nm'];r=z['r_nm'];current=float(z['current_A'])
        p=parameter_mapping(cases[record['design']['case']]);xx,rr=np.meshgrid(x,r,indexing='ij');exact=fields(p,xx,rr)
        rad=(nodes+1)*p.radius_nm/2;wr=weights*p.radius_nm/2;flux=fields(p,np.full_like(rad,p.length_nm/2),rad)['jx']
        reference=float(2*np.pi*FARADAY*p.salt_mM*1e-9*np.sum(wr*rad*(1.96e-9*flux[0]-2.03e-9*flux[1])))
        errors=dict(max_phi_error_VT=float(np.max(abs(y[0]-exact['psi']))),
            max_plus_relative_error=float(np.max(abs(np.exp(y[1])/exact['q'][0]-1))),
            max_minus_relative_error=float(np.max(abs(np.exp(y[2])/exact['q'][1]-1))),
            mid_current_relative_error=abs(current-reference)/max(abs(reference),1e-15))
        for k,v in errors.items():metric_discrepancy=max(metric_discrepancy,abs(v-record['errors'][k]))
        assert all(np.isclose(v,record['errors'][k],rtol=1e-8,atol=1e-10) for k,v in errors.items())
        assert bool(all(errors[k]<=v for k,v in protocol['exact_evaluation_tolerances'].items()))==record['exact_accuracy_pass']
    paired=[]
    for case in protocol['timing_followup']['cases']:
        means={policy:float(np.mean([r['wall_seconds'] for r in data['records'] if r['design']['case']==case and r['design']['policy']==policy])) for policy in protocol['policies']}
        paired.append(dict(case=case,mean_seconds=means,field_to_finest_ratio=means['field_matched']/means['uniform_finest']))
    output=dict(status='PASS_LIVE_TIMING_ARRAY_AND_ORDER_AUDIT',summary_sha256=sha256(path),attempts=72,
        unique_combinations=6,repetitions=3,maximum_independent_metric_discrepancy=metric_discrepancy,
        selection_grids_identical_across_repetitions=True,aggregate=data['aggregate'],paired_case_mean_times=paired,
        median_case_mean_field_to_finest_ratio=float(np.median([r['field_to_finest_ratio'] for r in paired])),
        limitations='SameWindows host/6fixed cases; different coverage/accuracy across policies; descriptive timing, no population or hardware-independent speedup',
        independent_reference_source_sha256=sha256(ROOT/'benchmarks/manufactured_families/independent_reference.py'),
        audit_source_sha256=sha256(Path(__file__)))
    write_json(ROOT/'results/verification_v5_timing/audit.json',output);print(json.dumps(output,indent=2))

if __name__=='__main__':main()
