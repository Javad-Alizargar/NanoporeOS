"""Independently rescore stored v5 arrays and replay matched policy decisions.

No PDE solves; exact fields/flux come from the pre-frozen disjoint SymPy source.
"""
from pathlib import Path
import sys
import hashlib,json
from datetime import datetime,timezone
import numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'manufactured_families'))
from independent_reference import Parameters,fields,parameter_mapping,FARADAY

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def exact_errors(case,grid,y,current):
    p=parameter_mapping(case);nx,nr=grid
    x=(np.arange(nx)+.5)*p.length_nm/nx;r=(np.arange(nr)+.5)*p.radius_nm/nr
    xx,rr=np.meshgrid(x,r,indexing='ij');exact=fields(p,xx,rr)
    nodes,weights=np.polynomial.legendre.leggauss(96)
    rad=(nodes+1)*p.radius_nm/2;weights=weights*p.radius_nm/2
    flux=fields(p,np.full_like(rad,p.length_nm/2),rad)['jx']
    reference=2*np.pi*FARADAY*p.salt_mM*1e-9*np.sum(weights*rad*(1.96e-9*flux[0]-2.03e-9*flux[1]))
    return dict(max_phi_error_VT=float(np.max(abs(y[0]-exact['psi']))),
        max_plus_relative_error=float(np.max(abs(np.exp(y[1])/exact['q'][0]-1))),
        max_minus_relative_error=float(np.max(abs(np.exp(y[2])/exact['q'][1]-1))),
        mid_current_relative_error=float(abs(current-reference)/max(abs(reference),1e-15)),
        mid_current_absolute_error_A=float(abs(current-reference)),exact_mid_current_A=float(reference))

def replay(policy,snapshots,budgets):
    if policy=='uniform_finest':
        idx=len(snapshots)-1;visited=[idx]
        selected=idx if snapshots[idx]['status']=='PASS' else None
    else:
        visited=[];selected=None
        for idx,fine in enumerate(snapshots):
            visited.append(idx)
            if policy=='residual_only':
                if fine['status']=='PASS':selected=idx;break
                continue
            if idx==0 or fine['status']!='PASS' or snapshots[idx-1]['status']!='PASS':continue
            coarse=snapshots[idx-1]
            delta_current=abs(fine['current']-coarse['current'])/max(abs(fine['current']),1e-15)
            passed=delta_current*(2/3)<=budgets['mid_current_relative_error']
            if policy=='field_matched' and passed:
                xx,rr=np.meshgrid(coarse['x'],coarse['r'],indexing='ij')
                points=np.column_stack([xx.ravel(),rr.ravel()])
                interpolated=np.array([RegularGridInterpolator((fine['x'],fine['r']),f)(points).reshape(xx.shape)
                    for f in fine['y']])
                errors=[np.max(abs(coarse['y'][0]-interpolated[0])),
                    np.max(abs(np.exp(coarse['y'][1]-interpolated[1])-1)),
                    np.max(abs(np.exp(coarse['y'][2]-interpolated[2])-1))]
                keys=['max_phi_error_VT','max_plus_relative_error','max_minus_relative_error']
                passed=all(e*(2/3)<=budgets[k] for e,k in zip(errors,keys))
            if passed:selected=idx;break
    return selected,visited,sum(int(np.prod(snapshots[i]['y'].shape[1:])) for i in visited)

def main():
    pointer=load(ROOT/'results/verification_v5/latest.json');summary_path=ROOT/pointer['summary']
    assert digest(summary_path)==pointer['sha256']
    summary=load(summary_path);directory=summary_path.parent
    protocol_path=directory/'provenance/verification_v5a_protocol.json';protocol=load(protocol_path)
    assert digest(protocol_path)==summary['protocol_sha256']
    header=load(directory/'header.json')
    original_path=ROOT/'config/verification_v5_protocol.json';original=load(original_path)
    assert digest(original_path)==protocol['supersedes_protocol_sha256']
    for key in ['development_cases','held_out_cases','exact_evaluation_tolerances',
        'controller_assumed_order','controller_safety_factor','controller_consecutive_pairs']:
        assert original[key]==protocol[key],key
    for name in ['residual_only','current_matched','field_matched']:
        assert original['policies'][name]==protocol['policies'][name]
    assert original['policies']['uniform_finest']=='Direct192x96 discrete PASS'
    assert protocol['policies']['uniform_finest']=='Direct128x64 strict discrete PASS'
    assert protocol['mesh_ladder']==[[8,4],[16,8],[32,16],[64,32],[128,64]]
    assert datetime.fromisoformat(protocol['declared_utc'])<datetime.fromisoformat(header['created_utc'])
    pilot_candidates=list((ROOT/'results/verification_v5/campaigns').glob(protocol['supersedes_protocol_sha256'][:12]+'_*'))
    assert len(pilot_candidates)==1
    pilot=pilot_candidates[0];pilot_results=[]
    for result_path in (pilot/'cases').glob('*/result.json'):
        result=load(result_path)
        assert result['manufactured_case']['name']=='v5_development_polynomial'
        pilot_results.append([result['scenario']['nx'],result['scenario']['nr']])
    assert sorted(pilot_results)==[[12,6],[24,12],[48,24],[96,48]]
    held=protocol['held_out_cases'];development=protocol['development_cases']
    assert len(held)==24 and len(development)==3
    tuple_fields=['length_nm','radius_nm','salt_mM','drive_VT','amplitude_psi','bplus','bminus','wall_amplitude']
    tuples={tuple(case[k] for k in tuple_fields) for case in held}
    assert len(tuples)==8
    for key,name in [('families_source_sha256','manufactured_families.py'),
        ('source_aware_audit_source_sha256','manufactured.py'),('controller_sha256','accuracy.py'),
        ('matched_controller_sha256','acceptance_matched.py'),('driver_sha256','verification_v5.py')]:
        assert digest(directory/'provenance'/name)==summary['source_hashes'][key]
    budgets=protocol['exact_evaluation_tolerances'];maximum=0.;array_count=0
    aggregate={name:dict(accurate=0,inaccurate=0,unresolved=0,cells=0) for name in protocol['policies']}
    rows=[];paired_corrected=[];paired_lost=[];per_family={};current_failures=[]
    for record in summary['records']:
        decision_path=directory/(record['case']['name']+'_decisions.json')
        assert digest(decision_path)==record['decisions_before_scoring_sha256']
        blind=load(decision_path)
        assert all('actual_errors' not in d and 'exact_accuracy_pass' not in d for d in blind['policies'])
        snapshots=[];passes=[];independent_errors=[]
        for grid,identity,reported in zip(record['grids'],record['solve_identities'],record['errors']):
            case_dir=directory/'cases'/identity;manifest=load(case_dir/'manifest.json')
            assert manifest['identity']==identity and manifest['protocol_sha256']==summary['protocol_sha256']
            for name,expected in manifest['files'].items():assert digest(case_dir/name)==expected
            saved=load(case_dir/'result.json')
            assert saved['identity']==identity and saved['manufactured_case']==record['case']
            assert saved['verification_only'] and not saved['training_qualified']
            assert saved['model']==protocol['model'] and saved['scenario']['model']==protocol['model']
            payload=dict(case=record['case'],nx=grid[0],nr=grid[1],
                implementation=saved['implementation'],environment=header['environment'])
            expected_identity=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            assert expected_identity==identity,'Semantic identity digest mismatch'
            with np.load(case_dir/'fields.npz',allow_pickle=False) as a:
                y=a['y'];x=a['x_nm'];r=a['r_nm']
            assert y.shape==(3,*grid) and np.all(np.isfinite(y))
            p=parameter_mapping(record['case'])
            np.testing.assert_allclose(x,(np.arange(grid[0])+.5)*p.length_nm/grid[0],rtol=1e-14,atol=1e-14)
            np.testing.assert_allclose(r,(np.arange(grid[1])+.5)*p.radius_nm/grid[1],rtol=1e-14,atol=1e-14)
            exact=exact_errors(record['case'],grid,y,saved['mid_current_A'])
            discrepancy=max(abs(exact[k]-reported[k]) for k in exact)
            maximum=max(maximum,discrepancy)
            assert discrepancy<1e-9,(record['case']['name'],grid,discrepancy)
            accurate=saved['status']=='PASS' and all(exact[k]<=v for k,v in budgets.items())
            passes.append(accurate)
            independent_errors.append(exact)
            snapshots.append(dict(status=saved['status'],y=y,x=x,r=r,current=saved['mid_current_A'],
                seconds=saved['construction_and_solve_seconds']))
            array_count+=1
        assert passes==record['exact_accuracy_passes']
        by_policy={}
        for decision,frozen in zip(record['policies'],blind['policies']):
            assert decision['policy']==frozen['policy']
            for key,value in frozen.items():assert decision[key]==value
            selected,visited,cells=replay(decision['policy'],snapshots,budgets)
            assert selected==decision['selected_index'] and visited==decision['visited_indices']
            assert cells==decision['attempted_cells']
            assert decision['exact_accuracy_pass']==(None if selected is None else passes[selected])
            np.testing.assert_allclose(sum(snapshots[i]['seconds'] for i in visited),decision['attempted_solve_seconds'],rtol=1e-14)
            np.testing.assert_allclose(decision['attempted_solve_seconds']+decision['selection_seconds'],
                decision['solve_plus_selection_seconds'],rtol=1e-14)
            by_policy[decision['policy']]=decision
            if decision['policy'] in ['current_matched','field_matched']:
                assert decision['shared_estimator_factor']==2/3
                for indicator in decision['indicators']:
                    expected=indicator['raw_differences']['mid_current_relative_error']*(2/3)
                    assert abs(indicator['estimated_errors']['mid_current_relative_error']-expected)<1e-14
            if record['split']=='held_out':
                category='unresolved' if selected is None else 'accurate' if passes[selected] else 'inaccurate'
                aggregate[decision['policy']][category]+=1;aggregate[decision['policy']]['cells']+=cells
        if record['split']=='held_out':
            current=by_policy['current_matched'];field=by_policy['field_matched']
            if current['exact_accuracy_pass'] is False and field['exact_accuracy_pass'] is True:
                paired_corrected.append(record['case']['name'])
                selected=current['selected_index'];field_selected=field['selected_index']
                e=independent_errors[selected]
                current_failures.append(dict(case=record['case']['name'],family=record['case']['family'],
                    current_selected_grid=record['grids'][selected],field_selected_grid=record['grids'][field_selected],
                    failed_budgets=[k for k,v in budgets.items() if e[k]>v],
                    current_selected_exact_errors=e,field_selected_exact_errors=independent_errors[field_selected]))
            if current['exact_accuracy_pass'] is True and field['exact_accuracy_pass'] is not True:
                paired_lost.append(record['case']['name'])
            family=record['case']['family'];per_family.setdefault(family,dict(current_bad=0,field_bad=0))
            per_family[family]['current_bad']+=int(current['exact_accuracy_pass'] is False)
            per_family[family]['field_bad']+=int(field['exact_accuracy_pass'] is False)
        rows.append(dict(case=record['case']['name'],split=record['split'],family=record['case']['family'],
            finest_exact_pass=passes[-1]))
    for row in summary['held_out_policy_aggregate']:
        a=aggregate[row['policy']]
        assert a['accurate']==row['actual_accurate_accepts'] and a['inaccurate']==row['inaccurate_accepts']
        assert a['unresolved']==row['unresolved'] and a['cells']==row['total_attempted_cells']
    failures_by_budget={k:sum(k in row['failed_budgets'] for row in current_failures) for k in budgets}
    report=dict(status='PASS',checked_utc=datetime.now(timezone.utc).isoformat(),
        summary_status=summary['status'],summary_sha256=digest(summary_path),
        arrays_independently_scored=array_count,records=len(rows),
        maximum_error_metric_discrepancy=maximum,policy_replay='PASS',
        manifest_and_source_snapshots='PASS',decisions_before_exact_scoring='PASS',
        resource_amendment=dict(status='PASS_LOCAL_RECORDS',original_protocol_sha256=digest(original_path),
            amended_protocol_sha256=digest(protocol_path),declared_utc=protocol['declared_utc'],
            campaign_created_utc=header['created_utc'],pilot_only_development_grids=sorted(pilot_results),
            unchanged_tuples_budgets_and_estimator_rules=True,
            uniform_baseline_resource_change='Direct192x96 becomes Direct128x64 with resource-valid ladder',
            statistical_unit='8 repeated parameter tuples across3 families',
            limitation='Local source/hash/time records, not external preregistration or human blinding.'),
        aggregate=aggregate,paired_current_bad_field_accurate=paired_corrected,
        paired_current_accurate_field_lost=paired_lost,per_family=per_family,
        current_inaccurate_selected_failures=current_failures,current_failure_counts_by_budget=failures_by_budget,
        records_checked=rows,PDE_solves=0,paid_calls=0,
        source_hashes={p.name:digest(p) for p in [Path(__file__),HERE.parent/'manufactured_families/independent_reference.py']},
        limitations='Independent continuous truth/scoring and policy replay; no independent discrete PDE solve, physical validation or certified estimator.')
    (HERE/'saved_campaign_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records_checked'},indent=2))
if __name__=='__main__':main()
