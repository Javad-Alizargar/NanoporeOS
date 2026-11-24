"""Read-only replay/scoring audit; no PDE solve or canonical writes."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks/manufactured')]
from nanoporeos.accuracy import Snapshot,select
from manufactured_math import Parameters,fields,VT,FARADAY


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def current_ref(c,order=96):
    node,weight=np.polynomial.legendre.leggauss(order)
    r=(node+1)*c.radius_nm/2;weight=weight*c.radius_nm/2
    f=fields(np.full_like(r,c.length_nm/2),r,c)
    return FARADAY*c.bulk_mol_m3*1e-9*2*np.pi*sum(z*d*np.sum(weight*r*f[name]['j_x'])
        for name,z,d in [('plus',1,1.96e-9),('minus',-1,2.03e-9)])


def main():
    audit=[];sets=[];max_scoring=0.;all_solve_pass=0
    for family in ['verification_v3','verification_v4']:
        pointer=json.loads((ROOT/'results'/family/'latest.json').read_text());path=ROOT/pointer['summary']
        assert sha(path)==pointer['sha256']
        summary=json.loads(path.read_text());directory=path.parent
        protocol_path=next((directory/'provenance').glob('*_protocol.json'))
        assert sha(protocol_path)==summary['protocol_sha256']
        protocol=json.loads(protocol_path.read_text());header=json.loads((directory/'header.json').read_text())
        assert header['source_hashes']==summary['source_hashes']
        assert protocol['declared_utc']<header['created_utc']
        actual_aggregate={name:dict(accurate=0,inaccurate=0,unresolved=0,cells=0,seconds=0.) for name in protocol['policies']}
        tuple_set=set();case_audits=[]
        for record in summary['records']:
            config=record['case']
            p=Parameters(length_nm=config['length_nm'],radius_nm=config['radius_nm'],bulk_mol_m3=config['salt_mM'],
                applied_thermal_voltages=config['drive_VT'],potential_amplitude=config['amplitude_psi'],
                plus_mu_amplitude=config['bplus'],minus_mu_amplitude=config['bminus'],wall_amplitude=config['wall_amplitude'])
            snapshots=[];recomputed=[];exact=current_ref(p)
            for grid,identity in zip(record['grids'],record['solve_identities']):
                parent=directory/'cases'/identity;manifest=json.loads((parent/'manifest.json').read_text())
                assert manifest['identity']==identity and manifest['protocol_sha256']==summary['protocol_sha256']
                assert all(sha(parent/name)==digest for name,digest in manifest['files'].items())
                saved=json.loads((parent/'result.json').read_text())
                assert saved['identity']==identity and saved['manufactured_case']==config
                assert saved['protocol_sha256']==summary['protocol_sha256']
                assert saved['verification_only'] and not saved['training_qualified']
                with np.load(parent/'fields.npz',allow_pickle=False) as z:y=z['y'];x=z['x_nm'];r=z['r_nm']
                assert y.shape==(3,*grid) and np.isfinite(y).all()
                np.testing.assert_allclose(x,(np.arange(grid[0])+.5)*p.length_nm/grid[0],rtol=0,atol=1e-12)
                np.testing.assert_allclose(r,(np.arange(grid[1])+.5)*p.radius_nm/grid[1],rtol=0,atol=1e-12)
                xx,rr=np.meshgrid(x,r,indexing='ij');target=fields(xx,rr,p)
                err=dict(max_phi_error_VT=float(abs(y[0]-target['psi']).max()),
                    max_plus_relative_error=float(abs(np.expm1(y[1]-target['plus']['logq'])).max()),
                    max_minus_relative_error=float(abs(np.expm1(y[2]-target['minus']['logq'])).max()),
                    mid_current_relative_error=float(abs(saved['mid_current_A']-exact)/max(abs(exact),1e-15)))
                recomputed.append(err)
                snapshots.append(Snapshot(saved['problem_id'],saved['model'],x,r,y,saved['mid_current_A'],saved['status'],saved['construction_and_solve_seconds']))
                all_solve_pass+=saved['status']=='PASS'
            expected_pass=[]
            for original,independent,snapshot in zip(record['errors'],recomputed,snapshots):
                for key,value in independent.items():
                    delta=abs(original[key]-value);max_scoring=max(max_scoring,delta)
                    assert delta<1e-10,(family,record['case']['name'],key,delta)
                expected_pass.append(snapshot.status=='PASS' and all(independent[k]<=v for k,v in protocol['exact_evaluation_tolerances'].items()))
            assert expected_pass==record['exact_accuracy_passes']
            decisions=[]
            for published in record['policies']:
                computed=select(published['policy'],snapshots,protocol['exact_evaluation_tolerances'],
                    protocol['controller_assumed_order'],protocol['controller_safety_factor'],protocol['controller_consecutive_pairs'])
                for key in ['selected_index','selected_grid','visited_indices','attempted_cells','attempted_solve_seconds']:
                    assert computed[key]==published[key],(family,config['name'],published['policy'],key)
                index=computed['selected_index'];correct=None if index is None else expected_pass[index]
                assert correct==published['exact_accuracy_pass']
                if index is not None and published['policy']!='uniform_finest':
                    prefix=select(published['policy'],snapshots[:index+1],protocol['exact_evaluation_tolerances'],
                        protocol['controller_assumed_order'],protocol['controller_safety_factor'],protocol['controller_consecutive_pairs'])
                    assert prefix['selected_index']==index and prefix['attempted_cells']==computed['attempted_cells']
                if record['split']=='held_out':
                    agg=actual_aggregate[published['policy']]
                    agg['accurate']+=correct is True;agg['inaccurate']+=correct is False;agg['unresolved']+=correct is None
                    agg['cells']+=computed['attempted_cells'];agg['seconds']+=computed['attempted_solve_seconds']
            if record['split']=='held_out':tuple_set.add(json.dumps({k:v for k,v in config.items() if k!='name'},sort_keys=True))
            case_audits.append(dict(case=config['name'],split=record['split'],independent_exact_pass=expected_pass))
        for original in summary['held_out_policy_aggregate']:
            a=actual_aggregate[original['policy']]
            assert [a['accurate'],a['inaccurate'],a['unresolved'],a['cells']]==[
                original['actual_accurate_accepts'],original['inaccurate_accepts'],original['unresolved'],original['total_attempted_cells']]
            assert abs(a['seconds']-original['total_attempted_solve_seconds'])<1e-10
        sets.append(tuple_set)
        audit.append(dict(family=family,summary_sha256=sha(path),declared_utc=protocol['declared_utc'],created_utc=header['created_utc'],
            aggregates=actual_aggregate,cases=case_audits))
    assert sets[0].isdisjoint(sets[1])
    output=dict(status='PASS_STORED_SCORING_POLICY_AND_COST_REPLAY_AUDIT',campaigns=audit,
        holdout_parameter_tuples_disjoint=True,stored_solves_pass=all_solve_pass,max_independent_scoring_discrepancy=max_scoring,
        scoring_reference='Disjoint continuous MMS module and Gauss96 current; no ManufacturedPNP exact_errors call',
        source_sha256=sha(Path(__file__)),PDE_solves=0,canonical_writes=False)
    (HERE/'audit_results.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='campaigns'},indent=2))


if __name__=='__main__':main()
