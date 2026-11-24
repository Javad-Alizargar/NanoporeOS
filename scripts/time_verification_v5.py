"""Prespecified repeated live policy executions; no cached solve reuse."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,json,sys,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import sha256,write_json,check_stop,StopRequested,environment
from nanoporeos.manufactured_families import FamilyCase,FamilyPNP,implementation_identity
from nanoporeos.accuracy import Snapshot
from nanoporeos.acceptance_matched import select

def main():
    p=argparse.ArgumentParser();p.add_argument('--minutes',type=float,default=20);args=p.parse_args()
    deadline=time.monotonic()+min(25,max(0,args.minutes))*60
    protocol_path=ROOT/'config/verification_v5a_protocol.json';protocol=json.loads(protocol_path.read_text())
    plan=protocol['timing_followup'];cases={r['name']:FamilyCase(**r) for r in protocol['held_out_cases']}
    sources=dict(**implementation_identity(),driver_sha256=sha256(Path(__file__)),
        matched_sha256=sha256(ROOT/'src/nanoporeos/acceptance_matched.py'),accuracy_sha256=sha256(ROOT/'src/nanoporeos/accuracy.py'),
        protocol_sha256=sha256(protocol_path),environment=environment())
    digest=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    directory=ROOT/'results/verification_v5_timing'/digest[:16];directory.mkdir(parents=True,exist_ok=True)
    provenance=directory/'provenance';provenance.mkdir(exist_ok=True)
    for source in [Path(__file__),protocol_path,ROOT/'src/nanoporeos/manufactured_families.py',
                   ROOT/'src/nanoporeos/manufactured.py',ROOT/'src/nanoporeos/accuracy.py',ROOT/'src/nanoporeos/acceptance_matched.py']:
        output=provenance/source.name
        if output.exists():assert sha256(output)==sha256(source)
        else:output.write_bytes(source.read_bytes())
    order=[];rng=np.random.default_rng(plan['seed'])
    for rep in range(plan['repetitions']):
        pairs=[(name,policy) for name in plan['cases'] for policy in protocol['policies']]
        for i in rng.permutation(len(pairs)):order.append(dict(repetition=rep+1,case=pairs[i][0],policy=pairs[i][1]))
    write_json(directory/'execution_order.json',dict(plan=plan,order=order,sources=sources))
    warm=FamilyPNP(FamilyCase('timing_warmup',family='polynomial'),12,6).solve(deadline=deadline)
    assert warm['status']=='PASS';records=[];status='RUNNING'
    try:
        for index,item in enumerate(order):
            check_stop(deadline);path=directory/f'attempt_{index:03}.json'
            if path.exists():
                record=json.loads(path.read_text());assert record['design']==item and record['source_identity']==digest
                assert sha256(directory/record['decision_file'])==record['decision_before_scoring_sha256']
                if record['selected_fields'] is not None:assert sha256(directory/record['selected_fields'])==record['selected_fields_sha256']
                records.append(record);continue
            case=cases[item['case']];snapshots=[];attempts=[];models=[];started=time.perf_counter()
            grids=protocol['mesh_ladder'][-1:] if item['policy']=='uniform_finest' else protocol['mesh_ladder']
            decision=None
            for grid in grids:
                check_stop(deadline);one_started=time.perf_counter();model=FamilyPNP(case,*grid);result=model.solve(deadline=deadline)
                seconds=time.perf_counter()-one_started;current=model.mid_current(result)
                snapshots.append(Snapshot(case.name,result['model'],model.x,model.r,result['y'].reshape(3,*grid),current,result['status'],seconds))
                models.append(model);attempts.append(dict(grid=grid,status=result['status'],seconds=seconds,diagnostics=result['diagnostics']))
                decision=select(item['policy'],snapshots,protocol['exact_evaluation_tolerances'],2,2,1)
                if decision['selected_index'] is not None:break
            wall=time.perf_counter()-started
            blind_path=directory/f'attempt_{index:03}_decision.json'
            write_json(blind_path,dict(design=item,case=asdict(case),decision=decision,wall_seconds=wall,attempts=attempts))
            selected=decision['selected_index'];errors=None;fields_path=None
            if selected is not None:
                s=snapshots[selected];errors=models[selected].exact_errors(s.y.ravel(),s.current_A)
                fields_path=directory/f'attempt_{index:03}_fields.npz';temporary=fields_path.with_suffix('.tmp')
                with temporary.open('wb') as f:np.savez_compressed(f,y=s.y,x_nm=s.x,r_nm=s.r,current_A=s.current_A)
                temporary.replace(fields_path)
            passed=None if errors is None else all(errors[k]<=v for k,v in protocol['exact_evaluation_tolerances'].items())
            record=dict(design=item,source_identity=digest,wall_seconds=wall,attempted_cells=decision['attempted_cells'],
                selected_grid=decision['selected_grid'],exact_accuracy_pass=passed,errors=errors,
                decision_file=blind_path.name,decision_before_scoring_sha256=sha256(blind_path),
                selected_fields=None if fields_path is None else fields_path.name,
                selected_fields_sha256=None if fields_path is None else sha256(fields_path))
            write_json(path,record);records.append(record)
            write_json(directory/'progress.json',dict(status='RUNNING',records=records))
            print(index+1,len(order),item,wall,flush=True)
        status='FINISHED_RANDOMIZED_REPEATED_POLICY_EXECUTIONS'
    except StopRequested as exc:status='CHECKPOINTED_'+str(exc)
    aggregate=[]
    for policy in protocol['policies']:
        group=[r for r in records if r['design']['policy']==policy]
        if group:aggregate.append(dict(policy=policy,attempts=len(group),unique_cases=len({r['design']['case'] for r in group}),
            accurate=sum(r['exact_accuracy_pass'] is True for r in group),inaccurate=sum(r['exact_accuracy_pass'] is False for r in group),
            unresolved=sum(r['exact_accuracy_pass'] is None for r in group),
            median_wall_seconds=float(np.median([r['wall_seconds'] for r in group])),total_wall_seconds=sum(r['wall_seconds'] for r in group)))
    payload=dict(status=status,records=records,aggregate=aggregate,sources=sources,
        interpretation='Three repetitions of six fixed cases; coverage differs; no equal-accuracy/population speedup claim',
        timing_boundary='Live sequential construction, solve, diagnostics and incremental selection; excludes fileI/O and exact scoring; warmup excluded',
        source_identity=digest,training_qualified=False)
    write_json(directory/'summary.json',payload);write_json(ROOT/'results/verification_v5_timing/latest.json',
        dict(summary=str((directory/'summary.json').relative_to(ROOT)),sha256=sha256(directory/'summary.json')))
    print(json.dumps(dict(status=status,aggregate=aggregate),indent=2));return 0 if status.startswith('FINISHED') else 2

if __name__=='__main__':raise SystemExit(main())
