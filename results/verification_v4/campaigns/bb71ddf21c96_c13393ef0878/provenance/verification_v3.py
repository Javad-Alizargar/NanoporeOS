"""Bounded source-bound MMS campaign and frozen accuracy-policy evaluation.

No paid APIs, new external data, production datasets or legacy model changes.
Each completed case is checkpointed. A new invocation requires user authorization.
"""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import write_json,sha256,environment,check_stop,StopRequested
from nanoporeos.manufactured import ManufacturedCase,ManufacturedPNP,implementation_identity
from nanoporeos.accuracy import Snapshot,select

def problem_id(case):return hashlib.sha256(json.dumps(asdict(case),sort_keys=True).encode()).hexdigest()

def one(case,grid,directory,deadline,protocol_hash):
    check_stop(deadline);started=time.perf_counter();model=ManufacturedPNP(case,*grid)
    identity=model.identity();output=directory/'cases'/identity;output.mkdir(parents=True,exist_ok=True)
    result_path=output/'result.json';array_path=output/'fields.npz';manifest_path=output/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['protocol_sha256']!=protocol_hash:raise RuntimeError('Protocol/cache mismatch')
        assert all(sha256(output/name)==digest for name,digest in manifest['files'].items())
        saved=json.loads(result_path.read_text(encoding='utf-8'))
        with np.load(array_path,allow_pickle=False) as archive:y=archive['y'];x=archive['x_nm'];r=archive['r_nm']
        assert y.shape==(3,*grid) and np.isfinite(y).all()
    else:
        solved=model.solve(deadline=deadline);seconds=time.perf_counter()-started
        y=solved.pop('y').reshape(3,*grid);x=model.x;r=model.r
        saved=dict(**solved,construction_and_solve_seconds=seconds,protocol_sha256=protocol_hash,
            problem_id=problem_id(case),mid_current_A=model.mid_current(solved))
        temporary=array_path.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,y=y,x_nm=x,r_nm=r)
        temporary.replace(array_path);write_json(result_path,saved)
        write_json(manifest_path,dict(protocol_sha256=protocol_hash,identity=identity,
            files={p.name:sha256(p) for p in [result_path,array_path]}))
    return Snapshot(problem_id(case),saved['model'],x,r,y,saved['mid_current_A'],saved['status'],saved['construction_and_solve_seconds']),saved,model

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--minutes',type=float,default=25)
    parser.add_argument('--protocol',default='verification_v3_protocol.json')
    parser.add_argument('--development-only',action='store_true');args=parser.parse_args()
    deadline=time.monotonic()+min(25,max(0,args.minutes))*60
    protocol_path=(ROOT/'config'/args.protocol).resolve()
    if protocol_path.parent!=ROOT/'config':raise ValueError('Protocol must be a local config file')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol_hash=sha256(protocol_path)
    sources=dict(**implementation_identity(),controller_sha256=sha256(ROOT/'src/nanoporeos/accuracy.py'),
        driver_sha256=sha256(Path(__file__)))
    source_hash=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    family=protocol.get('output_family','verification_v3')
    if family not in ['verification_v3','verification_v4']:raise ValueError('Unknown verification family')
    directory=ROOT/'results'/family/'campaigns'/(protocol_hash[:12]+'_'+source_hash[:12]);directory.mkdir(parents=True,exist_ok=True)
    header=dict(protocol_sha256=protocol_hash,source_hashes=sources,environment=environment(),
        created_utc=datetime.now(timezone.utc).isoformat(),data_mode='local_only',paid_calls=0)
    if not (directory/'header.json').exists():write_json(directory/'header.json',header)
    provenance=directory/'provenance';provenance.mkdir(exist_ok=True)
    for source in [Path(__file__),ROOT/'src/nanoporeos/manufactured.py',ROOT/'src/nanoporeos/accuracy.py',protocol_path]:
        snapshot=provenance/source.name
        if snapshot.exists() and sha256(snapshot)!=sha256(source):raise RuntimeError('Source snapshot conflict')
        if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    records=[];status='RUNNING'
    groups=[('development',protocol['development_cases'])]
    if not args.development_only:groups.append(('held_out',protocol['held_out_cases']))
    try:
        for split,cases in groups:
            for parameters in cases:
                case=ManufacturedCase(**parameters);snapshots=[];solves=[];models=[]
                for grid in protocol['mesh_ladder']:
                    check_stop(deadline);snapshot,solve,model=one(case,grid,directory,deadline,protocol_hash)
                    snapshots.append(snapshot);solves.append(solve);models.append(model)
                    print(case.name,grid,snapshot.status,flush=True)
                # Freeze numerical decisions before invoking exact error scoring.
                decisions=[]
                for name in protocol['policies']:
                    started=time.perf_counter()
                    decision=select(name,snapshots,protocol['exact_evaluation_tolerances'],protocol['controller_assumed_order'],
                        protocol['controller_safety_factor'],protocol['controller_consecutive_pairs'])
                    decision['selection_seconds']=time.perf_counter()-started
                    decision['solve_plus_selection_seconds']=decision['attempted_solve_seconds']+decision['selection_seconds']
                    decisions.append(decision)
                decision_seconds=sum(d['selection_seconds'] for d in decisions)
                decision_path=directory/(case.name+'_decisions.json')
                write_json(decision_path,dict(case=asdict(case),policies=decisions,decision_seconds=decision_seconds,
                    decision_inputs='Numerical snapshots, budgets and fixed policy parameters only'))
                errors=[model.exact_errors(s.y.ravel(),s.current_A) for model,s in zip(models,snapshots)]
                exact_pass=[s.status=='PASS' and all(e[k]<=v for k,v in protocol['exact_evaluation_tolerances'].items())
                    for s,e in zip(snapshots,errors)]
                orders={key:float(np.log(errors[-2][key]/errors[-1][key])/np.log(2))
                    for key in protocol['order_verification']['observables'] if errors[-1][key]>0 and errors[-2][key]>0}
                for decision in decisions:
                    selected=decision['selected_index']
                    decision['exact_accuracy_pass']=None if selected is None else bool(exact_pass[selected])
                    decision['actual_errors']=None if selected is None else errors[selected]
                record=dict(case=asdict(case),split=split,grids=protocol['mesh_ladder'],errors=errors,
                    exact_accuracy_passes=exact_pass,observed_last_pair_orders=orders,policies=decisions,
                    decisions_before_scoring_sha256=sha256(decision_path),decision_seconds=decision_seconds,
                    solve_identities=[s['identity'] for s in solves])
                records.append(record)
                write_json(directory/'progress.json',dict(status='RUNNING',records=records,protocol_sha256=protocol_hash))
            if split=='development':
                low,high=protocol['order_verification']['last_pair_order_interval']
                if not all(len(r['observed_last_pair_orders'])==3 and all(low<=p<=high for p in r['observed_last_pair_orders'].values()) for r in records):
                    status='DEVELOPMENT_ORDER_GATE_FAIL_HOLDOUT_NOT_RUN';break
        else:status='FINISHED_DEVELOPMENT' if args.development_only else 'FINISHED_FIXED_HOLDOUT'
    except StopRequested as exc:status='CHECKPOINTED_STOP_OR_DEADLINE';print(str(exc),flush=True)
    heldout=[r for r in records if r['split']=='held_out'];aggregate=[]
    for policy in protocol['policies']:
        decisions=[next(p for p in r['policies'] if p['policy']==policy) for r in heldout]
        aggregate.append(dict(policy=policy,scenarios=len(decisions),estimated_accepts=sum(d['selected_index'] is not None for d in decisions),
            actual_accurate_accepts=sum(d['exact_accuracy_pass'] is True for d in decisions),
            inaccurate_accepts=sum(d['exact_accuracy_pass'] is False for d in decisions),unresolved=sum(d['selected_index'] is None for d in decisions),
            total_attempted_solve_seconds=sum(d['attempted_solve_seconds'] for d in decisions),
            total_selection_seconds=sum(d['selection_seconds'] for d in decisions),
            total_solve_plus_selection_seconds=sum(d['solve_plus_selection_seconds'] for d in decisions),
            total_attempted_cells=sum(d['attempted_cells'] for d in decisions)))
    payload=dict(status=status,protocol_sha256=protocol_hash,source_hashes=sources,records=records,
        held_out_policy_aggregate=aggregate,campaign_directory=str(directory.relative_to(ROOT)),
        scientific_scope='Forced analytic verification only; legacy charged-wall domain HOLD unchanged',
        cost_interpretation=protocol['cost_estimand'],no_physical_training_qualification=True)
    write_json(directory/'summary.json',payload);write_json(ROOT/'results'/family/'latest.json',
        dict(summary=str((directory/'summary.json').relative_to(ROOT)),sha256=sha256(directory/'summary.json'),status=status))
    print(json.dumps(dict(status=status,campaign=str(directory),policies=aggregate),indent=2))
    return 0 if status.startswith('FINISHED') else 2

if __name__=='__main__':raise SystemExit(main())
