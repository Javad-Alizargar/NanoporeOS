"""Checkpointed, source-bound legacy/candidate matrix. One process, one case at a time."""
from pathlib import Path
import argparse
import dataclasses
import importlib.util
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import PNP,Scenario,check_stop,StopRequested,source_identity,sha256,write_json,ohmic_current
from nanoporeos.datasets import save_result,load_result


def legacy(s):
    spec=importlib.util.spec_from_file_location('legacy',ROOT/'benchmarks/legacy_kernel.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    model=m.AxisymPNP(s.nx,s.nr,Lnm=s.length_nm,Rnm=s.radius_nm,c_mM=s.salt_mM,
        V_mV=s.voltage_mV,zeta_mV=s.wall_offset_mV)
    start=time.perf_counter();out=model.solve()
    # Preserve original output and assess its actual final coupled equations.
    cp,cm=out['cp'],out['cm']
    audit=None
    if np.all(np.isfinite(np.r_[cp,cm,out['phi']])) and min(cp.min(),cm.min())>0:
        p=PNP(s);y=np.r_[out['phi'],np.log(cp/s.salt_mM),np.log(cm/s.salt_mM)]
        audit=p.audit(y)
    notebook=ROOT/'handoff/legacy/notebooks/Nanopore_ResearchOS_Stage12_16_2D_PNP_MegaRun.ipynb'
    return dict(scenario=dataclasses.asdict(s),reported_converged=out['success'],
        iterations=out['iterations'],update_error=out['error'],history=out['trace'],
        diagnostics=audit,seconds=time.perf_counter()-start,legacy_kernel_sha256=sha256(ROOT/'benchmarks/legacy_kernel.py'),
        supplied_notebook_sha256=sha256(notebook) if notebook.exists() else 'f115c53261fda1f9434f6c20b7b72874b7e5bf04874f2bc0ded191731764ee01',
        supplied_notebook_availability='LOCAL_HASH_CHECKED' if notebook.exists() else 'EXTERNAL_NOT_INCLUDED_EXPECTED_HASH_ONLY',
        historical_executed_source_identity='UNRESOLVED')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','matrix','stress'],default='smoke')
    parser.add_argument('--minutes',type=float,default=20);parser.add_argument('--legacy',action='store_true')
    args=parser.parse_args();deadline=time.monotonic()+min(args.minutes,120)*60
    if args.mode=='matrix':
        scenarios=[Scenario(nx=n,nr=m,radius_nm=r,salt_mM=c,voltage_mV=v,wall_offset_mV=w)
            for r in (5.,10.) for c in (10.,100.) for v in (-100.,0.,100.) for w in (-15.,0.,15.)
            for n,m in [(16,8),(24,12),(36,18)]]
    elif args.mode=='stress':
        protocol=json.loads((ROOT/'config/verification_protocol.json').read_text())
        scenarios=[Scenario(nx=n,nr=m,**s) for s in protocol['stress_cases'] for n,m in [(16,8),(24,12),(36,18)]]
    else:scenarios=[Scenario(nx=16,nr=8,wall_offset_mV=w) for w in [0,-15]]
    output=ROOT/'results'/args.mode
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    started=time.monotonic()
    try:
        for i,s in enumerate(scenarios):
            check_stop(deadline)
            target=output/s.identity()
            if (target/'manifest.json').exists():
                summary,_=load_result(target)
            else:
                p=PNP(s);result=p.solve(deadline=deadline);summary=save_result(target,p,result)
            baseline_path=target/'legacy.json'
            if args.legacy:
                if not baseline_path.exists():write_json(baseline_path,legacy(s))
                baseline=json.loads(baseline_path.read_text())
            else:baseline=None
            records.append(dict(case_id=s.identity(),scenario=dataclasses.asdict(s),status=summary['status'],
                diagnostics=summary['diagnostics'],seconds=summary['seconds'],
                legacy_reported_converged=baseline['reported_converged'] if baseline else None,
                legacy_diagnostics=baseline['diagnostics'] if baseline else None,
                legacy_seconds=baseline['seconds'] if baseline else None))
            write_json(output/'campaign.json',dict(status='RUNNING',source=source_identity(),completed=len(records),
                target=len(scenarios),records=records,protocol_sha256=sha256(ROOT/'config/verification_protocol.json')))
            print(f'{i+1}/{len(scenarios)} {summary["status"]} legacy={records[-1]["legacy_reported_converged"]}',flush=True)
        state='FINISHED'
    except StopRequested as e:
        state=str(e)
    write_json(output/'campaign.json',dict(status=state,source=source_identity(),completed=len(records),
        target=len(scenarios),records=records,protocol_sha256=sha256(ROOT/'config/verification_protocol.json'),
        campaign_seconds=time.monotonic()-started))


if __name__=='__main__':main()
