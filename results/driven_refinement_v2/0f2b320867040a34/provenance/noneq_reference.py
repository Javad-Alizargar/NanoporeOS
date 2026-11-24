"""Bounded independent reference cache audit/resumption, representative cases.

The disjoint reference assembly is unchanged from the executed worker source.
This driver validates config/source/field hashes and does not lift qualification.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks/independent')]
from nanoporeos.core import check_stop,StopRequested,write_json,sha256
from nodal_pnp import NodalPNP,Problem,save_result


def load(directory,problem):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if set(manifest)!={'fields.npz','result.json'}:raise ValueError('Reference manifest schema')
    for name,digest in manifest.items():
        if sha256(directory/name)!=digest:raise ValueError('Reference field/result hash mismatch')
    metadata=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if metadata['source_sha256']!=sha256(ROOT/'benchmarks/independent/nodal_pnp.py') or metadata['config']!=asdict(problem):
        raise ValueError('Reference source/config identity mismatch')
    with np.load(directory/'fields.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    for name,expected in [('x_nm',np.linspace(0,problem.length_nm,problem.nx+1)),('r_nm',np.linspace(0,problem.radius_nm,problem.nr+1))]:
        if arrays[name].shape!=expected.shape or not np.allclose(arrays[name],expected,atol=1e-12,rtol=1e-12):raise ValueError('Reference coordinate schema')
    for name in ['potential_V','cplus_mol_m3','cminus_mol_m3']:
        field=arrays[name]
        if field.shape!=(problem.nx+1,problem.nr+1) or not np.all(np.isfinite(field)):raise ValueError('Reference field schema')
    if min(arrays['cplus_mol_m3'].min(),arrays['cminus_mol_m3'].min())<=0:raise ValueError('Nonpositive reference concentration')
    return metadata,arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--minutes',type=float,default=20)
    p.add_argument('--output',type=Path,default=ROOT/'results/noneq_reference')
    args=p.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+max(0,min(args.minutes,120))*60;records=[]
    try:
        for voltage in [100.,-100.]:
            for nx,nr in [(32,16),(64,32),(128,64),(256,128)]:
                check_stop(deadline)
                problem=Problem(nx=nx,nr=nr,voltage_mV=voltage)
                directory=output/f'reference_v{voltage:+.0f}_{nx}x{nr}'
                if (directory/'manifest.json').exists():metadata,_=load(directory,problem)
                elif args.check:raise ValueError('Reference case absent')
                else:
                    if directory.exists():raise ValueError('Partial reference directory preserved; choose a new output path')
                    model=NodalPNP(problem);result=model.solve(seconds=max(.1,min(600,deadline-time.monotonic())))
                    save_result(directory,model,result);metadata,_=load(directory,problem)
                records.append(dict(config=asdict(problem),status=metadata['status'],audit=metadata['audit'],
                    path=str(directory),source_sha256=metadata['source_sha256']))
                write_json(output/'driver_state.json',dict(status='RUNNING',records=records,training_qualified=False))
        state='CHECK_PASS' if args.check else 'REPRESENTATIVE_REFERENCE_FINISHED_HOLD_REMAINS'
    except (StopRequested,RuntimeError,ValueError,OSError) as exc:state='CHECKPOINTED: '+str(exc)
    write_json(output/'driver_state.json',dict(status=state,records=records,training_qualified=False,
        scope='R10nm,100mM,offset-15mV,V+/-100mV; not full domain; rim reference uncertainty unresolved',
        source_sha256=sha256(ROOT/'benchmarks/independent/nodal_pnp.py'),driver_sha256=sha256(Path(__file__)),
        resume_command=r'.\.venv\Scripts\python.exe scripts\noneq_reference.py --minutes 20'))
    print(json.dumps(dict(status=state,cases=len(records),training_qualified=False),indent=2))
    return 0 if len(records)==8 and all(r['status']=='PASS' for r in records) else 2


if __name__=='__main__':raise SystemExit(main())
