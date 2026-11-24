"""Bounded directional independent refinement, preserving the legacy boundary."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,sys,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks/independent'),str(ROOT/'scripts')]
from nanoporeos.core import sha256,write_json,check_stop,StopRequested,environment,VT
from nodal_pnp import Problem,NodalPNP,save_result
from noneq_reference import load

def main():
    p=argparse.ArgumentParser();p.add_argument('--minutes',type=float,default=25);p.add_argument('--check',action='store_true');args=p.parse_args()
    deadline=time.monotonic()+min(25,max(0,args.minutes))*60
    protocol_path=ROOT/'config/driven_refinement_v2_protocol.json';protocol=json.loads(protocol_path.read_text())
    sources={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),protocol_path,ROOT/'benchmarks/independent/nodal_pnp.py',ROOT/'scripts/noneq_reference.py']}
    digest=hashlib.sha256(json.dumps(dict(sources=sources,environment=environment()),sort_keys=True).encode()).hexdigest()
    directory=ROOT/'results/driven_refinement_v2'/digest[:16];directory.mkdir(parents=True,exist_ok=True)
    provenance=directory/'provenance';provenance.mkdir(exist_ok=True)
    for name,value in sources.items():
        source=ROOT/name;dest=provenance/source.name
        if dest.exists():assert sha256(dest)==value
        else:dest.write_bytes(source.read_bytes())
    if not (directory/'header.json').exists():write_json(directory/'header.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,environment=environment()))
    records=[];status='RUNNING'
    try:
        for nx,nr in protocol['new_reference_grids']:
            check_stop(deadline);problem=Problem(nx=nx,nr=nr,**protocol['problem']);output=directory/f'reference_{nx}x{nr}'
            if (output/'manifest.json').exists():metadata,arrays=load(output,problem)
            elif args.check:raise ValueError('Missing reference checkpoint')
            else:
                if output.exists():raise ValueError('Partial output preserved; manual reconciliation required')
                model=NodalPNP(problem);result=model.solve(seconds=max(.1,min(600,deadline-time.monotonic())))
                save_result(output,model,result);metadata,arrays=load(output,problem)
            records.append(dict(grid=[nx,nr],status=metadata['status'],seconds=metadata['seconds'],audit=metadata['audit'],
                path=str(output.relative_to(ROOT)),manifest_sha256=sha256(output/'manifest.json')))
            write_json(directory/'progress.json',dict(status='RUNNING',records=records));print(nx,nr,metadata['status'],metadata['seconds'],flush=True)
        status='FINISHED' if all(r['status']=='PASS' for r in records) else 'FAIL_REFERENCE_DISCRETE_CHECK'
    except (StopRequested,RuntimeError,ValueError) as exc:status='CHECKPOINTED_'+str(exc)
    payload=dict(status=status,records=records,sources=sources,protocol_sha256=sha256(protocol_path),
        directory=str(directory.relative_to(ROOT)),training_qualified=False,original_gates_changed=False)
    write_json(directory/'summary.json',payload);write_json(ROOT/'results/driven_refinement_v2/latest.json',
        dict(summary=str((directory/'summary.json').relative_to(ROOT)),sha256=sha256(directory/'summary.json')))
    print(json.dumps(payload,indent=2));return 0 if status=='FINISHED' else 2

if __name__=='__main__':raise SystemExit(main())
