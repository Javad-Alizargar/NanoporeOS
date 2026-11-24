"""Bounded directional independent refinement, preserving the legacy boundary."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,sys,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks/independent'),str(ROOT/'scripts')]
from nanoporeos.core import sha256,write_json,check_stop,StopRequested,environment,VT
from nodal_pnp_ordered import Problem,NodalPNP,save_result
def load(directory,problem):
    manifest=json.loads((directory/'manifest.json').read_text())
    assert set(manifest)=={'fields.npz','result.json'}
    assert all(sha256(directory/n)==h for n,h in manifest.items())
    metadata=json.loads((directory/'result.json').read_text())
    from dataclasses import asdict
    assert metadata['config']==asdict(problem) and metadata['source_sha256']==sha256(ROOT/'benchmarks/independent/nodal_pnp_ordered.py')
    with np.load(directory/'fields.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    assert np.array_equal(arrays['x_nm'],np.linspace(0,problem.length_nm,problem.nx+1))
    assert np.array_equal(arrays['r_nm'],np.linspace(0,problem.radius_nm,problem.nr+1))
    for name in ['potential_V','cplus_mol_m3','cminus_mol_m3']:
        assert arrays[name].shape==(problem.nx+1,problem.nr+1) and np.isfinite(arrays[name]).all()
    assert min(arrays['cplus_mol_m3'].min(),arrays['cminus_mol_m3'].min())>0
    return metadata,arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--minutes',type=float,default=25);p.add_argument('--check',action='store_true');args=p.parse_args()
    deadline=time.monotonic()+min(25,max(0,args.minutes))*60
    protocol_path=ROOT/'config/driven_refinement_ordered_protocol.json';protocol=json.loads(protocol_path.read_text())
    sources={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),protocol_path,ROOT/'benchmarks/independent/nodal_pnp_ordered.py']}
    digest=hashlib.sha256(json.dumps(dict(sources=sources,environment=environment()),sort_keys=True).encode()).hexdigest()
    directory=ROOT/'results/driven_refinement_ordered'/digest[:16];directory.mkdir(parents=True,exist_ok=True)
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
    write_json(directory/'summary.json',payload);write_json(ROOT/'results/driven_refinement_ordered/latest.json',
        dict(summary=str((directory/'summary.json').relative_to(ROOT)),sha256=sha256(directory/'summary.json')))
    print(json.dumps(payload,indent=2));return 0 if status=='FINISHED' else 2

if __name__=='__main__':raise SystemExit(main())
