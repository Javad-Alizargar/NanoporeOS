"""Comparison layer only: independent reference module does not import candidate."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import sys
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from nodal_pnp import NodalPNP,Problem,save_result,THERMAL_VOLTAGE

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import PNP,Scenario,source_identity


def sample_reference(model,out,x,r):
    xx,rr=np.meshgrid(x,r,indexing='ij');points=np.c_[xx.ravel(),rr.ravel()]
    return [RegularGridInterpolator((model.x,model.r),out[name])(points).reshape(xx.shape)
        for name in ['potential_V','cplus_mol_m3','cminus_mol_m3']]


def difference(first,second,weight):
    result=[]
    for label,a,b in zip(['potential_V','cplus_mol_m3','cminus_mol_m3'],first,second):
        delta=a-b
        result.append(dict(field=label,max_absolute=float(abs(delta).max()),
            relative_volume_L2=float(np.sqrt(np.sum(weight*delta**2)/max(1e-30,np.sum(weight*b**2))))))
    return result


def main():
    started=time.monotonic();records=[]
    # All numerical outputs remain inside this isolated worker directory.
    for voltage in [100.,-100.]:
        references=[]
        for nx,nr in [(32,16),(64,32),(128,64),(256,128)]:
            model=NodalPNP(Problem(nx=nx,nr=nr,voltage_mV=voltage))
            directory=HERE/f'reference_v{voltage:+.0f}_{nx}x{nr}'
            if (directory/'result.json').exists():
                out=json.loads((directory/'result.json').read_text())
                with np.load(directory/'fields.npz',allow_pickle=False) as z:
                    for name in ['potential_V','cplus_mol_m3','cminus_mol_m3']:out[name]=z[name]
            else:
                out=model.solve(seconds=600);save_result(directory,model,out)
            print('reference',voltage,nx,nr,out['status'],out['audit']['current_A'],out['seconds'],flush=True)
            references.append((model,out))
        # Fixed common interior evaluation grid, no extrapolation and volume weights.
        diagnostic=PNP(Scenario(nx=36,nr=18,voltage_mV=voltage))
        weight=diagnostic.volume.reshape(36,18)
        reference_refinements=[]
        for (a,ao),(b,bo) in zip(references[:-1],references[1:]):
            aa=sample_reference(a,ao,diagnostic.x,diagnostic.r)
            bb=sample_reference(b,bo,diagnostic.x,diagnostic.r)
            reference_refinements.append(dict(coarse=[a.problem.nx,a.problem.nr],fine=[b.problem.nx,b.problem.nr],
                fixed_evaluation_grid='36x18 canonical cell centres',fields=difference(aa,bb,weight),
                current_relative_difference=float(abs(ao['audit']['current_A']-bo['audit']['current_A'])/abs(bo['audit']['current_A']))))
        comparisons=[]
        for nx,nr in [(16,8),(24,12),(36,18),(72,36),(144,72)]:
            canonical=PNP(Scenario(nx=nx,nr=nr,voltage_mV=voltage))
            output=canonical.solve(deadline=time.monotonic()+300)
            directory=HERE/f'candidate_v{voltage:+.0f}_{nx}x{nr}';directory.mkdir(exist_ok=True)
            psi,up,um=output['y'].reshape(3,nx,nr)
            candidate=[psi*THERMAL_VOLTAGE,canonical.s.salt_mM*np.exp(up),canonical.s.salt_mM*np.exp(um)]
            np.savez_compressed(directory/'fields.npz',x_nm=canonical.x,r_nm=canonical.r,
                potential_V=candidate[0],cplus_mol_m3=candidate[1],cminus_mol_m3=candidate[2])
            metadata={k:v for k,v in output.items() if k!='y'}
            metadata['source_sha256']=source_identity()
            (directory/'result.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
            reference=sample_reference(*references[-1],canonical.x,canonical.r)
            fields=difference(candidate,reference,canonical.volume.reshape(nx,nr))
            record=dict(candidate_grid=[nx,nr],reference_grid=[256,128],status=output['status'],fields=fields,
                current_relative_error=float(abs(output['diagnostics']['current_A']-references[-1][1]['audit']['current_A'])/abs(references[-1][1]['audit']['current_A'])),
                max_phi_difference_VT=fields[0]['max_absolute']/THERMAL_VOLTAGE,
                reference_remaining_spatial_error='NOT_BOUNDED; see independently refined reference comparisons')
            comparisons.append(record)
            print('comparison',voltage,nx,nr,record['max_phi_difference_VT'],record['current_relative_error'],flush=True)
        records.append(dict(voltage_mV=voltage,reference_refinements=reference_refinements,comparisons=comparisons,
            reference_configs=[asdict(m.problem) for m,o in references]))
        (HERE/'comparison.json').write_text(json.dumps(dict(status='RUNNING',records=records),indent=2),encoding='utf-8')
    payload=dict(status='FINISHED',records=records,seconds=time.monotonic()-started,
        scope='Representative hard family only, no full-domain qualification',
        reference_source_sha256=hashlib.sha256((HERE/'nodal_pnp.py').read_bytes()).hexdigest(),
        comparison_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_source_sha256=source_identity(),
        independence='Nodal electrochemical-potential diffusion with geometric face conductivity; no SG flux/canonical assembly import in reference module',
        limitations=['Rim Dirichlet jumps preserved','Different discretizations converge to matching continuum model only under refinement',
            'No independent derivative/electric-field accuracy bound','No experimental validation'])
    (HERE/'comparison.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
