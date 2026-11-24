"""Saved-array audits and provenance, without changing executed solver sources."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import platform
import sys
import numpy as np
import scipy
from scipy.constants import epsilon_0
from scipy.interpolate import RegularGridInterpolator
from nodal_pnp import NodalPNP,Problem,THERMAL_VOLTAGE

HERE=Path(__file__).resolve().parent


def load(voltage,nx,nr):
    directory=HERE/f'reference_v{voltage:+.0f}_{nx}x{nr}'
    metadata=json.loads((directory/'result.json').read_text())
    with np.load(directory/'fields.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    return metadata,arrays


def main():
    comparison=json.loads((HERE/'comparison.json').read_text())
    if comparison['status']!='FINISHED':raise RuntimeError('Incomplete comparison')
    audits=[]
    for voltage in [100.,-100.]:
        for nx,nr in [(32,16),(64,32),(128,64),(256,128)]:
            metadata,arrays=load(voltage,nx,nr)
            model=NodalPNP(Problem(**metadata['config']))
            phi=arrays['potential_V'].ravel()/THERMAL_VOLTAGE
            plus=np.log(arrays['cplus_mol_m3'].ravel()/model.problem.salt_mM)+phi
            minus=np.log(arrays['cminus_mol_m3'].ravel()/model.problem.salt_mM)-phi
            y=np.r_[phi[model.interior_phi],plus[model.interior_ions],minus[model.interior_ions]]
            scaled=model.equation(y)
            charge_C=scaled[:model.np]*model.scale_p*epsilon_0*78.3*THERMAL_VOLTAGE*1e-9
            audits.append(dict(config=metadata['config'],
                reloaded_scaled_residual_max=float(np.max(np.abs(scaled))),
                max_integrated_poisson_charge_residual_C=float(np.max(np.abs(charge_C))),
                end_concentration_error_mol_m3=float(max(np.max(abs(arrays['cplus_mol_m3'][[0,-1],:]-model.problem.salt_mM)),np.max(abs(arrays['cminus_mol_m3'][[0,-1],:]-model.problem.salt_mM)))),
                min_concentration_mol_m3=metadata['audit']['min_concentration_mol_m3'],
                current_spread_A=metadata['audit']['section_current_spread_A'],
                species_max_global_imbalance_mol_s=max(s['global_imbalance_mol_s'] for s in metadata['audit']['species']),
                species_max_local_imbalance_mol_s=max(s['max_local_imbalance_mol_s'] for s in metadata['audit']['species'])))
    positive,pa=load(100.,256,128);negative,na=load(-100.,256,128)
    symmetry=dict(potential_reflection_error_V=float(np.max(np.abs(na['potential_V']-(pa['potential_V'][::-1,:]-.1)))),
        cplus_reflection_error_mol_m3=float(np.max(abs(na['cplus_mol_m3']-pa['cplus_mol_m3'][::-1,:]))),
        cminus_reflection_error_mol_m3=float(np.max(abs(na['cminus_mol_m3']-pa['cminus_mol_m3'][::-1,:]))),
        current_odd_symmetry_error_A=float(abs(negative['audit']['current_A']+positive['audit']['current_A'])),
        justification='Axial reflection maps +V to -V and phi to reflected phi minus V; same wall offset retained. No charge-conjugation/species exchange is assumed.')
    perturbation=[]
    reference_resolution=[]
    for voltage in [100.,-100.]:
        meta,reference=load(voltage,256,128)
        for nx,nr in [(16,8),(24,12),(36,18),(72,36),(144,72)]:
            path=HERE/f'candidate_v{voltage:+.0f}_{nx}x{nr}'/'fields.npz'
            with np.load(path,allow_pickle=False) as z:candidate={k:z[k] for k in z.files}
            x,r=candidate['x_nm'],candidate['r_nm']
            xx,rr=np.meshgrid(x,r,indexing='ij');points=np.c_[xx.ravel(),rr.ravel()]
            dr=meta['config']['radius_nm']/nr;dx=meta['config']['length_nm']/nx
            weight=np.pi*((r+dr/2)**2-(r-dr/2)**2)[None,:]*dx
            metrics=[]
            _,coarse_reference=load(voltage,128,64)
            coarse_phi=RegularGridInterpolator((coarse_reference['x_nm'],coarse_reference['r_nm']),coarse_reference['potential_V'])(points).reshape(nx,nr)
            fine_phi=RegularGridInterpolator((reference['x_nm'],reference['r_nm']),reference['potential_V'])(points).reshape(nx,nr)
            reference_resolution.append(dict(voltage_mV=voltage,evaluation_grid=[nx,nr],
                reference_grids=[[128,64],[256,128]],max_refinement_difference_VT=float(np.max(np.abs(coarse_phi-fine_phi))/THERMAL_VOLTAGE),
                note='A refinement difference, not a rigorous fine-reference error bound. Corner interpolation can dominate fine target grids.'))
            for field,background in [('potential_V',voltage*.001*xx/20),('cplus_mol_m3',100),('cminus_mol_m3',100)]:
                ref=RegularGridInterpolator((reference['x_nm'],reference['r_nm']),reference[field])(points).reshape(nx,nr)
                difference=candidate[field]-ref
                perturb=ref-background
                metrics.append(dict(field=field,relative_perturbation_volume_L2=float(np.sqrt(np.sum(weight*difference**2)/np.sum(weight*perturb**2)))))
            perturbation.append(dict(voltage_mV=voltage,candidate_grid=[nx,nr],metrics=metrics))
    payload=dict(status='PASS_ARRAY_RELOAD_AND_AUDIT',reference_audits=audits,reflection_check=symmetry,
        perturbation_field_metrics=perturbation,
        reference_resolution_at_each_candidate_grid=reference_resolution,
        note='Potential background V*x/L and concentration background bulk removed only in these additional diagnostics; original frozen gates are unchanged.',
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'array_audit.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=payload['status'],reflection_check=symmetry),indent=2))


if __name__=='__main__':main()
