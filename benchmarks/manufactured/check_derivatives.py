"""Independent finite differences and integral balance of continuous MMS fields.

Only a few analytic evaluations/Gauss integrals: no PDE/training workload.
The divergence checker differentiates the flux, rather than reusing the
closed-form forcing expressions. No canonical discrete operator is imported.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from manufactured_math import Parameters,fields,ALPHA_NM,EPS0,E,KB,NA,VT


def approximation(x,r,p,relative_step):
    hx=relative_step*p.length_nm;hr=relative_step*p.radius_nm
    center=fields(x,r,p)
    xp,xm=fields(x+hx,r,p),fields(x-hx,r,p)
    rp,rm=fields(x,r+hr,p),fields(x,r-hr,p)
    dx2=(xp['psi']-2*center['psi']+xm['psi'])/(hx*hx)
    dr2=(rp['psi']-2*center['psi']+rm['psi'])/(hr*hr)
    dr1=(rp['psi']-rm['psi'])/(2*hr)
    radial=np.where(r==0,2*dr2,dr2+np.divide(dr1,r,out=np.zeros_like(r),where=r!=0))
    lap_psi_fd=dx2+radial
    psi_x_fd=(xp['psi']-xm['psi'])/(2*hx)
    psi_r_fd=dr1
    poisson=-lap_psi_fd-ALPHA_NM*p.bulk_mol_m3*(center['plus']['q']-center['minus']['q'])
    result={'poisson_forcing_nm2':poisson}
    for name in ['plus','minus']:
        axial=(xp[name]['j_x']-xm[name]['j_x'])/(2*hx)
        derivative=(rp[name]['j_r']-rm[name]['j_r'])/(2*hr)
        radial=np.where(r==0,2*derivative,derivative+np.divide(center[name]['j_r'],r,out=np.zeros_like(r),where=r!=0))
        result[name]=axial+radial
        # A second path uses only sampled q and psi values, not analytic mu
        # derivatives or analytic fluxes: -lap q-z*(grad q.grad psi+q lap psi).
        q=center[name]['q'];qx=(xp[name]['q']-xm[name]['q'])/(2*hx)
        qr=(rp[name]['q']-rm[name]['q'])/(2*hr)
        qxx=(xp[name]['q']-2*q+xm[name]['q'])/(hx*hx)
        qrr=(rp[name]['q']-2*q+rm[name]['q'])/(hr*hr)
        radial_q=np.where(r==0,2*qrr,qrr+np.divide(qr,r,out=np.zeros_like(r),where=r!=0))
        z=1 if name=='plus' else -1
        result[name+'_NP_values_only']=-(qxx+radial_q)-z*(qx*psi_x_fd+qr*psi_r_fd+q*lap_psi_fd)
    return result


def errors(approx,exact):
    return dict(max_absolute=float(np.max(np.abs(approx-exact))),
        relative_L2=float(np.linalg.norm(approx-exact)/np.linalg.norm(exact)))


def integral_checks(p,n=64,axial_fraction=1.):
    node,weight=leggauss(n)
    upper=axial_fraction*p.length_nm
    x=(node+1)*upper/2;wx=weight*upper/2
    r=(node+1)*p.radius_nm/2;wr=weight*p.radius_nm/2
    values=fields(x[:,None],r[None,:],p)
    volume_weights=wx[:,None]*wr[None,:]*2*np.pi*r[None,:]
    result={}
    for name in ['plus','minus']:
        source=float(np.sum(values[name]['forcing_nm2']*volume_weights))
        left=fields(np.zeros_like(r),r,p)[name]['j_x']
        right=fields(np.full_like(r,upper),r,p)[name]['j_x']
        boundary=float(np.sum((right-left)*2*np.pi*r*wr))
        wall=fields(x,np.full_like(x,p.radius_nm),p)[name]['j_r']
        boundary+=float(np.sum(wall*2*np.pi*p.radius_nm*wx))
        result[name]=dict(integrated_source_nm=source,boundary_outward_flux_nm=boundary,
            absolute_imbalance_nm=abs(source-boundary))
    # Poisson operator identity: integrated(-lap psi)=-outward gradient flux.
    poisson_integral=float(np.sum((values['poisson_forcing_nm2']+ALPHA_NM*p.bulk_mol_m3*(values['plus']['q']-values['minus']['q']))*volume_weights))
    left=fields(np.zeros_like(r),r,p)['psi_x'];right=fields(np.full_like(r,upper),r,p)['psi_x']
    wall=fields(x,np.full_like(x,p.radius_nm),p)['psi_r']
    boundary=-float(np.sum((right-left)*2*np.pi*r*wr)+np.sum(wall*2*np.pi*p.radius_nm*wx))
    result['poisson']=dict(integrated_negative_laplacian_nm=poisson_integral,negative_outward_gradient_flux_nm=boundary,
        absolute_imbalance_nm=abs(poisson_integral-boundary))
    return result


def main():
    p=Parameters()
    x=p.length_nm*np.array([.13,.27,.49,.68,.83,.37,.61,.22])
    r=p.radius_nm*np.array([.17,.39,.61,.76,.43,0.,0.,.91])
    exact=fields(x,r,p);records=[]
    for h in [.002,.001,.0005]:
        a=approximation(x,r,p,h)
        record=dict(relative_step=h,poisson=errors(a['poisson_forcing_nm2'],exact['poisson_forcing_nm2']))
        for name in ['plus','minus']:
            record[name]=errors(a[name],exact[name]['forcing_nm2'])
            record[name+'_NP_values_only']=errors(a[name+'_NP_values_only'],exact[name]['forcing_nm2'])
        records.append(record)
    for name in ['poisson','plus','minus','plus_NP_values_only','minus_NP_values_only']:
        ratios=[records[i][name]['relative_L2']/records[i+1][name]['relative_L2'] for i in [0,1]]
        if min(ratios)<3.5 or max(ratios)>4.5:raise AssertionError(('Expected independent central-difference order',name,ratios))
        if records[-1][name]['relative_L2']>1e-5:raise AssertionError((name,'derivative check accuracy'))
    xx=np.linspace(0,p.length_nm,65);rr=np.linspace(0,p.radius_nm,41)
    wall=fields(xx,np.full_like(xx,p.radius_nm),p)
    axis=fields(xx,np.zeros_like(xx),p)
    left=fields(np.zeros_like(rr),rr,p);right=fields(np.full_like(rr,p.length_nm),rr,p)
    checks=dict(wall_potential_trace_error=float(np.max(abs(wall['psi']-wall['wall_psi']))),
        left_potential_error=float(np.max(abs(left['psi']))),
        right_potential_error=float(np.max(abs(right['psi']-p.applied_thermal_voltages))),
        axis_radial_potential_derivative=float(np.max(abs(axis['psi_r']))))
    for name in ['plus','minus']:
        checks[name]=dict(wall_ionic_normal_flux=float(np.max(abs(wall[name]['j_r']))),
            axis_ionic_normal_flux=float(np.max(abs(axis[name]['j_r']))),
            left_density_error=float(np.max(abs(left[name]['q']-1))),right_density_error=float(np.max(abs(right[name]['q']-1))))
        if max(checks[name].values())>1e-12:raise AssertionError(('Boundary trace',name,checks[name]))
    if max(v for v in checks.values() if not isinstance(v,dict))>1e-12:raise AssertionError('Potential traces')
    balance={str(frac):integral_checks(p,axial_fraction=frac) for frac in [1.,.37]}
    if max(v['absolute_imbalance_nm'] for domain in balance.values() for v in domain.values())>1e-10:raise AssertionError(('Divergence theorem',balance))
    if min(abs(balance['0.37'][name]['integrated_source_nm']) for name in ['plus','minus'])<1e-3:
        raise AssertionError('Need nonzero partial-domain source balance to detect a sign mistake')
    directory=Path(__file__).resolve().parent
    payload=dict(status='PASS_INDEPENDENT_CONTINUOUS_DERIVATIVE_AND_BOUNDARY_CHECKS',parameters=asdict(p),
        finite_difference_records=records,boundary_checks=checks,integral_balance_checks=balance,
        physical_constants=dict(e_C=E,kB_J_K=KB,NA_per_mol=NA,epsilon0_F_m=EPS0,VT_V=VT,alpha_nm2_per_mol_m3=ALPHA_NM),
        proposed_species_sign='div(-q*grad(mu))=f_species; subtract volume*f from the raw outward-flux residual',
        proposed_poisson_sign='-lap(psi)-alpha*cb*(qplus-qminus)=f_poisson; subtract volume*f from raw Poisson residual',
        distinction='These are continuous forced-equation verification checks; no discrete PNP assembly or PDE solve was used.',
        python=sys.version,numpy=np.__version__,
        sources={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ['manufactured_math.py','check_derivatives.py']})
    output_directory=directory.parents[1]/'results/verification_v3/continuum_check'
    output_directory.mkdir(parents=True,exist_ok=True)
    (output_directory/'checks.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
