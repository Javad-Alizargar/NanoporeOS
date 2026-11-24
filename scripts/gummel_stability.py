"""Local linear stability of the exact legacy relaxed Gummel map at roots.

Analytic linearization uses the audited same final discrete equations. This is
an explanatory within-discretization diagnostic, not an independent PDE solver.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu,eigs,LinearOperator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import PNP,Scenario,ALPHA,write_json,source_identity


def main():
    records=[]
    for radius,salt in [(5,10),(5,100),(10,10),(10,100)]:
        s=Scenario(nx=16,nr=8,radius_nm=radius,salt_mM=salt,voltage_mV=0)
        p=PNP(s);o=p.solve();phi,up,um=o['y'].reshape(3,p.n)
        lap=splu(p.L.tocsc());pairs=[]
        for u,z in [(up,1),(um,-1)]:
            _,jp,ju=p.transport(phi,u,z,1,True)
            pairs.append((splu((ju@diags(np.exp(-u))).tocsc()),jp))
        def apply(v):
            dphi=lap.solve(ALPHA*s.salt_mM*p.volume*(v[:p.n]-v[p.n:]))
            response=np.r_[*[a.solve(-j@dphi) for a,j in pairs]]
            return .78*v+.22*response
        operator=LinearOperator((2*p.n,2*p.n),matvec=apply,dtype=float)
        eigenvalues,vectors=eigs(operator,k=3,which='LM',v0=np.random.default_rng(20261109).normal(size=2*p.n),tol=1e-10)
        dominant=int(np.argmax(np.abs(eigenvalues)));vec=vectors[:,dominant].real
        delta=1e-5
        c=np.exp(np.r_[up,um])
        def actual_map(concentrations):
            potential=lap.solve(p.pb+ALPHA*s.salt_mM*p.volume*(concentrations[:p.n]-concentrations[p.n:]))
            new=[]
            for z in [1,-1]:
                # transport is linear in concentration; recover A and reservoir RHS.
                res,jp,ju=p.transport(potential,np.zeros(p.n),z,1,True)
                new.append(splu(ju.tocsc()).solve(ju@np.ones(p.n)-res))
            return .78*concentrations+.22*np.r_[*new]
        finite_difference=(actual_map(c+delta*vec)-actual_map(c-delta*vec))/(2*delta)
        records.append(dict(radius_nm=radius,salt_mM=salt,voltage_mV=0,wall_offset_mV=-15,
            relaxed_map_spectral_radius=float(np.max(np.abs(eigenvalues))),
            eigenvalues=[dict(real=float(z.real),imag=float(z.imag)) for z in eigenvalues],
            eigenpair_residual=float(np.linalg.norm(apply(vec)-eigenvalues[dominant].real*vec)),
            map_finite_difference_relative_error=float(np.linalg.norm(finite_difference-apply(vec))/np.linalg.norm(apply(vec))),
            interpretation='locally unstable' if max(abs(eigenvalues))>1 else 'locally stable'))
    write_json(ROOT/'results/gummel_stability.json',dict(scope='local linearization; not unique global failure cause',
        source_sha256=source_identity(),records=records))
    print(records)


if __name__=='__main__':main()
