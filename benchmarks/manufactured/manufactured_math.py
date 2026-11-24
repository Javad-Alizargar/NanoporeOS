"""Independent continuous manufactured PNP fields and forcing in nm coordinates.

No canonical mesh, residual, Bernoulli function or solver is imported here.
These are forced-equation verification fields, not source-free physical PNP
simulations or qualified training targets.
"""
from dataclasses import dataclass,asdict
import numpy as np

# Independent SI constants (2019 SI exact e,kB,NA; supplied epsilon_r,T).
E=1.602176634e-19
KB=1.380649e-23
NA=6.02214076e23
# Match the detected SciPy1.16.3 physical-constant table, not solver assembly.
EPS0=8.8541878188e-12
TEMPERATURE_K=298.15
EPSILON_R=78.3
VT=KB*TEMPERATURE_K/E
FARADAY=E*NA
ALPHA_NM=E*E*NA/(EPS0*EPSILON_R*KB*TEMPERATURE_K)*1e-18


@dataclass(frozen=True)
class Parameters:
    length_nm: float=20.
    radius_nm: float=10.
    bulk_mol_m3: float=100.
    applied_thermal_voltages: float=1.3
    potential_amplitude: float=.3
    plus_mu_amplitude: float=.17
    minus_mu_amplitude: float=-.11
    wall_amplitude: float=.2

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):raise ValueError('Finite parameters required')
        if min(self.length_nm,self.radius_nm,self.bulk_mol_m3)<=0:raise ValueError('Positive scales required')


def fields(x_nm,r_nm,p=Parameters()):
    """Broadcast coordinates; smooth even-r extension permits independent checks.

    psi=phi/VT, q=ci/cbulk, mu_i=log(q_i)+zi*psi. Gradients are nm^-1,
    Laplacians and f are nm^-2. normalized flux j=-q*grad(mu) has nm^-1.
    The continuous forcing is computed from closed form, never a discrete
    operator. Physical domain:0<=x<=L,0<=r<=R.
    """
    x,r=np.broadcast_arrays(np.asarray(x_nm,dtype=float),np.asarray(r_nm,dtype=float))
    L,R=p.length_nm,p.radius_nm
    k=np.pi/L
    sx,cx=np.sin(k*x),np.cos(k*x)
    s2,c2=np.sin(2*k*x),np.cos(2*k*x)
    H=(1-(r/R)**2)**2
    Hr=-4*r/R**2+4*r**3/R**4
    # Hrr+Hr/r, evaluated by its regular polynomial including the axis limit.
    radial_lap_H=-8/R**2+16*r**2/R**4
    P=sx*H;Px=k*cx*H;Pr=sx*Hr
    M=s2*H;Mx=2*k*c2*H;Mr=s2*Hr
    lapP=sx*(radial_lap_H-k*k*H)
    lapM=s2*(radial_lap_H-4*k*k*H)
    a=p.potential_amplitude;v=p.applied_thermal_voltages;w=p.wall_amplitude
    W=w*sx*sx;Wx=w*k*s2;Wxx=2*w*k*k*c2
    psi=v*x/L+a*P+W
    psix=v/L+a*Px+Wx;psir=a*Pr
    lap_psi=a*lapP+Wxx
    result=dict(psi=psi,psi_x=psix,psi_r=psir,lap_psi=lap_psi,
        wall_psi=v*x/L+W,radial_lap_H=radial_lap_H)
    for name,z,b in [('plus',1,p.plus_mu_amplitude),('minus',-1,p.minus_mu_amplitude)]:
        mu=z*(v*x/L+W)+b*M
        mux=z*(v/L+Wx)+b*Mx;mur=b*Mr
        lap_mu=z*Wxx+b*lapM
        logq=b*M-z*a*P
        logqx=b*Mx-z*a*Px;logqr=b*Mr-z*a*Pr
        q=np.exp(logq)
        jx=-q*mux;jr=-q*mur
        forcing=-q*(lap_mu+logqx*mux+logqr*mur)
        result[name]=dict(z=z,q=q,logq=logq,mu=mu,mu_x=mux,mu_r=mur,
            logq_x=logqx,logq_r=logqr,lap_mu=lap_mu,j_x=jx,j_r=jr,
            forcing_nm2=forcing)
    result['poisson_forcing_nm2']=-lap_psi-ALPHA_NM*p.bulk_mol_m3*(result['plus']['q']-result['minus']['q'])
    result['external_charge_density_C_m3']=EPS0*EPSILON_R*VT*1e18*result['poisson_forcing_nm2']
    return result


def physical_species_source_mol_m3_s(result,name,bulk_mol_m3,diffusion_m2_s):
    """div(J_i)=D_i*cbulk*1e18*f_i, with J_i in mol/(m²s)."""
    return diffusion_m2_s*bulk_mol_m3*1e18*result[name]['forcing_nm2']
