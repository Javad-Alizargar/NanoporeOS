"""Disjoint symbolic continuous reference for three forced-PNP families.

No canonical package imports. Derivatives are obtained by SymPy from the exact
values, rather than the controller's hand-coded derivative implementation.
Coordinates are nm; q is bulk-normalized and psi is thermal-voltage normalized.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import numpy as np
import sympy as sp

ELEMENTARY_CHARGE=1.602176634e-19
BOLTZMANN=1.380649e-23
AVOGADRO=6.02214076e23
TEMPERATURE=298.15
EPSILON=78.3*8.8541878188e-12
FARADAY=ELEMENTARY_CHARGE*AVOGADRO
THERMAL_VOLTAGE=BOLTZMANN*TEMPERATURE/ELEMENTARY_CHARGE
ALPHA=FARADAY/(EPSILON*THERMAL_VOLTAGE)*1e-18

@dataclass(frozen=True)
class Parameters:
    family:str
    length_nm:float=20.
    radius_nm:float=7.
    salt_mM:float=30.
    drive_VT:float=2.
    amplitude_psi:float=.6
    bplus:float=.4
    bminus:float=-.25
    wall_amplitude:float=.3

def values(p,x,r):
    u,t=np.broadcast_arrays(np.asarray(x,dtype=float)/p.length_nm,np.asarray(r,dtype=float)/p.radius_nm)
    h=1-t*t
    if p.family=='polynomial':
        a=16*u*u*(1-u)**2; b=a*(2*u-1)
    elif p.family=='oscillatory':
        a=np.sin(3*np.pi*u);b=np.sin(4*np.pi*u)
    elif p.family=='localized':
        a=4*u*(1-u)*np.exp(-36*(u-.35)**2)
        b=4*u*(1-u)*(2*u-1)*np.exp(-25*(u-.65)**2)
    else:raise ValueError('Unknown independent family')
    shape_p=a*h**2;shape_m=b*h**3
    wall=p.wall_amplitude*np.sin(np.pi*u)**2
    psi=p.drive_VT*u+p.amplitude_psi*shape_p+wall
    logs=np.array([p.bplus*shape_m-p.amplitude_psi*shape_p,
                   p.bminus*shape_m+p.amplitude_psi*shape_p])
    return dict(psi=psi,q=np.exp(logs),logq=logs,
        mu=logs+np.array([psi,-psi]))

@lru_cache(None)
def _symbolic(family):
    u,t,L,R,c,v,a,bp,bm,w=sp.symbols('u t L R c v a bp bm w',real=True)
    h=1-t*t
    if family=='polynomial':
        axial_p=16*u*u*(1-u)**2;axial_m=axial_p*(2*u-1)
    elif family=='oscillatory':
        axial_p=sp.sin(3*sp.pi*u);axial_m=sp.sin(4*sp.pi*u)
    elif family=='localized':
        axial_p=4*u*(1-u)*sp.exp(-36*(u-sp.Rational(35,100))**2)
        axial_m=4*u*(1-u)*(2*u-1)*sp.exp(-25*(u-sp.Rational(65,100))**2)
    else:raise ValueError('Unknown family')
    pp=axial_p*h**2;mm=axial_m*h**3
    psi=v*u+a*pp+w*sp.sin(sp.pi*u)**2
    logqs=[bp*mm-a*pp,bm*mm+a*pp]
    qs=[sp.exp(q) for q in logqs]
    mus=[logqs[0]+psi,logqs[1]-psi]
    lap_psi=sp.diff(psi,u,2)/L**2+(sp.diff(psi,t,2)+sp.cancel(sp.diff(psi,t)/t))/R**2
    forces=[-lap_psi-ALPHA*c*(qs[0]-qs[1])]
    jx=[];jr=[]
    for q,mu in zip(qs,mus):
        fx=-q*sp.diff(mu,u)/L
        fr=-q*sp.diff(mu,t)/R
        # Cylindrical divergence formed directly from independently symbolic flux.
        source=sp.diff(fx,u)/L+sp.diff(fr,t)/R-q*sp.cancel(sp.diff(mu,t)/t)/R**2
        forces.append(source);jx.append(fx);jr.append(fr)
    expressions=[psi,*qs,*logqs,*mus,*jx,*jr,*forces]
    return sp.lambdify((u,t,L,R,c,v,a,bp,bm,w),expressions,modules='numpy',cse=True)

def fields(p,x,r):
    x,r=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(r,dtype=float))
    raw=_symbolic(p.family)(x/p.length_nm,r/p.radius_nm,p.length_nm,p.radius_nm,p.salt_mM,
        p.drive_VT,p.amplitude_psi,p.bplus,p.bminus,p.wall_amplitude)
    raw=[np.broadcast_to(np.asarray(v,dtype=float),x.shape) for v in raw]
    return dict(psi=raw[0],q=np.array(raw[1:3]),logq=np.array(raw[3:5]),mu=np.array(raw[5:7]),
        jx=np.array(raw[7:9]),jr=np.array(raw[9:11]),forcing=np.array(raw[11:14]))

def numerical_forcing(p,x,r,hfraction):
    """Second-order nested finite differences using ONLY q and psi values.

    Ionic flux uses -grad(q)-z*q*grad(psi), avoiding symbolic mu/flux formulas.
    Interior points only; evaluate axis separately through its regular limit.
    """
    x,r=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(r,dtype=float))
    hx=p.length_nm*hfraction;hr=p.radius_nm*hfraction
    center=values(p,x,r)
    def psi(xx,rr):return values(p,xx,rr)['psi']
    lap=(psi(x+hx,r)-2*center['psi']+psi(x-hx,r))/hx**2
    lap+=(psi(x,r+hr)-2*center['psi']+psi(x,r-hr))/hr**2
    lap+=(psi(x,r+hr)-psi(x,r-hr))/(2*hr*r)
    fpsi=-lap-ALPHA*p.salt_mM*(center['q'][0]-center['q'][1])
    def numerical_flux(xx,rr,axis):
        dx,dr=(hx,0) if axis==0 else (0,hr)
        step=hx if axis==0 else hr
        plus=values(p,xx+dx,rr+dr);minus=values(p,xx-dx,rr-dr)
        q=values(p,xx,rr)['q']
        dq=(plus['q']-minus['q'])/(2*step)
        dpsi=(plus['psi']-minus['psi'])/(2*step)
        return -dq-np.array([1.,-1.])[:,None]*q*dpsi
    fx=(numerical_flux(x+hx,r,0)-numerical_flux(x-hx,r,0))/(2*hx)
    fr=((r+hr)*numerical_flux(x,r+hr,1)-(r-hr)*numerical_flux(x,r-hr,1))/(2*hr*r)
    return np.array([fpsi,*(fx+fr)])

def partial_integral(p,order=64,fraction=.37):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    xmax=p.length_nm*fraction
    x=(nodes+1)*xmax/2;wx=weights*xmax/2
    r=(nodes+1)*p.radius_nm/2;wr=weights*p.radius_nm/2
    xx,rr=np.meshgrid(x,r,indexing='ij');volume=wx[:,None]*wr[None,:]*rr
    source=np.sum(fields(p,xx,rr)['forcing'][1:]*volume,axis=(1,2))
    boundary=np.sum((fields(p,np.full_like(r,xmax),r)['jx']-
        fields(p,np.zeros_like(r),r)['jx'])*r*wr,axis=1)
    # 2*pi is omitted on both sides; radial boundary flux vanishes exactly.
    return dict(integrated_source_nm=source.tolist(),boundary_flux_nm=boundary.tolist(),
        absolute_imbalance_nm=np.abs(source-boundary).tolist())

def numerical_axis_forcing(p,x,hfraction):
    """Values-only regular-axis limit, lap(f)=f_xx+2*f_rr at r=0."""
    x=np.asarray(x,dtype=float);zero=np.zeros_like(x)
    hx=p.length_nm*hfraction;hr=p.radius_nm*hfraction
    center=values(p,x,zero)
    xp=values(p,x+hx,zero);xm=values(p,x-hx,zero)
    rp=values(p,x,zero+hr);rm=values(p,x,zero-hr)
    psi_x=(xp['psi']-xm['psi'])/(2*hx)
    psi_xx=(xp['psi']-2*center['psi']+xm['psi'])/hx**2
    psi_rr=(rp['psi']-2*center['psi']+rm['psi'])/hr**2
    q_x=(xp['q']-xm['q'])/(2*hx)
    q_xx=(xp['q']-2*center['q']+xm['q'])/hx**2
    q_rr=(rp['q']-2*center['q']+rm['q'])/hr**2
    lap_psi=psi_xx+2*psi_rr
    fpsi=-lap_psi-ALPHA*p.salt_mM*(center['q'][0]-center['q'][1])
    species=-(q_xx+2*q_rr+np.array([1.,-1.])[:,None]*(q_x*psi_x+center['q']*lap_psi))
    return np.array([fpsi,*species])

def parameter_mapping(case):
    """Convert a plain dict/dataclass without importing the canonical model."""
    raw=case if isinstance(case,dict) else asdict(case)
    return Parameters(**{k:raw[k] for k in Parameters.__dataclass_fields__})
