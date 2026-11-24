"""Distinct smooth forced-PNP verification families; never physical training data.

Historical manufactured.py and core.py remain unchanged. Continuous sources use
analytic derivatives; no discrete exact-sample residual is used as forcing.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numpy.polynomial import Polynomial
from .core import PNP,ALPHA,DP,DM,F,sha256,source_identity,environment
from .manufactured import ManufacturedCase,ManufacturedPNP

MODEL='axisym-forced-pnp-distinct-families-v1'
FAMILIES=('polynomial','oscillatory','localized')

@dataclass(frozen=True)
class FamilyCase(ManufacturedCase):
    family: str='polynomial'

    def __post_init__(self):
        super().__post_init__()
        if self.family not in FAMILIES:raise ValueError('Unknown verification function family')


def axial(family,u,mode):
    """Value, first and second derivatives with respect to u=x/L."""
    if family=='oscillatory':
        k=(3 if mode=='P' else 4)*np.pi
        return np.sin(k*u),k*np.cos(k*u),-k*k*np.sin(k*u)
    if family=='polynomial':
        p=Polynomial([0,0,16,-32,16])
        if mode=='M':p=p*Polynomial([-1,2])
        return p(u),p.deriv()(u),p.deriv(2)(u)
    if family=='localized':
        p=Polynomial([0,4,-4])
        if mode=='M':p=p*Polynomial([-1,2])
        k,centre=(36.,.35) if mode=='P' else (25.,.65)
        d=u-centre;g=np.exp(-k*d*d);g1=-2*k*d*g;g2=(4*k*k*d*d-2*k)*g
        return p(u)*g,p.deriv()(u)*g+p(u)*g1,p.deriv(2)(u)*g+2*p.deriv()(u)*g1+p(u)*g2
    raise ValueError('Unknown verification family')


def separable(case,x,r,mode):
    u=x/case.length_nm;radius=case.radius_nm;m=2 if mode=='P' else 3
    a,a1,a2=axial(case.family,u,mode);h0=1-(r/radius)**2
    h=h0**m;hr=-2*m*r/radius**2*h0**(m-1)
    radial_lap=-4*m/radius**2*h0**(m-1)+4*m*(m-1)*r*r/radius**4*h0**(m-2)
    return a*h,a1*h/case.length_nm,a*hr,a2*h/case.length_nm**2+a*radial_lap


def continuous(case,x,r):
    x,r=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(r,dtype=float))
    p,px,pr,lp=separable(case,x,r,'P');m,mx,mr,lm=separable(case,x,r,'M')
    k=np.pi/case.length_nm;wall=case.wall_amplitude*np.sin(k*x)**2
    wx=case.wall_amplitude*k*np.sin(2*k*x);lw=2*case.wall_amplitude*k*k*np.cos(2*k*x)
    psi=case.drive_VT*x/case.length_nm+case.amplitude_psi*p+wall
    logs=[];qs=[];sources=[];jx=[];jr=[]
    for z,b in [(1,case.bplus),(-1,case.bminus)]:
        logq=b*m-z*case.amplitude_psi*p;q=np.exp(logq)
        ux=b*mx-z*case.amplitude_psi*px;ur=b*mr-z*case.amplitude_psi*pr
        mux=z*(case.drive_VT/case.length_nm+wx)+b*mx;mur=b*mr
        sources.append(-q*(b*lm+z*lw+ux*mux+ur*mur))
        logs.append(logq);qs.append(q);jx.append(-q*mux);jr.append(-q*mur)
    fp=-(case.amplitude_psi*lp+lw)-ALPHA*case.salt_mM*(qs[0]-qs[1])
    return dict(psi=psi,cp=qs[0],cm=qs[1],up=logs[0],um=logs[1],
        forcing=np.array([fp,*sources]),jx=np.array(jx),jr=np.array(jr))


def exact_mid_current(case,order=128):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    r=(nodes+1)*case.radius_nm/2;weights=weights*case.radius_nm/2
    values=continuous(case,np.full_like(r,case.length_nm/2),r)
    return float(2*np.pi*F*case.salt_mM*1e-9*sum(z*d*np.sum(weights*r*values['jx'][i])
        for i,(z,d) in enumerate([(1,DP),(-1,DM)])))


def implementation_identity():
    return dict(core_source_sha256=source_identity(),families_source_sha256=sha256(Path(__file__)),
        source_aware_audit_source_sha256=sha256(Path(__file__).with_name('manufactured.py')),
        model=MODEL,forcing_quadrature='cell-centre midpoint continuous analytic sources',
        boundaries='end psi0/v q1; sidewall psi=v*x/L+w*sin²(pi*x/L), zero species flux; regular axis',
        solver='unchanged core Newton with four continuation fractions; scaled residual2e-11; log bound30')


class FamilyPNP(ManufacturedPNP):
    def __init__(self,case,nx,nr):
        self.case=case;PNP.__init__(self,case.scenario(nx,nr))
        wall=self.side==0
        self.bv[wall]=case.drive_VT*self.x[self.bp[wall]//nr]/case.length_nm+case.wall_amplitude*np.sin(
            np.pi*self.x[self.bp[wall]//nr]/case.length_nm)**2
        self.pb=np.bincount(self.bp,self.bg*self.bv,minlength=self.n)
        xx,rr=np.meshgrid(self.x,self.r,indexing='ij');self.exact=continuous(case,xx,rr)
        self.integrated_sources=self.exact['forcing'].reshape(3,self.n)*self.volume
        self.scaled_sources=np.r_[self.integrated_sources[0]/self.scale_p,
            self.integrated_sources[1]/self.scale_c,self.integrated_sources[2]/self.scale_c]

    def identity(self):
        payload=dict(case=asdict(self.case),nx=self.s.nx,nr=self.s.nr,
            implementation=implementation_identity(),environment=environment())
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

    def solve(self,**kwargs):
        result=super().solve(**kwargs)
        result.update(model=MODEL,implementation=implementation_identity())
        result['scenario']['model']=MODEL
        return result

    def exact_errors(self,y,current_A):
        fields=y.reshape(3,self.s.nx,self.s.nr);reference=exact_mid_current(self.case)
        return dict(max_phi_error_VT=float(np.max(abs(fields[0]-self.exact['psi']))),
            max_plus_relative_error=float(np.max(abs(np.exp(fields[1])/self.exact['cp']-1))),
            max_minus_relative_error=float(np.max(abs(np.exp(fields[2])/self.exact['cm']-1))),
            mid_current_absolute_error_A=abs(current_A-reference),
            mid_current_relative_error=abs(current_A-reference)/max(abs(reference),1e-15),
            exact_mid_current_A=reference,exact_current_quadrature_change_A=abs(reference-exact_mid_current(self.case,64)))
