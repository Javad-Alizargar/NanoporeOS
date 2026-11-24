"""Opt-in, FORCED PNP code verification, never an unforced physical dataset.

Continuous sources are analytic derivatives, not discrete residuals of samples.
The historical unforced assembly/core hash and original cases remain unchanged.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from .core import PNP,Scenario,ALPHA,VT,DP,DM,F,sha256,source_identity,environment

MODEL='axisym-forced-pnp-manufactured-v1'

@dataclass(frozen=True)
class ManufacturedCase:
    name: str
    length_nm: float=20.
    radius_nm: float=7.
    salt_mM: float=30.
    drive_VT: float=2.
    amplitude_psi: float=.6
    bplus: float=.4
    bminus: float=-.25
    wall_amplitude: float=.3

    def __post_init__(self):
        for value in asdict(self).values():
            if not isinstance(value,str) and not np.isfinite(value):raise ValueError('Nonfinite MMS parameter')
        if max(abs(self.amplitude_psi),abs(self.bplus),abs(self.bminus),abs(self.wall_amplitude))>2:
            raise ValueError('MMS amplitude outside bounded verification family')
        self.scenario(12,6)

    def scenario(self,nx,nr):
        return Scenario(nx=nx,nr=nr,length_nm=self.length_nm,radius_nm=self.radius_nm,
            salt_mM=self.salt_mM,voltage_mV=self.drive_VT*VT*1000,wall_offset_mV=0)

def continuous(case,x,r):
    """Fields and continuum forcing in nm and normalized concentration units.

    psi=v*x/L+a*P+w*sin²(pi*x/L), mu_z=z*(v*x/L+w*sin²)+b_z*M;
    P=sin(pi*x/L)*H, M=sin(2*pi*x/L)*H, H=(1-r²/R²)².
    fpsi=-lap(psi)-ALPHA*cb*(qplus-qminus), fz=div(-qz grad muz).
    """
    x,r=np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(r,dtype=float))
    k=np.pi/case.length_nm;radius=case.radius_nm
    h=(1-(r/radius)**2)**2;hr=-4*r/radius**2+4*r**3/radius**4
    radial_lap_h=-8/radius**2+16*r**2/radius**4
    p=np.sin(k*x)*h;px=k*np.cos(k*x)*h;pr=np.sin(k*x)*hr
    lap_p=np.sin(k*x)*(radial_lap_h-k*k*h)
    m=np.sin(2*k*x)*h;mx=2*k*np.cos(2*k*x)*h;mr=np.sin(2*k*x)*hr
    lap_m=np.sin(2*k*x)*(radial_lap_h-4*k*k*h)
    wall=case.wall_amplitude*np.sin(k*x)**2
    wall_x=case.wall_amplitude*k*np.sin(2*k*x)
    lap_wall=2*case.wall_amplitude*k*k*np.cos(2*k*x)
    psi=case.drive_VT*x/case.length_nm+case.amplitude_psi*p+wall
    concentrations=[];sources=[];axial_flux=[];radial_flux=[];logs=[]
    for z,b in [(1,case.bplus),(-1,case.bminus)]:
        u=b*m-z*case.amplitude_psi*p;q=np.exp(u)
        ux=b*mx-z*case.amplitude_psi*px;ur=b*mr-z*case.amplitude_psi*pr
        mu_x=z*(case.drive_VT/case.length_nm+wall_x)+b*mx;mu_r=b*mr
        source=-q*(b*lap_m+z*lap_wall+ux*mu_x+ur*mu_r)
        concentrations.append(q);sources.append(source);axial_flux.append(-q*mu_x);radial_flux.append(-q*mu_r);logs.append(u)
    poisson=-(case.amplitude_psi*lap_p+lap_wall)-ALPHA*case.salt_mM*(concentrations[0]-concentrations[1])
    return dict(psi=psi,cp=concentrations[0],cm=concentrations[1],up=logs[0],um=logs[1],
        forcing=np.array([poisson,*sources]),jx=np.array(axial_flux),jr=np.array(radial_flux))

def exact_mid_current(case,order=128):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    r=(nodes+1)*case.radius_nm/2;weights=weights*case.radius_nm/2
    fields=continuous(case,np.full_like(r,case.length_nm/2),r)
    total=sum(z*d*np.sum(weights*r*fields['jx'][i]) for i,(z,d) in enumerate([(1,DP),(-1,DM)]))
    return float(2*np.pi*F*case.salt_mM*1e-9*total)

def implementation_identity():
    return dict(core_source_sha256=source_identity(),manufactured_source_sha256=sha256(Path(__file__)),
        model=MODEL,forcing_quadrature='cell-centre midpoint integration of continuous analytic sources',
        source_equations='continuous fpsi=-lap psi-alpha cb(q+-q-); fi=-qi(lap mu+grad logqi dot grad mu)',
        boundaries='end psi 0/v and q=1; wall psi=v*x/L+w*sin²(pi*x/L), zero ion flux; regular axis',
        solver='unchanged core Newton; four fixed fractions; 2e-11 scaled residual; log bound30')

class ManufacturedPNP(PNP):
    verification_only=True

    def __init__(self,case,nx,nr):
        self.case=case;super().__init__(case.scenario(nx,nr))
        wall=self.side==0
        self.bv[wall]=(case.drive_VT*self.x[self.bp[wall]//nr]/case.length_nm+
            case.wall_amplitude*np.sin(np.pi*self.x[self.bp[wall]//nr]/case.length_nm)**2)
        self.pb=np.bincount(self.bp,self.bg*self.bv,minlength=self.n)
        xx,rr=np.meshgrid(self.x,self.r,indexing='ij')
        self.exact=continuous(case,xx,rr)
        self.integrated_sources=self.exact['forcing'].reshape(3,self.n)*self.volume
        self.scaled_sources=np.r_[self.integrated_sources[0]/self.scale_p,
            self.integrated_sources[1]/self.scale_c,self.integrated_sources[2]/self.scale_c]

    def residual(self,y,fraction=1,jacobian=False):
        result=super().residual(y,fraction,jacobian)
        if jacobian:
            residual,matrix=result;return residual-fraction*self.scaled_sources,matrix
        return result-fraction*self.scaled_sources

    def audit(self,y):
        diagnostics=super().audit(y)
        sources=F*self.s.salt_mM*1e-9*(DP*self.integrated_sources[1]-DM*self.integrated_sources[2])
        corrected=np.asarray(diagnostics['section_currents_A'])-np.r_[0,np.cumsum(sources.reshape(self.s.nx,self.s.nr).sum(axis=1))]
        spread=float(np.ptp(corrected))
        diagnostics.update(unforced_current_consistency='NOT_APPLICABLE_PRESCRIBED_SPECIES_SOURCES',
            source_corrected_current_spread_A=spread,source_corrected_section_currents_A=corrected.tolist(),
            current_consistent=bool(spread<=1e-18+1e-6*np.max(np.abs(corrected))),
            physical_validation='NOT_PERFORMED_VERIFICATION_ONLY',
            boundary_conditions=implementation_identity()['boundaries'])
        diagnostics['source_corrected_global_species_imbalance_mol_s']=[
            abs(diagnostics['species_boundary_fluxes'][i]['left_outward_mol_s']+
                diagnostics['species_boundary_fluxes'][i]['right_outward_mol_s']-
                np.sum(self.integrated_sources[i+1])*d*self.s.salt_mM*1e-9)
            for i,d in enumerate([DP,DM])]
        diagnostics['raw_boundary_net_species_flux_mol_s']=diagnostics['species_global_imbalance_mol_s']
        diagnostics['raw_species_max_local_flux_divergence_mol_s']=diagnostics['species_max_local_imbalance_mol_s']
        diagnostics['species_global_imbalance_mol_s']=diagnostics['source_corrected_global_species_imbalance_mol_s']
        phi,up,um=y.reshape(3,self.n)
        diagnostics['species_max_local_imbalance_mol_s']=[float(np.max(np.abs(
            self.transport(phi,u,z)[0]-self.integrated_sources[i+1]))*d*self.s.salt_mM*1e-9)
            for i,(u,z,d) in enumerate([(up,1,DP),(um,-1,DM)])]
        diagnostics['species_balance_definition']='Outward numerical flux minus integrated prescribed continuous source'
        return diagnostics

    def identity(self):
        content=dict(case=asdict(self.case),nx=self.s.nx,nr=self.s.nr,implementation=implementation_identity(),environment=environment())
        return hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()

    def solve(self,**kwargs):
        result=super().solve(**kwargs)
        result.update(identity=self.identity(),model=MODEL,verification_only=True,
            dataset_class='manufactured_forced_PDE_code_verification_not_physical_training_data',training_qualified=False,
            manufactured_case=asdict(self.case),implementation=implementation_identity())
        result['scenario']['model']=MODEL
        return result

    def mid_current(self,result):
        return float(result['diagnostics']['section_currents_A'][self.s.nx//2])

    def exact_errors(self,y,current_A):
        fields=y.reshape(3,self.s.nx,self.s.nr);reference=exact_mid_current(self.case)
        return dict(max_phi_error_VT=float(np.max(abs(fields[0]-self.exact['psi']))),
            max_plus_relative_error=float(np.max(abs(np.exp(fields[1])/self.exact['cp']-1))),
            max_minus_relative_error=float(np.max(abs(np.exp(fields[2])/self.exact['cm']-1))),
            mid_current_absolute_error_A=abs(current_A-reference),
            mid_current_relative_error=abs(current_A-reference)/max(abs(reference),1e-15),
            exact_mid_current_A=reference,exact_current_quadrature_change_A=abs(reference-exact_mid_current(self.case,64)))
