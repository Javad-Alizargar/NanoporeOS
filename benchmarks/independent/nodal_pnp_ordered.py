"""Independent nodal reference with a symmetric-graph sparse ordering.

Only sparse factorization column ordering differs from nodal_pnp.py. Equations,
boundaries, nonlinear tolerances and continuation are unchanged. This version
must pass small-grid equivalence checks before large reference calculations.

This module imports no NanoporeOS solver, mesh, Bernoulli function or assembly.
It uses radial dual-volume nodal differences and geometric-mean concentration
conductivities, rather than Scharfetter--Gummel fluxes. All concentrations,
including at the wall, are nodal unknowns through electrochemical potential.
It is a numerical reference candidate, not an experimental validation source.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
from scipy.constants import elementary_charge,Boltzmann,epsilon_0,Avogadro
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve

E=elementary_charge
KT=Boltzmann*298.15
THERMAL_VOLTAGE=KT/E
FARADAY=E*Avogadro
CHARGE_COEFFICIENT=E*E*Avogadro/(epsilon_0*78.3*KT)*1e-18
DIFFUSIVITIES=(1.96e-9,2.03e-9)
PROJECT=Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Problem:
    nx: int=32
    nr: int=16
    length_nm: float=20.
    radius_nm: float=10.
    salt_mM: float=100.
    voltage_mV: float=100.
    wall_offset_mV: float=-15.


def sparse(values,rows,cols,shape):
    return coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=shape).tocsr()


class NodalPNP:
    def __init__(self,problem):
        self.problem=problem
        nx,nr=problem.nx,problem.nr
        self.x=np.linspace(0,problem.length_nm,nx+1)
        self.r=np.linspace(0,problem.radius_nm,nr+1)
        self.dx=problem.length_nm/nx;self.dr=problem.radius_nm/nr
        self.shape=(nx+1,nr+1);self.nodes=(nx+1)*(nr+1)
        grid=np.arange(self.nodes).reshape(self.shape)
        # Ion nodes include the outer wall, potential unknowns exclude it.
        self.interior_ions=grid[1:-1,:].ravel()
        self.interior_phi=grid[1:-1,:-1].ravel()
        self.np=len(self.interior_phi);self.nc=len(self.interior_ions)
        self.n=self.np+2*self.nc
        self.pi=np.full(self.nodes,-1,dtype=int);self.pi[self.interior_phi]=np.arange(self.np)
        self.mi=np.full(self.nodes,-1,dtype=int);self.mi[self.interior_ions]=np.arange(self.nc)
        inner=np.maximum(0,self.r-self.dr/2)
        outer=np.minimum(problem.radius_nm,self.r+self.dr/2)
        annulus=np.pi*(outer**2-inner**2)
        self.volume=np.tile(annulus*self.dx,nx+1)
        axial_p=grid[:-1,:].ravel();axial_q=grid[1:,:].ravel()
        axial_g=np.tile(annulus/self.dx,nx)
        radial_p=grid[:,:-1].ravel();radial_q=grid[:,1:].ravel()
        radial_g=np.tile(2*np.pi*(self.r[:-1]+self.dr/2)*self.dx/self.dr,nx+1)
        self.a_count=len(axial_p)
        self.p=np.r_[axial_p,radial_p];self.q=np.r_[axial_q,radial_q]
        self.g=np.r_[axial_g,radial_g]
        all_laplacian=sparse([self.g,-self.g,-self.g,self.g],
            [self.p,self.p,self.q,self.q],[self.p,self.q,self.p,self.q],(self.nodes,self.nodes))
        self.lap=all_laplacian[self.interior_phi,:]
        self.lap_unknown=self.lap[:,self.interior_phi]
        self.scale_p=self.lap_unknown.diagonal()
        self.scale_c=all_laplacian.diagonal()[self.interior_ions]
        self.scale=np.r_[self.scale_p,self.scale_c,self.scale_c]
        voltage=problem.voltage_mV*.001/THERMAL_VOLTAGE
        offset=problem.wall_offset_mV*.001/THERMAL_VOLTAGE
        self.boundary_phi=np.zeros(self.shape)
        self.boundary_phi[1:-1,-1]=voltage*self.x[1:-1]/problem.length_nm+offset
        self.boundary_phi[-1,:]=voltage
        self.boundary_mu=np.zeros(self.shape);self.boundary_mu[-1,:]=voltage

    def unpack(self,y,fraction):
        potential=fraction*self.boundary_phi.ravel().copy()
        potential[self.interior_phi]=y[:self.np]
        plus=fraction*self.boundary_mu.ravel().copy()
        minus=-fraction*self.boundary_mu.ravel().copy()
        plus[self.interior_ions]=y[self.np:self.np+self.nc]
        minus[self.interior_ions]=y[self.np+self.nc:]
        return potential,plus,minus

    def equation(self,y,fraction=1,jacobian=False):
        phi,mup,mum=self.unpack(y,fraction)
        cp=np.exp(mup-phi);cm=np.exp(mum+phi)
        charge=CHARGE_COEFFICIENT*self.problem.salt_mM*self.volume[self.interior_phi]
        poisson=self.lap@phi-charge*(cp[self.interior_phi]-cm[self.interior_phi])
        residual=[poisson];rows=[];cols=[];values=[]
        if jacobian:
            a=self.lap_unknown.tocoo()
            rows.append(a.row);cols.append(a.col);values.append(a.data)
            index=np.arange(self.np)
            rows.extend([index,index,index])
            cols.extend([index,self.np+self.mi[self.interior_phi],self.np+self.nc+self.mi[self.interior_phi]])
            values.extend([charge*(cp[self.interior_phi]+cm[self.interior_phi]),
                -charge*cp[self.interior_phi],charge*cm[self.interior_phi]])
        for z,mu,concentration,offset in [(1,mup,cp,self.np),(-1,mum,cm,self.np+self.nc)]:
            difference=mu[self.p]-mu[self.q]
            coefficient=self.g*np.exp(.5*(np.log(concentration[self.p])+np.log(concentration[self.q])))
            flux=coefficient*difference
            all_balance=np.bincount(self.p,flux,minlength=self.nodes)-np.bincount(self.q,flux,minlength=self.nodes)
            residual.append(all_balance[self.interior_ions])
            if jacobian:
                dp=coefficient*(1+.5*difference)
                dq=coefficient*(-1+.5*difference)
                dphi=-.5*z*flux
                for row_node,row_sign in [(self.p,1),(self.q,-1)]:
                    active_rows=self.mi[row_node]>=0
                    target=offset+self.mi[row_node]
                    for col_node,derivative,col_map,col_offset in [
                        (self.p,dp,self.mi,offset),(self.q,dq,self.mi,offset),
                        (self.p,dphi,self.pi,0),(self.q,dphi,self.pi,0)]:
                        active=active_rows & (col_map[col_node]>=0)
                        rows.append(target[active]);cols.append(col_offset+col_map[col_node[active]])
                        values.append(row_sign*derivative[active])
        residual=np.concatenate(residual)/self.scale
        if jacobian:
            jac=sparse(values,rows,cols,(self.n,self.n))
            return residual,diags(1/self.scale)@jac
        return residual

    def solve(self,seconds=1200):
        started=time.monotonic();deadline=started+seconds
        y=np.zeros(self.n);history=[];success=True
        for fraction in np.linspace(.125,1,8):
            for iteration in range(50):
                if (PROJECT/'STOP_REQUESTED').exists() or time.monotonic()>deadline:
                    raise RuntimeError('Reference STOP/deadline reached')
                residual,jac=self.equation(y,fraction,True)
                norm=float(np.max(np.abs(residual)))
                history.append(dict(fraction=float(fraction),iteration=iteration,residual=norm))
                if norm<1e-10:break
                direction=spsolve(jac.tocsc(),-residual,permc_spec='MMD_AT_PLUS_A')
                damping=1.
                accepted=False
                for _ in range(24):
                    trial=y+damping*direction
                    phi,mp,mm=self.unpack(trial,fraction)
                    if max(np.max(np.abs(mp-phi)),np.max(np.abs(mm+phi)))<25:
                        rt=self.equation(trial,fraction)
                        if np.isfinite(rt).all() and np.linalg.norm(rt)<=(1-1e-4*damping)*np.linalg.norm(residual):
                            y=trial;accepted=True;break
                    damping*=.5
                history[-1]['damping']=damping
                if not accepted:success=False;break
            else:success=False
            if not success:break
        phi,mp,mm=self.unpack(y,1)
        cp,cm=np.exp(mp-phi),np.exp(mm+phi)
        audit=self.audit(y)
        return dict(status='PASS' if success and audit['coupled_scaled_residual_max']<1e-10 else 'FAIL',
            config=asdict(self.problem),y=y,potential_V=phi.reshape(self.shape)*THERMAL_VOLTAGE,
            cplus_mol_m3=cp.reshape(self.shape)*self.problem.salt_mM,
            cminus_mol_m3=cm.reshape(self.shape)*self.problem.salt_mM,
            history=history,audit=audit,seconds=time.monotonic()-started,
            reference_class='independent_nodal_electrochemical_PNP_geometric_face_conductivity',
            physical_validation='NOT_PERFORMED')

    def audit(self,y):
        phi,mup,mum=self.unpack(y,1)
        residual=self.equation(y)
        currents=np.zeros(self.problem.nx)
        conservation=[]
        for z,mu,diff in [(1,mup,DIFFUSIVITIES[0]),(-1,mum,DIFFUSIVITIES[1])]:
            logc=mu-z*phi
            flux=self.g*np.exp(.5*(logc[self.p]+logc[self.q]))*(mu[self.p]-mu[self.q])
            row=np.bincount(self.p,flux,minlength=self.nodes)-np.bincount(self.q,flux,minlength=self.nodes)
            sections=flux[:self.a_count].reshape(self.problem.nx,self.problem.nr+1).sum(axis=1)
            molar=sections*diff*self.problem.salt_mM*1e-9
            currents+=z*FARADAY*molar
            conservation.append(dict(max_local_imbalance_mol_s=float(np.max(np.abs(row[self.interior_ions]))*diff*self.problem.salt_mM*1e-9),
                global_imbalance_mol_s=float(abs(molar[0]-molar[-1])),
                left_outward_mol_s=float(-molar[0]),right_outward_mol_s=float(molar[-1]),
                section_fluxes_mol_s=molar.tolist()))
        cp,cm=np.exp(mup-phi),np.exp(mum+phi)
        return dict(coupled_scaled_residual_max=float(np.max(np.abs(residual))),
            poisson_scaled_residual_max=float(np.max(np.abs(residual[:self.np]))),
            current_A=float(np.mean(currents)),section_current_spread_A=float(np.ptp(currents)),
            section_currents_A=currents.tolist(),species=conservation,
            positive=bool(np.all(cp>0)&np.all(cm>0)&np.all(np.isfinite(np.r_[cp,cm]))),
            min_concentration_mol_m3=float(min(cp.min(),cm.min())*self.problem.salt_mM),
            ion_wall_flux='zero by omitted exterior face; electrochemical potential at wall is solved',
            potential_end_wall_rim='Dirichlet jump preserved; endpoint corner potential takes reservoir value')


def save_result(directory,model,result):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'fields.npz',x_nm=model.x,r_nm=model.r,
        potential_V=result['potential_V'],cplus_mol_m3=result['cplus_mol_m3'],cminus_mol_m3=result['cminus_mol_m3'])
    payload={k:v for k,v in result.items() if k not in ['y','potential_V','cplus_mol_m3','cminus_mol_m3']}
    payload['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path=directory/'result.json';temp=directory/'result.tmp.json'
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [directory/'fields.npz',path]}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
