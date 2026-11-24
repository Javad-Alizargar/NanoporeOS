from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time
import numpy as np
import scipy
from scipy.constants import elementary_charge, Boltzmann, epsilon_0, Avogadro
from scipy.special import exprel
from scipy.sparse import coo_matrix, bmat, diags
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[2]
F = elementary_charge * Avogadro
VT = Boltzmann * 298.15 / elementary_charge
ALPHA = F / (epsilon_0 * 78.3 * VT) * 1e-18
DP, DM = 1.96e-9, 2.03e-9
MODEL = "axisym-pnp-ramp-offset-v1"


class StopRequested(RuntimeError):
    pass


def check_stop(deadline=None):
    if (ROOT / "STOP_REQUESTED").exists():
        raise StopRequested("Root STOP_REQUESTED exists")
    if deadline is not None and time.monotonic() >= deadline:
        raise StopRequested("Bounded run deadline reached")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__)


def source_identity():
    # Solver cache identity tracks numerical implementation, not unrelated UI edits.
    files = [Path(__file__)]
    return hashlib.sha256("".join(p.name + sha256(p) for p in files).encode()).hexdigest()


@dataclass(frozen=True)
class Scenario:
    nx: int = 24
    nr: int = 12
    length_nm: float = 20.0
    radius_nm: float = 10.0
    salt_mM: float = 100.0
    voltage_mV: float = 100.0
    wall_offset_mV: float = -15.0
    model: str = MODEL

    def __post_init__(self):
        if self.model != MODEL:
            raise ValueError("Unsupported model")
        if type(self.nx) is not int or type(self.nr) is not int:
            raise ValueError("Mesh dimensions must be integers")
        if not (4 <= self.nx <= 160 and 3 <= self.nr <= 80 and self.nx * self.nr <= 12800):
            raise ValueError("Mesh outside bounded limits")
        bounds = dict(length_nm=(5, 200), radius_nm=(1, 30), salt_mM=(1, 300),
                      voltage_mV=(-150, 150), wall_offset_mV=(-30, 30))
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} outside [{low}, {high}]")

    def identity(self):
        return hashlib.sha256(json.dumps(dict(scenario=asdict(self), source=source_identity(),
            environment=environment(), tolerance=2e-11), sort_keys=True).encode()).hexdigest()


def bern(x):
    return 1.0 / exprel(np.asarray(x, dtype=float))


def bern_prime(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    t = x[small]
    out[small] = -.5 + t/6 - t**3/180 + t**5/5040
    t = x[~small]
    b = bern(t)
    out[~small] = b/t - b*(1 + b/t)
    return out


class PNP:
    """Cell-centred cylindrical FV; normalized concentrations; log-density Newton.

    Internal flux p->q: g[B(z*(psi_q-psi_p))*c_p - B(-z*dpsi)*c_q].
    Wall psi = voltage*x/L + offset; wall ionic flux = 0; axis symmetry.
    End psi=0,voltage and c+/c-=salt. No flow, molecule, or transient.
    """

    def __init__(self, s: Scenario):
        self.s = s
        self.n = s.nx*s.nr
        dx, dr = s.length_nm/s.nx, s.radius_nm/s.nr
        self.x = (np.arange(s.nx)+.5)*dx
        self.r = (np.arange(s.nr)+.5)*dr
        ann = np.pi*((self.r+dr/2)**2-(self.r-dr/2)**2)
        self.volume = np.tile(ann*dx, s.nx)
        p, q, g, axis, bp, bg, bv, bc, side = [], [], [], [], [], [], [], [], []
        for i in range(s.nx):
            for j in range(s.nr):
                k = i*s.nr+j
                if i+1 < s.nx:
                    p.append(k); q.append(k+s.nr); g.append(ann[j]/dx); axis.append(0)
                if j+1 < s.nr:
                    p.append(k); q.append(k+1); g.append(2*np.pi*(self.r[j]+dr/2)*dx/dr); axis.append(1)
                if i == 0:
                    bp.append(k); bg.append(2*ann[j]/dx); bv.append(0); bc.append(True); side.append(-1)
                if i == s.nx-1:
                    bp.append(k); bg.append(2*ann[j]/dx); bv.append(s.voltage_mV*1e-3/VT); bc.append(True); side.append(1)
                if j == s.nr-1:
                    bp.append(k); bg.append(4*np.pi*s.radius_nm*dx/dr)
                    bv.append((s.voltage_mV*self.x[i]/s.length_nm+s.wall_offset_mV)*1e-3/VT)
                    bc.append(False); side.append(0)
        self.p, self.q = np.array(p), np.array(q)
        self.g, self.axis = np.array(g), np.array(axis)
        self.bp, self.bg, self.bv = np.array(bp), np.array(bg), np.array(bv)
        self.bc, self.side = np.array(bc), np.array(side)
        self.L = self.matrix(self.g, -self.g, -self.g, self.g) + diags(np.bincount(self.bp, self.bg, minlength=self.n))
        self.pb = np.bincount(self.bp, self.bg*self.bv, minlength=self.n)
        self.scale_p = self.L.diagonal()
        self.scale_c = np.bincount(np.r_[self.p,self.q,self.bp[self.bc]],
            np.r_[self.g,self.g,self.bg[self.bc]], minlength=self.n)

    def matrix(self, pp, pq, qp, qq):
        return coo_matrix((np.r_[pp,pq,qp,qq],
            (np.r_[self.p,self.p,self.q,self.q],np.r_[self.p,self.q,self.p,self.q])), shape=(self.n,self.n)).tocsr()

    def transport(self, phi, u, z, fraction=1, jacobian=False):
        c = np.exp(u)
        d = z*(phi[self.q]-phi[self.p])
        a, b = bern(d), bern(-d)
        flux = self.g*(a*c[self.p]-b*c[self.q])
        residual = np.bincount(self.p,flux,minlength=self.n)-np.bincount(self.q,flux,minlength=self.n)
        bp, bg = self.bp[self.bc], self.bg[self.bc]
        db = z*(fraction*self.bv[self.bc]-phi[bp])
        fb = bg*(bern(db)*c[bp]-bern(-db))
        residual += np.bincount(bp,fb,minlength=self.n)
        if not jacobian:
            return residual, flux, fb
        j_u = self.matrix(self.g*a*c[self.p], -self.g*b*c[self.q],
                           -self.g*a*c[self.p], self.g*b*c[self.q])
        j_u += diags(np.bincount(bp, bg*bern(db)*c[bp], minlength=self.n))
        h = z*self.g*(bern_prime(d)*c[self.p]+bern_prime(-d)*c[self.q])
        j_phi = self.matrix(-h,h,h,-h)
        hb = -z*bg*(bern_prime(db)*c[bp]+bern_prime(-db))
        j_phi += diags(np.bincount(bp,hb,minlength=self.n))
        return residual,j_phi,j_u

    def residual(self, y, fraction=1, jacobian=False):
        phi, up, um = y.reshape(3,self.n)
        cp,cm = np.exp(up),np.exp(um)
        rp = self.L@phi-fraction*self.pb-ALPHA*self.s.salt_mM*self.volume*(cp-cm)
        if not jacobian:
            rc = [self.transport(phi,u,z,fraction)[0]/self.scale_c for u,z in [(up,1),(um,-1)]]
            return np.r_[rp/self.scale_p,*rc]
        r1,j1,a1 = self.transport(phi,up,1,fraction,True)
        r2,j2,a2 = self.transport(phi,um,-1,fraction,True)
        sp,sc = diags(1/self.scale_p),diags(1/self.scale_c)
        charge = ALPHA*self.s.salt_mM*self.volume
        jac = bmat([[sp@self.L,sp@diags(-charge*cp),sp@diags(charge*cm)],
                    [sc@j1,sc@a1,None],[sc@j2,None,sc@a2]],format="csc")
        return np.r_[rp/self.scale_p,r1/self.scale_c,r2/self.scale_c],jac

    def solve(self, deadline=None, initial=None, cancel=None):
        started = time.perf_counter()
        y = np.zeros(3*self.n) if initial is None else np.array(initial,copy=True)
        history=[]
        success=True
        # Fixed, predeclared continuation: boundary values only, same final BVP.
        fractions = [.25,.5,.75,1.] if initial is None else [1.]
        for fraction in fractions:
            for it in range(45):
                check_stop(deadline)
                if cancel is not None and cancel.is_set():
                    raise StopRequested('Job cancellation requested')
                r,j = self.residual(y,fraction,True)
                err=float(np.max(np.abs(r)))
                history.append(dict(fraction=fraction,iteration=it,residual=err))
                if err <= 2e-11:
                    break
                delta=spsolve(j,-r)
                damping=1.0
                accepted=False
                for _ in range(22):
                    trial=y+damping*delta
                    if np.max(np.abs(trial[self.n:])) < 30:
                        rt=self.residual(trial,fraction)
                        if np.all(np.isfinite(rt)) and np.linalg.norm(rt) <= (1-1e-4*damping)*np.linalg.norm(r):
                            y=trial;accepted=True;break
                    damping*=.5
                history[-1]['damping']=damping
                if not accepted:
                    success=False;break
            else:
                success=False
            if not success:
                break
        diag=self.audit(y)
        success=success and diag['coupled_residual_max']<=2e-11 and diag['positive'] and diag['current_consistent']
        return dict(status="PASS" if success else "FAIL", y=y, diagnostics=diag, history=history,
                    seconds=time.perf_counter()-started, scenario=asdict(self.s), identity=self.s.identity(),
                    dataset_class="exploratory_numerical_simulation", training_qualified=False)

    def audit(self,y):
        phi,up,um=y.reshape(3,self.n)
        r=self.residual(y).reshape(3,self.n)
        total=np.zeros(self.s.nx+1)
        species=[]
        local_abs=[]
        boundary=[]
        for u,z,D in [(up,1,DP),(um,-1,DM)]:
            res,flux,fb=self.transport(phi,u,z)
            sec=np.bincount(self.p[self.axis==0]//self.s.nr,flux[self.axis==0],minlength=self.s.nx-1)
            left=fb[self.side[self.bc]==-1].sum()
            right=fb[self.side[self.bc]==1].sum()
            molar = np.r_[-left,sec,right]*D*self.s.salt_mM*1e-9
            total+=z*F*molar
            species.append(float(abs(left+right)*D*self.s.salt_mM*1e-9))
            local_abs.append(float(np.max(np.abs(res))*D*self.s.salt_mM*1e-9))
            boundary.append(dict(left_outward_mol_s=float(left*D*self.s.salt_mM*1e-9),
                                 right_outward_mol_s=float(right*D*self.s.salt_mM*1e-9)))
        spread=float(np.ptp(total)); current=float(np.mean(total))
        return dict(coupled_residual_max=float(np.max(np.abs(r))),
            poisson_scaled=float(np.max(np.abs(r[0]))),plus_scaled=float(np.max(np.abs(r[1]))),
            minus_scaled=float(np.max(np.abs(r[2]))),
            poisson_integrated_charge_residual_C=float(np.max(np.abs(r[0]*self.scale_p))*epsilon_0*78.3*VT*1e-9),
            species_max_local_imbalance_mol_s=local_abs,species_global_imbalance_mol_s=species,
            species_boundary_fluxes=boundary,current_A=current,section_currents_A=total.tolist(),
            current_spread_A=spread,current_consistent=bool(spread<=1e-18+1e-6*max(abs(total))),
            positive=bool(np.all(np.isfinite(y)) and np.all(np.exp(np.r_[up,um])>0)),
            min_concentration_mol_m3=float(self.s.salt_mM*np.exp(np.r_[up,um]).min()),
            boundary_conditions="end Dirichlet face values; axis/wall ion flux zero by construction",
            debye_length_nm=float(1/np.sqrt(2*ALPHA*self.s.salt_mM)),
            radial_cells_per_debye_length=float(self.s.nr/self.s.radius_nm/np.sqrt(2*ALPHA*self.s.salt_mM)))


def ohmic_current(s):
    conductivity=F*(DP+DM)*s.salt_mM/VT
    return -conductivity*np.pi*(s.radius_nm*1e-9)**2/(s.length_nm*1e-9)*s.voltage_mV*1e-3
