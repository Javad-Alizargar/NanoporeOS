import numpy as np
from scipy.special import exprel
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
CFG={'profile': 'standard', 'model': 'steady axisymmetric 2D PNP, cylindrical channel, no fluid', 'temperature_K': 298.15, 'epsilon_r': 78.3, 'Dplus_m2_s': 1.96e-09, 'Dminus_m2_s': 2.03e-09, 'geometry': 'ideal cylindrical channel with charged side wall', 'experimental_validation': False, 'seed': 20261102}
# Numerical kernel: conservative cylindrical finite-volume Scharfetter–Gummel transport
from scipy.constants import elementary_charge as qe, Boltzmann as kb, epsilon_0 as eps0, Avogadro as NA
F=qe*NA
Vt=kb*CFG['temperature_K']/qe
alpha=(qe*F/(eps0*CFG['epsilon_r']*kb*CFG['temperature_K']))*1e-18 # nm^-2 (mol/m3)^-1

def bern(x):
    # Bernoulli B(x)=x/(exp(x)-1), stable near zero and far from zero
    x=np.asarray(x,dtype=float)
    return 1/np.asarray(exprel(x))

class AxisymPNP:
    def __init__(self,Nx=24,Nr=12,Lnm=20.,Rnm=5.,c_mM=100.,V_mV=50.,zeta_mV=-10.):
        self.Nx,self.Nr=Nx,Nr; self.N=Nx*Nr
        self.L,self.R=Lnm,Rnm; self.dx=Lnm/Nx;self.dr=Rnm/Nr
        self.c0=c_mM # 1 mM = 1 mol/m3
        self.voltage=V_mV*1e-3/Vt;self.zeta=zeta_mV*1e-3/Vt
        self.x=(np.arange(Nx)+.5)*self.dx
        self.r=(np.arange(Nr)+.5)*self.dr
        self.volume=np.pi*((self.r+self.dr/2)**2-(self.r-self.dr/2)**2)*self.dx
        self.edges=[];self.boundary=[]
        def idx(i,j):return i*Nr+j
        for i in range(Nx):
            for j in range(Nr):
                p=idx(i,j);rp=self.r[j]
                if i<Nx-1:
                    area=np.pi*((rp+self.dr/2)**2-(rp-self.dr/2)**2)
                    self.edges.append((p,idx(i+1,j),area/self.dx,'x',area))
                if j<Nr-1:
                    area=2*np.pi*(rp+self.dr/2)*self.dx
                    self.edges.append((p,idx(i,j+1),area/self.dr,'r',area))
                area_x=np.pi*((rp+self.dr/2)**2-(rp-self.dr/2)**2)
                if i==0:self.boundary.append((p,2*area_x/self.dx,0.,self.c0,'left',area_x))
                if i==Nx-1:self.boundary.append((p,2*area_x/self.dx,self.voltage,self.c0,'right',area_x))
                if j==Nr-1:
                    area=2*np.pi*self.R*self.dx
                    wallphi=self.voltage*self.x[i]/self.L+self.zeta
                    self.boundary.append((p,2*area/self.dr,wallphi,None,'wall',area))
    def poisson(self,cp,cm):
        A=lil_matrix((self.N,self.N)); b=np.zeros(self.N)
        for p,q,g,_,_ in self.edges:
            A[p,p]+=g; A[q,q]+=g;A[p,q]-=g;A[q,p]-=g
        for p,g,phi,_,_,_ in self.boundary:A[p,p]+=g;b[p]+=g*phi
        charge=cp-cm
        b+=alpha*charge*np.tile(self.volume,self.Nx)
        return spsolve(A.tocsr(),b)
    def species(self,phi,z):
        A=lil_matrix((self.N,self.N)); b=np.zeros(self.N)
        for p,q,g,_,_ in self.edges:
            d=z*(phi[q]-phi[p]); bpos=float(bern(d));bneg=float(bern(-d))
            A[p,p]+=g*bpos;A[p,q]-=g*bneg
            A[q,q]+=g*bneg;A[q,p]-=g*bpos
        for p,g,phib,cb,face,area in self.boundary:
            if cb is None:continue
            d=z*(phib-phi[p]);bp=float(bern(d));bn=float(bern(-d))
            A[p,p]+=g*bp;b[p]+=g*bn*cb
        return spsolve(A.tocsr(),b)
    def solve(self,maxiter=180,tol=3e-7,relax=.22):
        cp=np.full(self.N,self.c0);cm=cp.copy(); trace=[]; success=False
        for k in range(maxiter):
            phi=self.poisson(cp,cm)
            cp0=self.species(phi,1);cm0=self.species(phi,-1)
            if not np.all(np.isfinite(cp0)) or not np.all(np.isfinite(cm0)) or min(cp0.min(),cm0.min())<=0:break
            cpnew=cp+relax*(cp0-cp);cmnew=cm+relax*(cm0-cm)
            err=max(np.max(np.abs(cpnew-cp))/self.c0,np.max(np.abs(cmnew-cm))/self.c0)
            cp,cm=cpnew,cmnew;trace.append(float(err))
            if err<tol:success=True;break
        phi=self.poisson(cp,cm)
        return {'success':success,'iterations':len(trace),'error':trace[-1] if trace else None,'phi':phi,'cp':cp,'cm':cm,'trace':trace}
    def flux_audit(self,out):
        phi,cp,cm=out['phi'],out['cp'],out['cm']
        resid=[]; currents=[]
        for conc,z,D in [(cp,1,CFG['Dplus_m2_s']),(cm,-1,CFG['Dminus_m2_s'])]:
            row=np.zeros(self.N); by_section=np.zeros(self.Nx-1)
            for p,q,g,direction,area in self.edges:
                d=z*(phi[q]-phi[p]);flux=g*(float(bern(d))*conc[p]-float(bern(-d))*conc[q])
                row[p]+=flux;row[q]-=flux
                if direction=='x':by_section[p//self.Nr]+=flux
            for p,g,phib,cb,face,area in self.boundary:
                if cb is None:continue
                d=z*(phib-phi[p]); row[p]+=g*(float(bern(d))*conc[p]-float(bern(-d))*cb)
            resid.append(np.max(np.abs(row))/max(1e-20,self.c0*np.mean([e[2] for e in self.edges])))
            # Flux (nm^2 mol/m^3) * D (m^2/s) / nm^2 to mol/s ; then F converts to A
            currents.append(z*F*D*1e-9*np.mean(by_section))
        return {'max_scaled_species_balance':float(max(resid)),'ionic_current_A':float(sum(currents)),'section_current_spread':None}
print('Kernel loaded, thermal voltage mV',Vt*1e3,'Poisson alpha',alpha)