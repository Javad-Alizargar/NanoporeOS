"""Independent nodal finite-difference PB reference for V=0 ONLY.

No PNP assembly/import is reused. Boltzmann distributions follow from zero
electrochemical-potential gradients. Finite cylinder end effects retained.
Not an experimental reference; corner discontinuity remains.
"""
import numpy as np
from scipy.constants import epsilon_0, Boltzmann, elementary_charge, Avogadro
from scipy.sparse import lil_matrix,diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator


def equilibrium_pb(length_nm=20,radius_nm=10,salt_mM=100,wall_mV=-15,nx=160,nr=80):
    vt=Boltzmann*298.15/elementary_charge
    kappa2=2*elementary_charge**2*Avogadro*salt_mM/(epsilon_0*78.3*Boltzmann*298.15)*1e-18
    dx,dr=length_nm/nx,radius_nm/nr
    n=(nx-1)*nr
    a=lil_matrix((n,n));b=np.zeros(n)
    index=lambda i,j:(i-1)*nr+j
    wall=wall_mV*.001/vt
    for i in range(1,nx):
        for j in range(nr):
            k=index(i,j)
            a[k,k]=2/dx**2+(4 if j==0 else 2)/dr**2
            for ii in [i-1,i+1]:
                if 0<ii<nx:a[k,index(ii,j)]=-1/dx**2
            if j==0:
                a[k,index(i,1)]=-4/dr**2
            else:
                a[k,index(i,j-1)]=-(1-1/(2*j))/dr**2
                weight=(1+1/(2*j))/dr**2
                if j+1==nr:b[k]+=weight*wall
                else:a[k,index(i,j+1)]=-weight
    a=a.tocsr();u=np.zeros(n)
    for it in range(35):
        f=a@u+kappa2*np.sinh(u)-b
        if np.max(np.abs(f))<1e-10:break
        u+=spsolve(a+diags(kappa2*np.cosh(u)),-f)
    else:raise RuntimeError('Independent PB failed')
    grid=np.zeros((nx+1,nr+1));grid[1:nx,:nr]=u.reshape(nx-1,nr);grid[1:nx,nr]=wall
    return dict(x=np.linspace(0,length_nm,nx+1),r=np.linspace(0,radius_nm,nr+1),phi=grid,
        residual=float(np.max(np.abs(f))),iterations=it+1)


def sample(ref,x,r):
    xx,rr=np.meshgrid(x,r,indexing='ij')
    return RegularGridInterpolator((ref['x'],ref['r']),ref['phi'])(np.c_[xx.ravel(),rr.ravel()]).reshape(xx.shape)
