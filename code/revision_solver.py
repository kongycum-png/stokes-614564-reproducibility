"""Independent Eq.(7) quadrature, cached Bessel kernels, controlled phase series.
Fields omit only common carrier exp(i*k*z). Physical units: mm.
The y mesh parameterizes transverse source-kernel frequency a=y/(2*R0).
"""
from dataclasses import dataclass
from pathlib import Path
import math,time
import numpy as np
from scipy.special import airy,jv
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.linalg.blas import dgemm
@dataclass(frozen=True)
class Grid:
    ds:float=.04
    cutoff:float=11.
    ny:int=601
    ymax:float=12.
    phase_terms:int=3
    source_start:float=0.
class Solver:
 def __init__(self,R0,eta,qmax=7,grid=Grid(),kw=2*np.pi*.08/.0006328,alpha=None):
  self.R0=float(R0); self.eta=float(eta); self.alpha=float(eta/(2*np.sqrt(R0)) if alpha is None else alpha); self.kw=kw;self.qmax=qmax;self.grid=grid
  self.w=kw*.0006328/(2*np.pi);self.k=2*np.pi/.0006328
  smax=R0+max(45.,grid.cutoff/self.alpha)
  if grid.source_start>=smax:raise ValueError('source_start must be below source cutoff')
  ns=max(2049,int(np.ceil((smax-grid.source_start)/grid.ds))+1);ns+=1-ns%2
  self.s=np.linspace(grid.source_start,smax,ns);self.y=np.linspace(0,grid.ymax,grid.ny);self.a=self.y/(2*R0)
  self.c=np.sqrt(1-(self.a/kw)**2)
  # stable c-1 avoids cancellation for small angles
  self.dc=-(self.a/kw)**2/(1+self.c)
  self.weights=np.ones(ns);self.weights[1:-1:2]=4;self.weights[2:-1:2]=2;self.weights*=(self.s[1]-self.s[0])/3
  self.f=airy(R0-self.s)[0]*np.exp(self.alpha*(R0-self.s))
  self.Pin=2*np.pi*self.w**2*np.dot(self.weights,self.f**2*self.s)
  self.kernel=np.stack([jv(q,self.a[:,None]*self.s[None,:])*(self.weights*self.f*self.s)[None,:] for q in range(qmax+1)])
 def product(self,operand):
  real=np.asfortranarray(operand.real);imag=np.asfortranarray(operand.imag)
  return np.stack([dgemm(1.,k,real)+1j*dgemm(1.,k,imag) for k in self.kernel])
 def propagate(self,tau,model='reduced_RS',source_multiplier=None):
  tau=np.atleast_1d(tau);zeta=2*np.sqrt(self.R0)+tau/np.sqrt(self.R0)
  if np.any(zeta<=0):raise ValueError('z must be positive')
  phase=np.exp(1j*self.s[:,None]**2/(2*zeta[None,:]))
  if source_multiplier is not None:phase=phase*np.asarray(source_multiplier)[:,None]
  total=self.product(phase)
  if model=='reduced_RS':
   M=float(np.max(self.s)**2/(2*np.min(zeta))*np.max(abs(self.dc)))
   for n in range(1,self.grid.phase_terms):
    operand=phase*(1j*self.s[:,None]**2/(2*zeta[None,:]))**n
    term=self.product(operand)
    total+=term*self.dc[None,:,None]**n/math.factorial(n)
   c=self.c
   # Absolute integral error <= L1(source)*exp(M)*M^N/N! before prefactor.
   err=np.sum(abs(self.weights*self.f*self.s))*np.exp(M)*M**self.grid.phase_terms/math.factorial(self.grid.phase_terms)/np.min(zeta)
  elif model=='fresnel':c=np.ones_like(self.c);M=0.;err=0.
  else:raise ValueError(model)
  z=self.k*self.w**2*zeta
  rho=z[:,None]*self.a[None,:]/(self.kw*c[None,:])
  # exp(ik(D-z)), stable D-z=z*(1-c)/c
  phase_common=np.exp(1j*self.k*z[:,None]*(-self.dc[None,:])/c[None,:]) if model=='reduced_RS' else np.exp(1j*self.k*rho**2/(2*z[:,None]))
  U=np.moveaxis(total,2,0)*((-1j)**np.arange(self.qmax+1))[None,:,None]*(-1j*c[None,None,:]**2/zeta[:,None,None])*phase_common[:,None,:]
  x=z[:,None]*rho/(2*self.k*self.w**3)
  return {'tau':tau,'x':x,'rho_mm':rho,'z_mm':z,'U':U,'Pin':self.Pin,'phase_remainder_absolute_bound':err,'phase_M':M,'source_n':len(self.s),'source_smin':self.s[0],'source_smax':self.s[-1],'model':model}
 def q(self,tau,m=1,P=1,model='reduced_RS',factor=1.):
  from scipy.special import jnp_zeros
  d=self.propagate(tau,model);out=[]
  for i,x in enumerate(d['x']):
   b=factor*jnp_zeros(m,1)[0];rho=d['rho_mm'][i];dens=np.pi*(abs(d['U'][i,abs(m-P)])**2-abs(d['U'][i,abs(m+P)])**2)*rho/self.Pin
   integral=CubicSpline(rho,dens).antiderivative();r=np.interp(b,x,rho)
   out.append(float(integral(r)-integral(0)))
  return np.array(out)
