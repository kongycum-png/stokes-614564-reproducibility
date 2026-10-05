"""Unexpanded scalar Rayleigh-Sommerfeld I kernel, independent spatial quadrature.

K=z exp(ikd)(1/d-ik)/(2*pi*d^2), d=|r_observer-r_source|.
Dimensionless radii use w; carrier exp(ikz) is removed analytically.
"""
import numpy as np
from scipy.special import airy
from scipy.fft import fft
class FullRS:
 def __init__(self,R0,eta,qmax=7,ds=.02,cutoff=20,kw=2*np.pi*.08/.0006328,angular_multiplier=1):
  self.R0=R0;self.eta=eta;self.alpha=eta/(2*np.sqrt(R0));self.kw=kw;self.k=2*np.pi/.0006328;self.w=kw/self.k;self.qmax=qmax;self.angular_multiplier=angular_multiplier
  smax=R0+max(45,cutoff/self.alpha);ns=int(np.ceil(smax/ds))+1;ns+=1-ns%2;self.s=np.linspace(0,smax,ns);self.f=airy(R0-self.s)[0]*np.exp(self.alpha*(R0-self.s));self.weights=np.ones(ns);self.weights[1:-1:2]=4;self.weights[2:-1:2]=2;self.weights*=self.s[1]/3;self.Pin=2*np.pi*self.w**2*np.dot(self.weights,self.f**2*self.s)
 def propagate(self,tau,x):
  tau=np.atleast_1d(tau);x=np.atleast_1d(x);zeta=2*np.sqrt(self.R0)+tau/np.sqrt(self.R0);rr=2*x[None,:]/zeta[:,None];U=np.zeros((len(tau),self.qmax+1,len(x)),complex);ntheta_used=[]
  for it,zet in enumerate(zeta):
   Z=self.kw*zet
   for ir,r in enumerate(rr[it]):
    maxarg=r*self.s[-1]/zet;nphi=int(2**np.ceil(np.log2(max(64,2*maxarg+2*self.qmax+32))))*self.angular_multiplier;cos=np.cos(2*np.pi*np.arange(nphi)/nphi);total=np.zeros(self.qmax+1,complex)
    for start in range(0,len(self.s),1024):
     s=self.s[start:start+1024];v=s[:,None]**2+r*r-2*s[:,None]*r*cos[None,:];distance=np.sqrt(Z*Z+v);delta=v/(distance+Z)
     kernel=Z/distance**2*(1/distance-1j*self.kw)*np.exp(1j*self.kw*delta)
     angular=fft(kernel,axis=1)[:,:self.qmax+1]/nphi
     # 2*pi from the angular integral cancels the kernel denominator.
     total+=np.sum(angular*(self.weights[start:start+len(s)]*self.f[start:start+len(s)]*s)[:,None],axis=0)
    U[it,:,ir]=total;ntheta_used.append(nphi)
  return {'tau':tau,'x':np.broadcast_to(x,(len(tau),len(x))).copy(),'rho_mm':rr*self.w,'z_mm':self.k*self.w**2*zeta,'U':U,'Pin':self.Pin,'model':'unexpanded_scalar_RS','source_n':len(self.s),'source_smax':self.s[-1],'max_angular_samples':max(ntheta_used)}
