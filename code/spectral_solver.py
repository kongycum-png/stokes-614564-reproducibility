"""Independent logarithmic Hankel transform angular-spectrum solver.
Uses scipy.fft.fht convention A(k)=integral a(r)Jq(kr) k dr.
Input a=s*f(s); therefore H=A/u. Inverse uses a=u*H*phase, output U=A/s.
Prescribed transverse channels are not projected or redefined.
"""
import numpy as np
from scipy.fft import fht,fhtoffset
from scipy.special import airy,jv
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
class SpectralSolver:
 def __init__(self,R0,eta,qmax=7,nlog=262144,smin=1e-6,smax=1e4,kw=2*np.pi*.08/.0006328,umax=None):
  self.R0=R0;self.eta=eta;self.alpha=eta/(2*np.sqrt(R0));self.kw=kw;self.w=kw*.0006328/(2*np.pi);self.k=2*np.pi/.0006328;self.qmax=qmax
  self.s=np.geomspace(smin,smax,nlog);self.dln=(np.log(smax)-np.log(smin))/(nlog-1);self.f=airy(R0-self.s)[0]*np.exp(self.alpha*(R0-self.s));self.input_integral=simpson(self.f**2*self.s,x=self.s);self.Pin=2*np.pi*self.w**2*self.input_integral;self.spectra={};self.diagnostics={}
  self.h0=simpson(self.s*self.f,x=self.s)
  self.umax=min(np.sqrt(24/self.alpha) if umax is None else umax,.95*kw)
  for q in range(qmax+1):
   off=fhtoffset(self.dln,mu=q,initial=0.);u=np.exp(off)/self.s[::-1];H=fht(self.s*self.f,self.dln,mu=q,offset=off)/u
   # clear only evanescent propagation; preserve spectrum for diagnostics
   powall=simpson(u*H*H,x=u);mask=u>=kw*.95;tail=simpson((u*H*H)[mask],x=u[mask])/powall
   occupied=u<self.umax
   self.spectra[q]=(off,u,H);self.diagnostics[str(q)]={'parseval_relative_error':float(powall/self.input_integral-1),'power_fraction_u_above_0p95kw':float(tail),'retained_power_fraction':float(simpson((u*H*H)[occupied],x=u[occupied])/self.input_integral),'nlog':nlog,'smin':smin,'smax':smax,'umax':self.umax,'near_axis_method':'direct spectral Simpson for rho/w < .02; q0 Gaussian analytic subtraction elsewhere'}
 def propagate(self,tau,x,q_orders=None,model='exact_kz'):
  tau=np.atleast_1d(tau);x=np.asarray(x);zeta=2*np.sqrt(self.R0)+tau/np.sqrt(self.R0);rr=2*x[None,:]/zeta[:,None];orders=list(range(self.qmax+1)) if q_orders is None else q_orders
  out=np.zeros((len(tau),len(orders),len(x)),complex)
  for j,q in enumerate(orders):
   off,u,H=self.spectra[abs(q)];valid=u<self.umax
   ku=np.sqrt(self.kw**2-u[valid]**2);phase_rate=-u[valid]**2/(ku+self.kw)*self.kw if model=='exact_kz' else -u[valid]**2/2
   for start in range(0,len(tau),4):
    z=zeta[start:start+4];phase=np.zeros((len(z),len(u)),complex);phase[:,valid]=np.exp(1j*z[:,None]*phase_rate[None,:]);spec=phase*(H*u)[None,:]
    # Remove the nonzero low-frequency endpoint analytically. This is a
    # quadrature correction, not a change of the physical spectrum/model.
    if q==0:
     aa=1+1j*z
     spec-=self.h0*u[None,:]*np.exp(-aa[:,None]*u[None,:]**2/2)
    vals=(fht(spec.real,self.dln,mu=abs(q),offset=off)+1j*fht(spec.imag,self.dln,mu=abs(q),offset=off))/self.s[None,:]
    if q==0:vals+=self.h0/aa[:,None]*np.exp(-self.s[None,:]**2/(2*aa[:,None]))
    for k,v in enumerate(vals):
     radii=rr[start+k];cs=CubicSpline(self.s,v);row=cs(np.maximum(radii,self.s[0]))
     near=radii<.02
     if near.any():
      integ=(u[valid]*H[valid]*phase[k,valid])[None,:]*jv(abs(q),radii[near,None]*u[None,valid])
      row[near]=simpson(integ,x=u[valid],axis=1)
     row[radii==0]=row[radii==0] if q==0 else 0.
     out[start+k,j]=row
  zmm=self.k*self.w**2*zeta
  return {'tau':tau,'x':np.broadcast_to(x,(len(tau),len(x))).copy(),'rho_mm':rr*self.w,'z_mm':zmm,'U':out,'Pin':self.Pin,'model':model,'q_orders':np.array(orders)}
