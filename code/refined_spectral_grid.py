"""Memory-bounded independent Hankel propagation with exact logarithmic step."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import jv
from scipy.linalg.blas import dgemm
from scipy.integrate import simpson
from spectral_solver import SpectralSolver
class RefinedSpectral(SpectralSolver):
 def propagate_grid(self,tau,ny=401,xmax_at_zc=16.,du=.000625,models=('fresnel','exact_kz')):
  tau=np.asarray(tau);zeta=2*np.sqrt(self.R0)+tau/np.sqrt(self.R0);rr=np.linspace(0,xmax_at_zc/np.sqrt(self.R0),ny);nu=int(np.ceil(self.umax/du))+1;nu+=1-nu%2;u=np.linspace(0,self.umax,nu);weights=np.ones(nu);weights[1:-1:2]=4;weights[2:-1:2]=2;weights*=u[1]/3;rates={mod:(-u*u/2 if mod=='fresnel' else -u*u*self.kw/(np.sqrt(self.kw**2-u*u)+self.kw)) for mod in models};outputs={mod:np.empty((len(tau),self.qmax+1,ny),complex) for mod in models};powers=[]
  for q in range(self.qmax+1):
   _,ulog,Hlog=self.spectra[q];H=CubicSpline(ulog,Hlog)(u);H[0]=self.h0 if q==0 else 0.;powers.append(float(simpson(u*H*H,x=u)/self.input_integral));kernel=jv(q,rr[:,None]*u[None,:])*(weights*u*H)[None,:]
   for mod in models:
    for start in range(0,len(tau),48):
     phase=np.exp(1j*rates[mod][:,None]*zeta[None,start:start+48]);out=dgemm(1.,kernel,np.asfortranarray(phase.real))+1j*dgemm(1.,kernel,np.asfortranarray(phase.imag));outputs[mod][start:start+48,q]=out.T
  common={'tau':tau,'x':zeta[:,None]*rr[None,:]/2,'rho_mm':np.broadcast_to(rr*self.w,(len(tau),ny)).copy(),'z_mm':self.k*self.w**2*zeta,'Pin':self.Pin,'q_orders':np.arange(self.qmax+1),'spectral_nu':nu,'spectral_umax':self.umax,'spectral_du':u[1],'forward_nlog':len(self.s),'retained_channel_power_relative_input':np.array(powers)}
  return {mod:dict(common,U=field,model=mod) for mod,field in outputs.items()}
