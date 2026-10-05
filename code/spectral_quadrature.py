"""Independent FFTLog forward transform + uniform inverse Hankel quadrature.

Uses a fixed physical-radius grid, allowing all axial planes to share kernels.
This avoids periodic inverse FFTLog artifacts without modifying the model.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import jv
from scipy.linalg.blas import dgemm
from spectral_solver import SpectralSolver

class HankelQuadrature(SpectralSolver):
 def propagate_grid(self,tau,ny=601,xmax_at_zc=16.,du=.0025,model='exact_kz'):
  tau=np.asarray(tau);zeta=2*np.sqrt(self.R0)+tau/np.sqrt(self.R0)
  rr=np.linspace(0,xmax_at_zc/np.sqrt(self.R0),ny)
  nu=int(np.ceil(self.umax/du))+1;nu+=1-nu%2
  u=np.linspace(0,self.umax,nu);weights=np.ones(nu);weights[1:-1:2]=4;weights[2:-1:2]=2;weights*=u[1]/3
  models=[model] if isinstance(model,str) else list(model)
  rates=[-u*u/2 if mod=='fresnel' else -u*u*self.kw/(np.sqrt(self.kw**2-u*u)+self.kw) for mod in models]
  phase=np.concatenate([np.exp(1j*rate[:,None]*zeta[None,:]) for rate in rates],axis=1);pre=np.asfortranarray(phase.real);pim=np.asfortranarray(phase.imag)
  fields=[]
  for q in range(self.qmax+1):
   off,ulog,Hlog=self.spectra[q];H=CubicSpline(ulog,Hlog)(u);H[0]=self.h0 if q==0 else 0.
   kernel=jv(q,rr[:,None]*u[None,:])*(weights*u*H)[None,:]
   fields.append((dgemm(1.,kernel,pre)+1j*dgemm(1.,kernel,pim)).T)
  zmm=self.k*self.w**2*zeta
  result={'tau':tau,'x':zeta[:,None]*rr[None,:]/2,'rho_mm':np.broadcast_to(rr*self.w,(len(tau),ny)).copy(),'z_mm':zmm,'Pin':self.Pin,'q_orders':np.arange(self.qmax+1),'spectral_nu':nu,'spectral_umax':self.umax,'spectral_du':u[1],'forward_nlog':len(self.s)}
  full=np.stack(fields,axis=1)
  outputs={mod:dict(result,U=full[i*len(tau):(i+1)*len(tau)],model=mod) for i,mod in enumerate(models)}
  return outputs[model] if isinstance(model,str) else outputs
