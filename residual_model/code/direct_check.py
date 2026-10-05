"""Independent positive-source Gauss quadrature of the manuscript radial integral.
Uses no old propagation solver, no axial/radial interpolation, no phase expansion.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import airy,jv,roots_legendre,jnp_zeros
from scipy.optimize import minimize_scalar,brentq
from compact_airy import events,signal
ROOT=Path(__file__).resolve().parents[1]
def direct(tau,R0,eta,m,nx=36,order=16,width=1.,cutoff=22.,model='fresnel'):
    z=2*np.sqrt(R0)+tau/np.sqrt(R0);alpha=eta/(2*np.sqrt(R0));kw=2*np.pi*.08/.0006328
    b=jnp_zeros(m,1)[0];u,wu=roots_legendre(nx);x=(u+1)*b/2;wx=wu*b/2;rho=2*x/z
    top=R0+cutoff/alpha;edges=np.linspace(0,top,int(np.ceil(top/width))+1)
    v,wv=roots_legendre(order);s=((edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])/2*v).ravel();ws=((edges[1:,None]-edges[:-1,None])/2*wv).ravel()
    f=airy(R0-s)[0]*np.exp(alpha*(R0-s))*s*ws
    D=np.sqrt(z*z+(rho/kw)**2) if model=='reduced_RS' else np.full_like(rho,z)
    phase=np.exp(1j*s[None,:]**2/(2*D[:,None]));arg=rho[:,None]*s[None,:]/D[:,None]
    U=np.array([np.einsum("xs,s->x",jv(q,arg)*phase,f,optimize=False)*z/(D*D) for q in [m-1,m+1]])
    Q=np.sum(wx*x*(abs(U[0])**2-abs(U[1])**2))*(2/z)**2
    # Normalize to exactly the same tau-independent scale as compact signal.
    I0=2*m*jv(m,b)**2
    return Q/(4*np.pi*np.sqrt(R0)*np.exp(-eta*np.sqrt(R0))*I0)

def run():
    rows=[];start=time.time()
    for R in [24,40,96]:
      for m in [1,3]:
        ev=events(R,m=m)
        for label,t in [('h',ev['h']),('peak',ev['peak']),('other',.5)]:
          ref=signal(t,R,m=m,nx=72,nt=128)
          vals={f'{mod}_w{width}':direct(t,R,.35,m,width=width,model=mod) for mod,width in [('fresnel',1.),('fresnel',.5),('reduced_RS',.5)]}
          row=dict(R0=R,m=m,point=label,tau=t,compact=ref,values=vals,relative_errors={k:v/ref-1 for k,v in vals.items()});rows.append(row);print(row,flush=True)
          (ROOT/'data/direct_check.json').write_text(json.dumps(dict(rows=rows,seconds=time.time()-start),indent=2))
if __name__=='__main__':run()
