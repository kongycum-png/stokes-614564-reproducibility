"""Full-line Airy/Fresnel identity followed by exact angular Fourier projection.
New implementation; no propagation or fitted data are read.
Positive-radius source differs by the explicitly bounded negative-source integral.
"""
import numpy as np
from scipy.special import airy, roots_legendre, jnp_zeros, jv
from scipy.optimize import minimize_scalar, brentq
from functools import lru_cache

@lru_cache(None)
def grid(m,nx,nt):
    t,w=roots_legendre(nx);b=jnp_zeros(m,1)[0];x=(t+1)*b/2;w=w*b/2
    th=2*np.pi*np.arange(nt)/nt
    return x,w,np.cos(th),np.exp(1j*np.outer([m-1,m+1],th))

def signal(tau,R0,eta=.35,m=3,nx=48,nt=64,parts=None):
    e=R0**-.5;V=1+tau*e*e/2
    x,w,c,proj=grid(m,nx,nt);a=e*x[:,None]*c[None,:]/V
    X=-tau+1j*eta+e*e*(-tau*tau/4+1j*eta*tau/2)
    A,Ap=airy(X-a)[:2]
    G=np.exp(-1j*x[:,None]*c[None,:]-eta*e*a/2-1j*e*a*a/(4*V))
    H=G*((1-1j*eta*e*e/(2*V)+e*e*a/(2*V*V))*A-1j*e/V*Ap)
    F=H@proj.T/nt
    q=np.sum(w*x*(abs(F[:,0])**2-abs(F[:,1])**2))
    I0=2*m*jv(m,jnp_zeros(m,1)[0])**2
    return float(q/I0*V*np.exp(-2*eta*tau*e-eta*tau*tau*e**3/2))

def events(R0,eta=.35,m=3,nx=48,nt=64,gamma=.5):
    fun=lambda t:signal(t,R0,eta,m,nx,nt)
    scan=np.linspace(-3,5,161);v=np.array([fun(t) for t in scan])
    ids=np.where((v[1:-1]>v[:-2])&(v[1:-1]>v[2:]))[0]+1
    ps=[minimize_scalar(lambda t:-fun(t),bounds=(scan[i-1],scan[i+1]),method='bounded',options={'xatol':1e-13}).x for i in ids]
    if not ps:raise ValueError('No maximum')
    p=ps[0];peak=fun(p)
    ids=np.where((v[:-1]<gamma*peak)&(v[1:]>gamma*peak)&(scan[1:]<p))[0]
    j=ids[-1];h=brentq(lambda t:fun(t)-gamma*peak,scan[j],scan[j+1],xtol=2e-14)
    step=1e-4
    return dict(R0=R0,eta=eta,m=m,peak=p,h=h,value=peak,all_maxima=ps,slope=(fun(h+step)-fun(h-step))/(2*step),curvature=(fun(p+step)-2*peak+fun(p-step))/step**2)

if __name__=='__main__':
    import json,time
    rows=[];start=time.time()
    for R in [24,40,96]:
        a=events(R,m=1);b=events(R,m=3);row=dict(R0=R,reference=a,target=b,advance=a['h']-b['h'],scaled_advance=R*(a['h']-b['h']))
        rows.append(row);print(row,flush=True)
    from pathlib import Path
    Path(__file__).resolve().parents[1].joinpath('data/compact_pilot.json').write_text(json.dumps(dict(rows=rows,seconds=time.time()-start),indent=2))
