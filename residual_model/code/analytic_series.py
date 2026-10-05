"""Arbitrary finite epsilon jets of the compact identity, without fit data.
Cauchy coefficients in tau supply local derivatives for implicit event recursion.
"""
import math,json,time
from pathlib import Path
import numpy as np
from scipy.special import airy,jv,jnp_zeros
from scipy.optimize import brentq
from compact_airy import grid

def mul(a,b,N):
    shape=np.broadcast_shapes(a.shape[1:],b.shape[1:]);out=np.zeros((N+1,)+shape,complex)
    for n in range(N+1):
      for k in range(n+1):out[n]+=a[k]*b[n-k]
    return out

def expjet(a,N):
    out=np.zeros_like(a);out[0]=np.exp(a[0])
    for n in range(1,N+1):
      for k in range(1,n+1):out[n]+=k*a[k]*out[n-k]/n
    return out

def shift(a,k):
    out=np.zeros_like(a)
    if k<len(a):out[k:]=a[:-k] if k else a
    return out

def field(tau,eta,m,N=10,nx=32,nt=64):
    x,w,c,proj=grid(m,nx,nt);xc=x[:,None]*c[None,:]
    one=np.zeros((N+1,1,1),complex);one[0]=1
    inv=one.copy()
    for k in range(1,N//2+1):inv[2*k]=(-tau/2)**k
    a=shift(inv,1)*xc;in2=mul(inv,inv,N)
    arg=-a;arg[2]+=(-tau*tau/4+1j*eta*tau/2)
    X=-tau+1j*eta;ad=list(airy(X)[:2])
    for k in range(2,N+2):ad.append(X*ad[k-2]+(k-2)*ad[k-3] if k>2 else X*ad[0])
    A=np.zeros_like(a);Ap=np.zeros_like(a);term=one
    for k in range(N+1):
      if k:term=mul(term,arg,N)/k
      A+=ad[k]*term;Ap+=ad[k+1]*term
    E=-eta*shift(a,1)/2-1j*shift(mul(mul(a,a,N),inv,N),1)/4
    B=one-1j*eta*shift(inv,2)/2+shift(mul(a,in2,N),2)/2
    C=-1j*shift(inv,1)
    H=mul(mul(B,A,N)+mul(C,Ap,N),expjet(E,N),N)
    H*=np.exp(-1j*xc)[None,:,:]
    out=np.einsum("nxt,qt->nxq",H,proj,optimize=False)/nt
    if not np.isfinite(out).all():raise FloatingPointError("Nonfinite analytic field coefficient")
    return out

def qcoeff(tau,eta,m,N=10,nx=32,nt=64,deco=False):
    F=field(tau,eta,m,N,nx,nt);Fc=field(np.conj(tau),eta,m,N,nx,nt).conj()
    x,w,_,_=grid(m,nx,nt);b=jnp_zeros(m,1)[0];I0=2*m*jv(m,b)**2
    raw=np.zeros(N+1,complex);pairs={}
    for n in range(N+1):
      for k in range(n+1):
        v=np.sum(w*x*(F[k,:,0]*Fc[n-k,:,0]-F[k,:,1]*Fc[n-k,:,1]))/I0
        raw[n]+=v
        if deco:pairs[f'{k},{n-k}']=complex(v)
    V=np.zeros(N+1,complex);V[0]=1;V[2]=tau/2
    E=np.zeros(N+1,complex);E[1]=-2*eta*tau;E[3]=-eta*tau*tau/2
    pref=mul(V,expjet(E,N),N)
    out=mul(raw,pref,N)
    return (out,raw,pref,pairs) if deco else out

def local(m,t,eta,N=10,nx=32,nt=64,points=48,radius=.7):
    theta=2*np.pi*np.arange(points)/points
    v=np.array([qcoeff(t+radius*np.exp(1j*ph),eta,m,N,nx,nt) for ph in theta])
    # q[n,k] = coefficient of epsilon^n (tau-t)^k, by Cauchy integral.
    return (np.fft.fft(v,axis=0)[:N+2].T/(radius**np.arange(N+2))[None,:]).real/points

def evaljet(c,d,N,derivative=False):
    out=np.zeros(N+1);power=np.zeros(N+1);power[0]=1
    for k in range(N+1):
      if k:power=np.convolve(power,d)[:N+1]
      for n in range(N+1-k):
        factor=(k+1)*c[n,k+1] if derivative else c[n,k]
        out[n:]+=factor*power[:N+1-n]
    return out

def reference(eta):
    def f(t):return abs(airy(-t+1j*eta)[0])**2
    def fp(t):
      a,ap=airy(-t+1j*eta)[:2];return -2*np.real(a.conjugate()*ap)
    p=brentq(fp,.5,1.7,xtol=2e-14);h=brentq(lambda t:f(t)-.5*f(p),-1,.5,xtol=2e-14)
    return p,h

def event_coefficients(m,eta=.35,N=10,nx=32,nt=64,points=48,radius=.7):
    p,h=reference(eta);cp=local(m,p,eta,N,nx,nt,points,radius);ch=local(m,h,eta,N,nx,nt,points,radius)
    dp=np.zeros(N+1);dh=np.zeros(N+1)
    for n in range(1,N+1):dp[n]=-evaljet(cp,dp,N,True)[n]/(2*cp[0,2])
    peak=evaljet(cp,dp,N)
    for n in range(1,N+1):dh[n]=(.5*peak[n]-evaljet(ch,dh,N)[n])/ch[0,1]
    return dict(m=m,eta=eta,N=N,nx=nx,nt=nt,points=points,radius=radius,p0=p,h0=h,dp=dp.tolist(),dh=dh.tolist(),cp=cp.tolist(),ch=ch.tolist())

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];start=time.time();rows=[]
    for cfg in [dict(N=10,nx=32,nt=64,points=48,radius=.7),dict(N=10,nx=48,nt=96,points=64,radius=.8)]:
      evs={str(m):event_coefficients(m,**cfg) for m in [1,3]};delta=np.array(evs['1']['dh'])-evs['3']['dh'];rows.append(dict(config=cfg,events=evs,delta=delta.tolist()))
      print('COEFFICIENTS',cfg,delta,flush=True)
      (root/'data/analytic_coefficients.json').write_text(json.dumps(dict(rows=rows,seconds=time.time()-start,reads_propagation_data=False),indent=2))
