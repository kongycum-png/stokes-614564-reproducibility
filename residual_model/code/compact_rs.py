"""Compact full-line identity for the reduced R-S kernel, with exact D geometry.
General formula independent of epsilon-jet implementation.
"""
import numpy as np
from scipy.special import airy,jv,jnp_zeros
from compact_airy import grid
KW=2*np.pi*.08/.0006328

def signal_rs(tau,R0,eta=.35,m=3,nx=48,nt=64,kw=KW,ablation='full'):
    e=R0**-.5;z=2/e+tau*e;alpha=eta*e/2
    x,w,c,proj=grid(m,nx,nt);r=2*x/z;d=np.sqrt(z*z+(r/kw)**2)
    a=r[:,None]*c[None,:]
    X=(-tau-tau*tau/(4*R0)-(r/kw)**2/4+1j*alpha*d)[:,None]
    A,Ap=airy(X-a)[:2]
    G=np.exp(-1j*d[:,None]*a/2-alpha*a-1j*a*a/(2*d[:,None]))
    H=G*((1-2j*alpha/d[:,None]+2*a/d[:,None]**2)*A-2j/d[:,None]*Ap)
    F=H@proj.T/nt
    pref=d*e/2*np.exp(-2*eta*tau*e-eta*tau*tau*e**3/2-alpha*(r/kw)**2)
    I0=2*m*jv(m,jnp_zeros(m,1)[0])**2
    return float(np.sum(w*x*pref*(abs(F[:,0])**2-abs(F[:,1])**2))/I0)
