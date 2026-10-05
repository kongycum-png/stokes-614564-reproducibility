"""Formal O(epsilon^4) full-amplitude fold expansion. No fitted coefficients.
Bivariate polynomials (epsilon,t); t monomials integrated by Airy moments.
Includes complex u0,X, full g, first Debye term and all intensity cross terms.
Excludes the outward Hankel branch and spectral endpoint; tracked remainder.
"""
import math
import numpy as np
from scipy.special import airy,jvp
from scipy.integrate import simpson
N=4

def add(a,b):
 c=dict(a)
 for k,v in b.items():c[k]=c.get(k,0)+v
 return c

def scale(a,s):return {k:v*s for k,v in a.items()}
def mul(a,b):
 c={}
 for (i,j),v in a.items():
  for (k,l),w in b.items():
   if i+k<=N:c[i+k,j+l]=c.get((i+k,j+l),0)+v*w
 return c

def power(a,p):
 a0=a[(0,0)];u=add(scale(a,1/a0),{(0,0):-1});c={(0,0):1};term={(0,0):1};coef=1
 for k in range(1,N+1):
  term=mul(term,u);coef*= (p-k+1)/k;c=add(c,scale(term,coef))
 return scale(c,a0**p)

def airy_derivs(z,n):
 a,b=airy(z)[:2];vals=[a,b]
 for k in range(2,n+1):vals.append(z*vals[k-2]+((k-2)*vals[k-3] if k>2 else 0))
 return vals

def field_coefficients(q,x,tau,eta,source_saddle_correction=False):
 U={(0,0):1.,(1,1):1.,(2,0):(tau-1j*eta)/2}
 V={(0,0):1.,(2,0):tau/2}
 W={(0,0):1.,(1,1):1.,(2,0):tau/2}
 S=add({(0,0):1},mul(W,W));g=power(scale(mul(U,S),.5),.5)
 delta=scale(add(mul(U,power(V,-1)),{(0,0):-1}),x)
 bessel={};term={(0,0):np.ones_like(x)}
 for k in range(5):
  if k:term=mul(term,delta)
  bessel=add(bessel,scale(term,jvp(q,x,k)/math.factorial(k)))
 debye=add({(0,0):1.},mul({(3,0):1j*(4*q*q-1)/8},power(mul(U,S),-1)))
 poly=mul(mul(g,bessel),debye)
 if source_saddle_correction:
  correction=add({(0,0):1.},mul({(3,0):.25j},mul(W,power(S,-2))))
  poly=mul(poly,correction)
 X=-tau+1j*eta;X2=-tau*tau/4+1j*eta*tau/2;ad=airy_derivs(X,8)
 h=np.zeros((5,len(x)),complex)
 for (i,j),v in poly.items():
  for k in range((4-i)//2+1):h[i+2*k]+=v*(-1j)**j*ad[j+k]*X2**k/math.factorial(k)
 return h

def signal_coefficients(m,tau,eta,b=None,nx=801,P=1,source_saddle_correction=False,observable='Q'):
 from scipy.special import jnp_zeros
 if b is None:b=jnp_zeros(m,1)[0]
 x=np.linspace(0,b,nx);hp=field_coefficients(m-P,x,tau,eta,source_saddle_correction);hm=field_coefficients(m+P,x,tau,eta,source_saddle_correction)
 if observable not in ('Q','T'):raise ValueError("observable must be 'Q' or 'T'")
 channel_sign=-1 if observable=='Q' else 1
 intensity=np.zeros((5,nx))
 for n in range(5):
  for i in range(n+1):intensity[n]+=np.real(hp[i]*hp[n-i].conjugate()+channel_sign*hm[i]*hm[n-i].conjugate())
 raw=simpson(intensity*x[None,:],x=x,axis=1)
 # tau-dependent modulus of the full caustic prefactor and radial Jacobian
 a=-2*eta*tau;bb=-.5*eta*tau*tau
 expc=np.array([1,a,a*a/2,a**3/6+bb,a**4/24+a*bb]);jac=np.array([1,0,-tau,0,3*tau*tau/4])
 pref=np.convolve(expc,jac)[:5];return np.convolve(raw,pref)[:5]
