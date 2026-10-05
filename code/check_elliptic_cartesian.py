"""Independent 2D Fresnel quadrature, without angular modal propagation."""
from pathlib import Path
import json,numpy as np
from scipy.special import airy
from scipy.interpolate import CubicSpline
OUT=Path(__file__).resolve().parents[1];R=24;eta=.7;alpha=eta/(2*np.sqrt(R));e=.01;a=np.exp(e/2);d=np.load(OUT/'data/input_2d/ellipticity/ellipticity0.01_mode_fields.npz');records=[]
for ds,nphi in [(.025,256),(.025,512),(.0125,512)]:
 smax=R+24/alpha;ns=int(np.ceil(smax/ds))+1;ns+=1-ns%2;s=np.linspace(0,smax,ns);sw=np.ones(ns);sw[1:-1:2]=4;sw[2:-1:2]=2;sw*=s[1]/3;phi=2*np.pi*np.arange(nphi)/nphi;scale=np.sqrt(np.cos(phi)**2/a**2+a*a*np.sin(phi)**2);theta0=np.arctan2(a*np.sin(phi),np.cos(phi)/a)
 for tau,x,theta in [(0.,1.2,.23),(0.,3.,1.12),(1.,6.,2.1)]:
  zeta=2*np.sqrt(R)+tau/np.sqrt(R);r=x/np.sqrt(R);actual=np.zeros(4,complex)
  for j in range(0,ns,256):
   ss=s[j:j+256];arg=R-ss[:,None]*scale[None,:];f=airy(arg)[0]*np.exp(alpha*arg);phase=np.exp(1j*(ss[:,None]**2/(2*zeta)-ss[:,None]*r/zeta*np.cos(phi-theta)[None,:]));op=f*phase;weight=sw[j:j+256]*ss
   for iq,q in enumerate([0,2,3,5]):actual[iq]+=np.sum(weight*np.mean(op*np.exp(1j*q*theta0)[None,:],axis=1))
  actual*=(-1j/zeta)*np.exp(.5j*r*r/zeta);it=int(np.argmin(abs(d['tau']-tau)));target=[]
  for q in [0,2,3,5]:
   modes=d[f'q{q}_orders'];value=CubicSpline(d['rho_w'],d[f'q{q}_U'][it],axis=1)(r);target.append(np.sum(value*np.exp(1j*modes*theta)))
  target=np.array(target);records.append({'ds':ds,'nphi':nphi,'tau':tau,'x':x,'theta':theta,'relative_channel_L2_error':float(np.linalg.norm(actual-target)/np.linalg.norm(actual)),'actual_real':actual.real.tolist(),'actual_imag':actual.imag.tolist()})
  print(ds,nphi,tau,x,records[-1]['relative_channel_L2_error'],flush=True)
(OUT/'baseline/elliptic_cartesian_check.json').write_text(json.dumps({'comparison':'direct two-dimensional Fresnel integral versus stored signed angular modes; includes stored radial interpolation','records':records},indent=2))
