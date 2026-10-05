from pathlib import Path
import json,numpy as np
from scipy.special import jv,jvp,jnp_zeros,airy
from scipy.integrate import quad
from formal_asymptotics import signal_coefficients
OUT=Path(__file__).resolve().parents[1];rows=[]
for m in [1,2,3,4,6,8,12,16]:
 b=float(jnp_zeros(m,1)[0]);I0=2*m*jv(m,b)**2;h=lambda q,x:.5*jv(q,x)+x*jvp(q,x);dd=lambda q,x:q*q-x*x-.25
 funcs=[lambda x:x*(jv(m-1,x)**2-jv(m+1,x)**2),lambda x:x*(jv(m-1,x)*h(m-1,x)-jv(m+1,x)*h(m+1,x)),lambda x:x*(h(m-1,x)**2-h(m+1,x)**2),lambda x:x*(jv(m-1,x)**2*dd(m-1,x)-jv(m+1,x)**2*dd(m+1,x))];v=np.array([quad(f,0,b,epsabs=1e-11,epsrel=1e-11)[0] for f in funcs])/I0;target=np.array([1,-.5,.25,m*m+.75-b*b]);rows.append({'m':m,'normalized_integrals':v.tolist(),'identity_absolute_errors':(v-target).tolist()})
corr=[]
for eta in [.35,.7,.9]:
 for tau in [-.5,0,.5,1.]:
  A,Ap=airy(-tau+1j*eta)[:2];prediction=-np.real(A.conjugate()*Ap)/8;diff=[]
  for m in [1,2,4,6]:
   b=float(jnp_zeros(m,1)[0]);I0=2*m*jv(m,b)**2;a=signal_coefficients(m,tau,eta,nx=1201);b=signal_coefficients(m,tau,eta,nx=1201,source_saddle_correction=True);diff.append((b-a)/I0)
  diff=np.array(diff);corr.append({'eta':eta,'tau':tau,'predicted_common_fourth_signal_term':float(prediction),'per_order_signal_changes':diff.tolist(),'max_order_spread_eps4':float(np.ptp(diff[:,4])),'max_error_vs_common_term':float(np.max(abs(diff[:,4]-prediction)))})
# Differentiate sqrt(u)Jq(uR) independently by a five-point finite difference.
der=[]
for q in [0,1,2,5,7,17]:
 u=5.;R=.7;x=u*R;f=lambda z:np.sqrt(z)*jv(q,z*R);step=.002;first=(f(u-2*step)-8*f(u-step)+8*f(u+step)-f(u+2*step))/(12*step);second=(-f(u+2*step)+16*f(u+step)-30*f(u)+16*f(u-step)-f(u-2*step))/(12*step**2);pred1=u**(-.5)*(.5*jv(q,x)+x*jvp(q,x));pred2=u**(-1.5)*(q*q-x*x-.25)*jv(q,x);der.append({'q':q,'first_error':float(first-pred1),'second_error':float(second-pred2)})
report={'Bessel_domain_identities':rows,'F_derivative_checks':der,'source_Airy_moment_correction':corr,'meaning':'next radial saddle-amplitude term changes the common eps4 signal but not order differences through eps4; preregistered advance coefficients remain untouched'};(OUT/'theory/equation_algebra_checks.json').write_text(json.dumps(report,indent=2));print('max Bessel integral error',max(abs(v) for r in rows for v in r['identity_absolute_errors']));print('source common eps4 max spread',max(r['max_order_spread_eps4'] for r in corr),'error',max(r['max_error_vs_common_term'] for r in corr))
