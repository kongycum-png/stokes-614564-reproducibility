"""Compute analytic predictions before propagation beyond original m=1..3."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.special import airy,jnp_zeros,jv,jvp
from scipy.integrate import quad
from scipy.optimize import brentq
OUT=Path(__file__).resolve().parents[1]
def f(t,e): return abs(airy(-t+1j*e)[0])**2
def fp(t,e):
 a,b=airy(-t+1j*e)[:2];return -2*np.real(a.conjugate()*b)
def events(e):
 grid=np.linspace(-6,6,2401); d=np.array([fp(t,e) for t in grid]); inds=np.where((d[:-1]>0)&(d[1:]<0))[0]
 if not len(inds):return {'status':'no_peak_in_valid_interval','window':[-6,6]}
 p=brentq(lambda t:fp(t,e),grid[inds[0]],grid[inds[0]+1]); out={'status':'defined','peak':p,'thresholds':{}}
 for g in np.arange(1,10)/10:
  h=brentq(lambda t:f(t,e)-g*f(p,e),-12,p)
  v=lambda t:float(np.imag(airy(-t+1j*e)[1]/airy(-t+1j*e)[0]))
  K=g*f(p,e)*(p-h)/fp(h,e)
  W=lambda t:t-v(t)**2
  Kr=g*f(p,e)*(W(p)-W(h))/fp(h,e)
  out['thresholds'][f'{g:.1f}']={'h':h,'K_proxy':K,'K_actual_root_candidate':Kr}
 return out
orders=[]
for m in range(1,17):
 b=float(jnp_zeros(m,1)[0]); D=m*m+.75-b*b; j=lambda x:jv(m,x)
 den=quad(lambda r:r*j(b*r)**2,0,1,epsabs=1e-13)[0]
 num=quad(lambda r:j(b*r)**2/r,0,1,epsabs=1e-13)[0]
 orders.append({'m':m,'b0':b,'D':D,'Dprime_Hellmann_Feynman':-2*m*(num/den-1)})
eta=[.2,.3,.35,.4,.5,.6,.7,.8,.9,1.]
profiles={str(e):events(e) for e in eta}
m4=[]
for e,ev in profiles.items():
 if ev['status']=='defined':
  K=ev['thresholds']['0.5']['K_proxy'];A=K*(orders[0]['D']-orders[3]['D'])
  for R in [16,24,32,40,56,72,96,128,192,256]:m4.append({'eta':float(e),'R0':R,'m':4,'A':A,'predicted_Delta_tau50':A/R})
checks=[]
for m in range(1,7):
 b=orders[m-1]['b0'];J=lambda q,x:jv(q,x);h=lambda q,x:.5*J(q,x)+x*jvp(q,x);d=lambda q,x:q*q-x*x-.25
 for scale in [.8,.9,1.,1.1,1.2]:
  s=scale*b;I0=quad(lambda x:x*(J(m-1,x)**2-J(m+1,x)**2),0,s,epsabs=1e-12)[0]
  I1=quad(lambda x:x*(J(m-1,x)*h(m-1,x)-J(m+1,x)*h(m+1,x)),0,s,epsabs=1e-12)[0]
  I11=quad(lambda x:x*(h(m-1,x)**2-h(m+1,x)**2),0,s,epsabs=1e-12)[0]
  I2=quad(lambda x:x*(J(m-1,x)**2*d(m-1,x)-J(m+1,x)**2*d(m+1,x)),0,s,epsabs=1e-12)[0]
  B=J(m-1,s)**2-J(m+1,s)**2
  checks.append({'m':m,'b_factor':scale,'I0':I0,'I0_identity_error':I0-2*m*J(m,s)**2,'I1':I1,'I1_identity_error':I1+.5*I0-.5*s*s*B,'I11':I11,'I2':I2,'full_first_ratio':I1/I0+.5,'full_first_square_ratio':I11/I0+I1/I0+.25,'full_second_ratio':I2/I0+I1/I0+.25})
result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PREREGISTERED_ANALYTIC_PREDICTIONS_NOT_PROPAGATION_VALIDATED','preexisting_m1_to_m3_results_seen':True,'new_m4_propagation_seen':False,'formula':'A=K_eta*(D1-D4); Dm=m^2+3/4-jprime(m,1)^2','no_fitted_coefficients':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'orders':orders,'airy_events':profiles,'m4_predictions':m4,'bessel_integral_checks':checks,'double_ratio_protocol':{'tau_star':0.,'window':[-.5,.5],'samples':51,'positive_floor_relative_to_peak':1e-8,'invalid_case':'report undefined; do not adjust window'},'root_prediction_status':'candidate from full O(epsilon) amplitude, awaiting direct independent verification'}
p=OUT/'theory/analytic_preregistration.json'
if p.exists():raise RuntimeError('Preserve preregistration: file already exists')
p.write_text(json.dumps(result,indent=2));print('registered',result['timestamp_utc']);print('D4=',orders[3]['D']);print([(e,profiles[str(e)]['thresholds']['0.5']) for e in [.35,.7]])
