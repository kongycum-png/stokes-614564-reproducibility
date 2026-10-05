"""Independent roots and event coefficients for P=2,3; no propagation fit."""
from pathlib import Path
import json,datetime
import numpy as np
from scipy.special import jv,jvp,airy
from scipy.integrate import quad
from scipy.optimize import brentq
from formal_asymptotics import signal_coefficients
OUT=Path(__file__).resolve().parents[1];pre=json.loads((OUT/'theory/analytic_preregistration.json').read_text())
def domain(m,P):
 a=abs(m-P);b=abs(m+P)
 if a==b:return {'status':'identically_zero_S3','root':None}
 x=np.linspace(.001,b+12,30001);v=jv(a,x)**2-jv(b,x)**2;ids=np.where((v[:-1]>0)&(v[1:]<0))[0]
 if not len(ids):return {'status':'no_root_in_search_interval','root':None}
 root=brentq(lambda t:jv(a,t)**2-jv(b,t)**2,x[ids[0]],x[ids[0]+1]);l=lambda q,t:jv(q,t)+t*jvp(q,t);n=lambda q,t:(q*q-t*t+.5)*jv(q,t)+t*jvp(q,t)
 I0=quad(lambda t:t*(jv(a,t)**2-jv(b,t)**2),0,root,epsabs=1e-12)[0]
 L=quad(lambda t:t*(l(a,t)**2-l(b,t)**2),0,root,epsabs=1e-12)[0]/I0
 N=quad(lambda t:t*(jv(a,t)*n(a,t)-jv(b,t)*n(b,t)),0,root,epsabs=1e-12)[0]/I0
 H0p=2*root*(jv(a,root)*jvp(a,root)-jv(b,root)*jvp(b,root))
 return {'status':'simple_positive_to_negative_root','m':m,'P':P,'qplus':m-P,'qminus':m+P,'root':root,'I0':I0,'L':L,'N':N,'H0_derivative':H0p,'actual_root_v2_coefficient':-root**2*H0p/(2*I0)}
if __name__=='__main__':
 result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_P_propagation_seen':False,'formula':'Q/I0=common+epsilon^2*(N*tau*f+L*|Ai_prime|^2); advance=A/R0','status':'FORMAL_INWARD_BRANCH_PREDICTIONS','domains':{},'predictions':[],'algebra_checks':[]}
 for P in [2,3]:
  ms=sorted(set([P,P+1,P+2,P+4]+list(range(1,P))));domains={m:domain(m,P) for m in ms};result['domains'][str(P)]={str(m):v for m,v in domains.items()}
  for eta in [.35,.7]:
   ev=pre['airy_events'][str(eta)];p=ev['peak'];h=ev['thresholds']['0.5']['h'];fp=-2*np.real(airy(-h+1j*eta)[0].conjugate()*airy(-h+1j*eta)[1]);K=ev['thresholds']['0.5']['K_proxy'];KL=(.5*abs(airy(-p+1j*eta)[1])**2-abs(airy(-h+1j*eta)[1])**2)/fp
   for m,r in domains.items():
    A=K*(domains[P]['N']-r['N'])+KL*(domains[P]['L']-r['L']);result['predictions'].append({'P':P,'m':m,'reference_m':P,'eta':eta,'A':A,'K':K,'K_L':KL})
    for t in [-.5,0,.5,1.]:
     c=signal_coefficients(m,t,eta,b=r['root'],P=P,nx=1201)/r['I0'];Ai,Ap=airy(-t+1j*eta)[:2];f=abs(Ai)**2;X2=-t*t/4+1j*eta*t/2;pred=np.array([f,-2*eta*t*f,(r['N']*t+2*eta**2*t*t)*f+r['L']*abs(Ap)**2+2*np.real(Ai.conjugate()*Ap*X2)])
     result['algebra_checks'].append({'P':P,'m':m,'eta':eta,'tau':t,'max_loworder_error':float(np.max(abs(c[:3]-pred)))})
 (OUT/'theory/general_P_preregistration.json').write_text(json.dumps(result,indent=2));print('max loworder error',max(r['max_loworder_error'] for r in result['algebra_checks']))
 for P,rows in result['domains'].items():
  for m,row in rows.items():print(P,m,'root',row['root'],'L',row['L'],'N',row['N'])
