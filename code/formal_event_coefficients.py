from pathlib import Path
import json,math
import numpy as np
from numpy.polynomial import Chebyshev,Polynomial
from scipy.special import jnp_zeros,jv,airy
from scipy.integrate import simpson
from scipy.optimize import brentq
from formal_asymptotics import signal_coefficients
OUT=Path(__file__).resolve().parents[1]

def local_taylor(m,t,eta,points=21,radius=.3,nx=801,observable='Q'):
 nodes=t+radius*np.cos(np.pi*(np.arange(points)+.5)/points)
 b=jnp_zeros(m,1)[0]
 if observable=='Q':normalization=2*m*jv(m,b)**2
 elif observable=='T':
  x=np.linspace(0,b,nx);normalization=simpson(x*(jv(m-1,x)**2+jv(m+1,x)**2),x=x)
 else:raise ValueError("observable must be 'Q' or 'T'")
 values=np.array([signal_coefficients(m,u,eta,nx=nx,observable=observable)/normalization for u in nodes])
 co=[]
 for n in range(5):
  ch=Chebyshev.fit(nodes,values[:,n],points-1,domain=[t-radius,t+radius])
  co.append([float(ch.deriv(k)(t))/math.factorial(k) for k in range(6)])
 return np.array(co)
def series_value(c,d,derivative=False):
 out=np.zeros(5);powd=np.array([1.,0,0,0,0])
 for k in range(5):
  if k:powd=np.convolve(powd,d)[:5]
  for n in range(5-k):
   factor=(k+1)*c[n,k+1] if derivative else c[n,k]
   out[n:]+=factor*powd[:5-n]
 return out
def airy_reference_event(eta,gamma=.5,window=(-4.,6.),samples=20001):
 nodes=np.linspace(window[0],window[1],samples)
 def profile(t):
  A,Ap=airy(-t+1j*eta)[:2];return float(abs(A)**2),float(-2*np.real(A.conjugate()*Ap))
 derivative=np.array([profile(t)[1] for t in nodes])
 brackets=[(a,b) for a,b,fa,fb in zip(nodes[:-1],nodes[1:],derivative[:-1],derivative[1:]) if fa>0 and fb<0]
 if not brackets:raise ValueError(f'No leading Airy maximum for eta={eta} in {window}')
 p=brentq(lambda t:profile(t)[1],*brackets[0],xtol=1e-13);peak=profile(p)[0]
 values=np.array([profile(t)[0]-gamma*peak for t in nodes])
 rises=[(a,b) for a,b,fa,fb in zip(nodes[:-1],nodes[1:],values[:-1],values[1:]) if b<p and fa<0 and fb>0]
 if not rises:raise ValueError(f'No rising gamma={gamma} crossing for eta={eta}')
 h=brentq(lambda t:profile(t)[0]-gamma*peak,*rises[-1],xtol=1e-13)
 return {'peak':p,'crossing':h,'peak_value':peak,'crossing_slope':profile(h)[1],'gamma':gamma}

def coefficients(m,eta,points=21,nx=801,gamma=.5,event_record=None,observable='Q'):
 ev=airy_reference_event(eta,gamma) if event_record is None else event_record;p=ev['peak'];h=ev.get('crossing',ev.get('h'));cp=local_taylor(m,p,eta,points=points,nx=nx,observable=observable);ch=local_taylor(m,h,eta,points=points,nx=nx,observable=observable)
 dp=np.zeros(5)
 for n in range(1,5):dp[n]=-series_value(cp,dp,True)[n]/(2*cp[0,2])
 peak=series_value(cp,dp);dh=np.zeros(5)
 for n in range(1,5):dh[n]=(gamma*peak[n]-series_value(ch,dh)[n])/ch[0,1]
 return {'p0':p,'h0':h,'gamma':gamma,'observable':observable,'peak_coefficients':dp.tolist(),'half_height_coefficients':dh.tolist()}
if __name__=='__main__':
 pr=json.loads((OUT/'theory/analytic_preregistration.json').read_text())
 checks=[]
 for m in [1,2,3,4,6]:
  b=jnp_zeros(m,1)[0];I0=2*m*jv(m,b)**2;D=m*m+.75-b*b
  for t in [-.5,0.,.5,1.]:
   eta=.7;c=signal_coefficients(m,t,eta)/I0;A,Ap=airy(-t+1j*eta)[:2];f=abs(A)**2;X2=-t*t/4+1j*eta*t/2
   pred=[f,-2*eta*t*f,((D-.25)*t+2*eta**2*t*t)*f+2*np.real(A.conjugate()*Ap*X2)]
   checks.append({'m':m,'tau':t,'coefficients':c.tolist(),'low_order_errors':(c[:3]-pred).tolist()})
 (OUT/'theory/formal_expansion_checks.json').write_text(json.dumps(checks,indent=2));print('low_order_max_error',max(abs(v) for r in checks for v in r['low_order_errors']),flush=True)
 result={}
 for eta in [.2,.35,.7,.9]:
  ev={'peak':pr['airy_events'][str(eta)]['peak'],'crossing':pr['airy_events'][str(eta)]['thresholds']['0.5']['h']}
  rows={str(m):coefficients(m,eta,event_record=ev) for m in range(1,7)};h1=np.array(rows['1']['half_height_coefficients'])
  for m in range(2,7):
   delta=h1-np.array(rows[str(m)]['half_height_coefficients']);A=pr['airy_events'][str(eta)]['thresholds']['0.5']['K_proxy']*(pr['orders'][0]['D']-pr['orders'][m-1]['D']);rows[str(m)].update(advance_coefficients=delta.tolist(),A_preregistered=A,A_check_error=float(delta[2]-A))
  # quadrature/differentiation sensitivity for a representative high-order coefficient
  repeat=coefficients(4,eta,points=25,nx=1201,event_record=ev)
  rows['derivative_quadrature_check_m4']=(np.array(repeat['half_height_coefficients'])-rows['4']['half_height_coefficients']).tolist()
  result[str(eta)]=rows;(OUT/'theory/formal_event_coefficients.json').write_text(json.dumps({'status':'FORMAL_INWARD_BRANCH_COEFFICIENTS_NOT_FIT_TO_PROPAGATION','terms':['epsilon^0','epsilon','epsilon^2','epsilon^3','epsilon^4'],'outward_and_endpoint_not_included':True,'results':result},indent=2));print(eta,rows['4']['advance_coefficients'],flush=True)
