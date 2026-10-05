"""mpmath arbitrary precision, angular moments by Bessel derivatives.
Independent angular evaluator; does not use angular quadrature or numpy Airy.
"""
import json,time,math
from pathlib import Path
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]

def signal_mp(tau,R0,m,eta='0.35',nx=24,K=44,dps=40):
    mp.mp.dps=dps;tau=mp.mpf(str(tau));R=mp.mpf(str(R0));eta=mp.mpf(eta)
    e=1/mp.sqrt(R);z=2/e+tau*e;alpha=eta*e/2
    b=mp.besseljzero(m,1,derivative=1);I0=2*m*mp.besselj(m,b)**2
    X=-tau-tau*tau/(4*R)+1j*alpha*z
    ad=[mp.airyai(X),mp.airyai(X,1)]
    for k in range(2,K+2):ad.append(X*ad[k-2]+((k-2)*ad[k-3] if k>2 else 0))
    nodes,weights=mp.gauss_quadrature(nx,'legendre');total=mp.mpf(0)
    for v,w in zip(nodes,weights):
      x=(v+1)*b/2;r=2*x/z
      a=[(-r)**k*ad[k]/mp.factorial(k) for k in range(K+1)]
      ap=[(-r)**k*ad[k+1]/mp.factorial(k) for k in range(K+1)]
      B=1-2j*alpha/z;C=-2j/z
      f=[B*a[k]+C*ap[k]+(2*r/z**2*a[k-1] if k else 0) for k in range(K+1)]
      L=-alpha*r;M=-1j*r*r/(2*z);g=[mp.mpc(1)]
      for k in range(1,K+1):g.append((L*g[k-1]+(2*M*g[k-2] if k>1 else 0))/k)
      h=[sum(f[j]*g[k-j] for j in range(k+1)) for k in range(K+1)]
      js={n:mp.besselj(n,x) for n in range(-K,m+K+2)};fields=[]
      for q in [m-1,m+1]:
        moments=[1j**k*(-1j)**q*sum((-1)**j*math.comb(k,j)*js[q-k+2*j] for j in range(k+1))/2**k for k in range(K+1)]
        fields.append(sum(h[k]*moments[k] for k in range(K+1)))
      total+=w*b/2*x*(abs(fields[0])**2-abs(fields[1])**2)
    return total/I0*(1+tau/(2*R))*mp.exp(-2*eta*tau*e-eta*tau*tau*e**3/2)

def main():
    from compact_airy import events,signal
    rows=[];start=time.time()
    for R in [24,40,96]:
      for m in [1,3]:
        ev=events(R,m=m)
        for where,tau in [('h',ev['h']),('p',ev['peak'])]:
          vals={}
          for nx,K,dp in [(20,36,35),(28,48,50)]:
            vals[f'nx{nx}_K{K}_dps{dp}']=mp.nstr(signal_mp(tau,R,m,nx=nx,K=K,dps=dp),50)
          row=dict(R0=R,m=m,point=where,tau=float(tau),numpy_value=signal(tau,R,m=m,nx=64,nt=128),mp_values=vals);rows.append(row);print(row,flush=True)
          (ROOT/'data/high_precision.json').write_text(json.dumps(dict(rows=rows,seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
