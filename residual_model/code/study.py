"""New radius controls, event branches, no-fit series and ablations."""
import json,time,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar,brentq
from compact_airy import signal
from compact_rs import signal_rs
ROOT=Path(__file__).resolve().parents[1]
RADIUS=[16,20,24,28,32,36,40,44,48,56,64,80,96,128,192,256]
# Newly computed controls not present in old principal grid: 20,28,36,44,48,64,80.
def event(fun,scan=True):
    if scan:
      t=np.linspace(-3,5,161);v=np.array([fun(a) for a in t]);ids=np.where((v[1:-1]>v[:-2])&(v[1:-1]>v[2:]))[0]+1
      brackets=[(float(t[i-1]),float(t[i+1])) for i in ids]
    else:brackets=[(.3,1.5)]
    maxima=[float(minimize_scalar(lambda t:-fun(t),bounds=b,method='bounded',options={'xatol':1e-13}).x) for b in brackets]
    p=maxima[0];q=fun(p);h=brentq(lambda t:fun(t)-.5*q,-1.,p,xtol=2e-14)
    d=2e-4;slope=(fun(h-2*d)-8*fun(h-d)+8*fun(h+d)-fun(h+2*d))/(12*d)
    curv=(-fun(p+2*d)+16*fun(p+d)-30*q+16*fun(p-d)-fun(p-2*d))/(12*d*d)
    return dict(h=h,p=p,q=q,slope=slope,curvature=curv,maxima=maxima)

def advance(R,model='rs',nx=32,nt=64,scan=False):
    f=signal_rs if model=='rs' else signal
    es=[event(lambda t:f(t,R,m=m,nx=nx,nt=nt),scan) for m in [1,3]]
    return dict(R0=float(R),model=model,nx=nx,nt=nt,reference=es[0],target=es[1],advance=es[0]['h']-es[1]['h'],scaled=R*(es[0]['h']-es[1]['h']))

def main():
    start=time.time();rows=[]
    coeff=json.loads((ROOT/'data/analytic_coefficients.json').read_text())['rows'][-1]['delta']
    for R in RADIUS:
      row=advance(R,scan=True);row['series']={str(N):float(np.dot(coeff[2:N+1],float(R)**(1-np.arange(2,N+1)/2))) for N in [2,3,4,5,6,8,10]}
      rows.append(row);print('RADIUS',R,row['scaled'],row['series'],flush=True)
      (ROOT/'data/radius_study.json').write_text(json.dumps(dict(rows=rows,seconds=time.time()-start),indent=2))
    opt=minimize_scalar(lambda R:advance(R)['scaled'],bounds=(24.,56.),method='bounded',options={'xatol':2e-4})
    minima={'rs':dict(R0=float(opt.x),scaled=float(opt.fun))}
    for N in [4,5,6,8,10]:
      def f(R):return float(np.dot(coeff[2:N+1],R**(1-np.arange(2,N+1)/2)))
      op=minimize_scalar(f,bounds=(10.,96.),method='bounded');minima[str(N)]=dict(R0=float(op.x),scaled=float(op.fun))
    (ROOT/'data/minima.json').write_text(json.dumps(minima,indent=2));print('MINIMA',minima,flush=True)
    controls=[]
    for R in [24,40,96]:
      for model,nx,nt in [('fresnel',48,96),('rs',48,96),('rs',72,128)]:
        row=advance(R,model,nx,nt);controls.append(row);print('CONTROL',row,flush=True)
        (ROOT/'data/compact_convergence.json').write_text(json.dumps(controls,indent=2))
    # Freeze-model peak normalization and fixed crossing diagnose nonlinear event map.
    freeze=[]
    for R in [24,40,96]:
      e=advance(R);p1=e['reference']['p'];p3=e['target']['p'];h1=e['reference']['h']
      target=lambda t:signal_rs(t,R,m=3)
      hf=brentq(lambda t:target(t)-.5*target(p1),-1,p3)
      freeze.append(dict(R0=R,normal=e['scaled'],target_peak_forced_to_reference=R*(h1-hf),target_peak_shift=p1-p3))
    (ROOT/'data/event_controls.json').write_text(json.dumps(freeze,indent=2))
if __name__=='__main__':main()
