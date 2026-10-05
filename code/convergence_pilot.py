from pathlib import Path
import sys,time,json,warnings,resource
import numpy as np
from scipy.special import jnp_zeros
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar,brentq
from revision_solver import Solver,Grid
OUT=Path(__file__).resolve().parents[1]
levels=[('base',.04,11.,401),('source_extent_9',.04,9.,401),('source_extent_13',.04,13.,401),('source_ds_055',.055,11.,401),('source_ds_025',.025,11.,401),('radial_201',.04,11.,201),('radial_801',.04,11.,801)]
rows=[]
for label,ds,cutoff,ny in levels:
 t0=time.perf_counter();s=Solver(24,.7,4,Grid(ds=ds,cutoff=cutoff,ny=ny,ymax=8.5,phase_terms=4));t=np.linspace(-.5,1.5,101);d=s.propagate(t);events={}
 for m in [1,2,3]:
  q=[];b=jnp_zeros(m,1)[0]
  for i in range(len(t)):
   rho=d['rho_mm'][i];den=np.pi*(abs(d['U'][i,m-1])**2-abs(d['U'][i,m+1])**2)*rho/s.Pin
   integ=CubicSpline(rho,den).antiderivative();r=b*2*s.k*s.w**3/d['z_mm'][i];q.append(float(integ(r)-integ(0)))
  cs=CubicSpline(t,q);r=cs.derivative().roots(extrapolate=False);p=next(float(x) for x in r if cs(x,2)<0)
  # direct event evaluation, cubic spline only brackets, not final precision claim
  fun=lambda tau:float(s.q(tau,m)[0]);peak=minimize_scalar(lambda x:-fun(x),bounds=(p-.06,p+.06),method='bounded',options={'xatol':1e-9});p=float(peak.x);h=brentq(lambda x:fun(x)-.5*fun(p),-.5,p,xtol=2e-10)
  events[str(m)]={'tau50':h,'peak':p,'sampled_cubic_tau50':brentq(lambda x:cs(x)-.5*cs(p),-.5,p)}
 rows.append({'label':label,'ds':ds,'cutoff':cutoff,'ny':ny,'events':events,'Delta2':events['1']['tau50']-events['2']['tau50'],'Delta3':events['1']['tau50']-events['3']['tau50'],'seconds':time.perf_counter()-t0,'phase_bound':d['phase_remainder_absolute_bound']})
 print(json.dumps(rows[-1]),flush=True)
 (OUT/'baseline/convergence_pilot.json').write_text(json.dumps(rows,indent=2))
