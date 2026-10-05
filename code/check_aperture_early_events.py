"""Targeted source/axial refinement of tiny early finite-aperture peaks."""
from pathlib import Path
import numpy as np,json,time,hashlib
from scipy.interpolate import CubicSpline
from scipy.special import jnp_zeros
from sparse_channels import SparseSolver
from revision_solver import Grid
from observables import event
O=Path(__file__).resolve().parents[1];D=O/'data/aperture_refinement';D.mkdir(exist_ok=True)
protocol={'R0':24,'eta':.7,'source_aperture_r0':3,'levels':[[.025,601,.025],[.0125,601,.0125],[.0125,1201,.00625]],'tau_window':[-8,-4],'purpose':'check tiny early stationary maxima, without discarding or changing the event definition','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(O/'config/APERTURE_EARLY_EVENTS.json').write_text(json.dumps(protocol,indent=2));records=[]
for ds,ny,dt in protocol['levels']:
 tag=f'ds{ds:g}_ny{ny}_dt{dt:g}';print('START',tag,flush=True);sol=SparseSolver(24,.7,[0,1,2,3,5],Grid(ds=ds,cutoff=22,ny=ny,ymax=16,phase_terms=4));tau=np.linspace(-8,-4,round(4/dt)+1);edge=72;width=.5;window=np.where(sol.s<edge-width,1.,np.where(sol.s>=edge,0.,.5*(1+np.cos(np.pi*(sol.s-edge+width)/width))));pin=2*np.pi*sol.w**2*np.dot(sol.weights,abs(sol.f*window)**2*sol.s);chunks=[sol.propagate(tau[j:j+64],source_multiplier=window) for j in range(0,len(tau),64)];d=dict(chunks[0]);d.update({k:np.concatenate([v[k] for v in chunks]) for k in ['tau','x','rho_mm','z_mm','U']});np.savez_compressed(D/f'{tag}_fields.npz',**d);save={'tau':tau};ev={}
 for m in [1,2,4]:
  plus=d['U'][:,[0,1,2,3,5].index(m-1)];minus=d['U'][:,[0,1,2,3,5].index(m+1)];q=[]
  for i in range(len(tau)):
   density=np.pi*(abs(plus[i])**2-abs(minus[i])**2)*d['rho_mm'][i]*np.gradient(d['rho_mm'][i],d['x'][i])/pin;cs=CubicSpline(d['x'][i],density).antiderivative();q.append(float(cs(jnp_zeros(m,1)[0])-cs(0)))
  save[f'm{m}_Q']=np.array(q);ev[str(m)]=event(tau,q)
 np.savez_compressed(D/f'{tag}_curves.npz',**save);records.append({'tag':tag,'events':ev,'Pin':pin});(D/'results.json').write_text(json.dumps(records,indent=2));print('END',tag,[(m,e['tau_peak'],e.get('thresholds',{}).get('0.5',{}).get('tau')) for m,e in ev.items()],flush=True)
