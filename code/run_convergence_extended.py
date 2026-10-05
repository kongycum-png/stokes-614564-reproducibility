"""Separate source, radial, axial and spectral convergence axes."""
from pathlib import Path
import json,time,gc,traceback,hashlib
import numpy as np
from scipy.special import jnp_zeros
from scipy.interpolate import CubicSpline
from sparse_channels import SparseSolver
from revision_solver import Grid
from spectral_quadrature import HankelQuadrature
from observables import event
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/convergence';DEST.mkdir(exist_ok=True)
def extract(d,ms=[1,4,6]):
 out={};orders=list(d['q_orders']);rho=d['rho_mm'];tau=d['tau'];x=d['x']
 for m in ms:
  diff=abs(d['U'][:,orders.index(m-1)])**2-abs(d['U'][:,orders.index(m+1)])**2;Q=[]
  for i in range(len(tau)):
   sp=CubicSpline(rho[i],np.pi*diff[i]*rho[i]/float(d['Pin'])).antiderivative();r=float(jnp_zeros(m,1)[0])*2*(rho[i,-1]/x[i,-1])/2;Q.append(float(sp(r)-sp(0)))
  out[str(m)]=event(tau,Q)
 return out
if __name__=='__main__':
 axes=[('cutoff11',.025,11,601,.025),('cutoff15',.025,15,601,.025),('ds004',.04,13,601,.025),('ds00125',.0125,13,601,.025),('ny301',.025,13,301,.025),('ny1201',.025,13,1201,.025),('dt00125',.025,13,601,.0125)]
 protocol={'source_cases':[[24,.7],[192,.2],[256,.9]],'source_axes':axes,'spectral_cases':[[24,.7],[192,.2]],'spectral_axes':[('base',262144,.0025,24),('nlog131072',131072,.0025,24),('nlog524288',524288,.0025,24),('du005',262144,.005,24),('du00125',262144,.00125,24),('umax20',262144,.0025,20),('umax28',262144,.0025,28)],'m':[1,4,6],'source_orders':[0,2,3,5,7],'reference_source':'principal grid ds=.025, cutoff=13, ny=601','thread_limit':1,'error_target_tau':1e-6,'no_claim_of_rigorous_error_bound':True}
 protocol['code_hashes']={name:hashlib.sha256((OUT/'code'/name).read_bytes()).hexdigest() for name in ['run_convergence_extended.py','sparse_channels.py','revision_solver.py','spectral_solver.py','spectral_quadrature.py']};(OUT/'config/CONVERGENCE_RUN.json').write_text(json.dumps(protocol,indent=2));jobs=[]
 tasks=[('source',R,eta,name,ds,cut,ny,dt) for R,eta in protocol['source_cases'] for name,ds,cut,ny,dt in axes]
 tasks += [('spectrum',R,eta,name,nlog,du,cut) for R,eta in protocol['spectral_cases'] for name,nlog,du,cut in protocol['spectral_axes']]
 for task in tasks:
  kind,R,eta,name=task[:4];tag=f'{kind}_R{R}_eta{eta:g}_{name}';fn=DEST/f'{tag}.json'
  if fn.exists():jobs.append(json.loads(fn.read_text()));continue
  start=time.perf_counter();rec={'tag':tag,'kind':kind,'R0':R,'eta':eta,'axis':name,'job_status':'RUNNING','task':task};print('START',tag,flush=True)
  try:
   if kind=='source':
    _,_,_,_,ds,cut,ny,dt=task;sol=SparseSolver(R,eta,[0,2,3,5,7],Grid(ds=ds,cutoff=cut,ny=ny,ymax=16,phase_terms=4));d=sol.propagate(np.linspace(-3,5,round(8/dt)+1));rec.update(source_n=int(d['source_n']),source_smax=float(d['source_smax']),phase_bound=float(d['phase_remainder_absolute_bound']))
   else:
    _,_,_,_,nlog,du,cut=task;sol=HankelQuadrature(R,eta,7,nlog=nlog,umax=np.sqrt(cut/(eta/(2*np.sqrt(R)))));d=sol.propagate_grid(np.linspace(-3,5,321),ny=601,xmax_at_zc=16,du=du,model='exact_kz');rec['spectra']=sol.diagnostics
   rec['events']=extract(d);np.savez_compressed(DEST/f'{tag}_fields.npz',**d);rec['job_status']='RUN_COMPLETED';del d,sol;gc.collect()
  except Exception as exc:rec.update(job_status='FAILED',error=repr(exc),traceback=traceback.format_exc());print(rec['traceback'],flush=True)
  rec['seconds']=time.perf_counter()-start;fn.write_text(json.dumps(rec,indent=2));jobs.append(rec);(DEST/'RUN_STATUS.json').write_text(json.dumps({'jobs':[{'tag':r['tag'],'status':r['job_status'],'seconds':r.get('seconds')} for r in jobs],'planned_jobs':len(tasks)},indent=2));print('END',tag,rec['job_status'],rec['seconds'],flush=True)
