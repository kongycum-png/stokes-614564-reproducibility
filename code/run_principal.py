"""600 beam cases. Unique q channels reused across boundary/event families."""
from pathlib import Path
import json,time,hashlib,datetime,traceback,gc,resource
import numpy as np
from revision_solver import Solver,Grid
from observables import profiles,boundary_event
OUT=Path(__file__).resolve().parents[1];DATA=OUT/'data/principal';DATA.mkdir(parents=True,exist_ok=True)
ETA=[.2,.3,.35,.4,.5,.6,.7,.8,.9,1.];R0=[16,24,32,40,56,72,96,128,192,256]
# This numerical grid is frozen before principal production. Event absence is scoped to its window.
protocol={'tau_min':-3.,'tau_max':5.,'tau_step':.025,'source_cutoff_exponents':13.,'source_ds':.025,'ny':601,'ymax':16.,'phase_terms':4,'source_cutoff_test_required':[11,13,15],'max_processes':1,'thread_limit':1,'memory_target_GiB':8,'purpose':'main model; independent exact-kz and convergence coverage separate','source_truncation_is_not_certified_by_phase_bound':True}
code_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUT/'code/revision_solver.py',OUT/'code/observables.py']};protocol['code_hashes']=code_hash
cfg_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest();(OUT/'config/PRINCIPAL_RUN.json').write_text(json.dumps(protocol,indent=2))
status_file=DATA/'RUN_STATUS.json';jobs=[]
for eta in ETA:
 for R in R0:
  tag=f'R{R}_eta{eta:g}';done=DATA/f'{tag}.json'
  if done.exists():
   old=json.loads(done.read_text())
   if old.get('config_hash')!=cfg_hash:raise RuntimeError(f'Existing incompatible result {tag}')
   jobs.append({'case':tag,'status':old['job_status']});continue
  t0=time.perf_counter();print('START',tag,flush=True);report={'case':tag,'R0':R,'eta':eta,'config_hash':cfg_hash,'code_hashes':code_hash,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_status':'RUNNING'}
  try:
   sol=Solver(R,eta,7,Grid(ds=.025,cutoff=13.,ny=601,ymax=16.,phase_terms=4));tau=np.linspace(-3,5,321);d=sol.propagate(tau)
   if not np.isfinite(d['U']).all():raise FloatingPointError('non-finite complex field')
   field=DATA/f'{tag}_fields.npz';np.savez_compressed(field,**d,R0=R,eta=eta,alpha=sol.alpha,kw=sol.kw,w_mm=sol.w,config_hash=cfg_hash)
   records={};curves={'tau':tau};roots={}
   for m in range(1,7):
    pp,rr=profiles(d,m);records[str(m)]={label:boundary_event(tau,v['Q'],rr if label=='actual_root' else None) for label,v in pp.items()};roots[str(m)]=rr
    for label,v in pp.items():
     for key,arr in v.items():curves[f'm{m}__{label}__{key}']=arr
   np.savez_compressed(DATA/f'{tag}_curves.npz',**curves)
   # Save channel radial cumulative powers so alternative boundaries never require repropagation.
   from scipy.integrate import cumulative_simpson
   cum=cumulative_simpson(np.pi*abs(d['U'])**2*d['rho_mm'][:,None,:],x=np.broadcast_to(d['rho_mm'][:,None,:],d['U'].shape),axis=2,initial=0)/sol.Pin
   np.savez_compressed(DATA/f'{tag}_cumulative.npz',tau=tau,rho_mm=d['rho_mm'],channel_power=cum)
   report.update(job_status='RUN_COMPLETED',fields=str(field.relative_to(OUT)),events=records,root_tracking=roots,source_n=d['source_n'],source_smax=d['source_smax'],phase_remainder_absolute_bound=d['phase_remainder_absolute_bound'],Pin=sol.Pin)
   del sol,d,cum,curves,records,roots;gc.collect()
  except Exception as exc:
   report.update(job_status='FAILED',exception=repr(exc),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
  report['seconds']=time.perf_counter()-t0;report['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;done.write_text(json.dumps(report,indent=2,allow_nan=True));jobs.append({'case':tag,'status':report['job_status'],'seconds':report['seconds']})
  status_file.write_text(json.dumps({'jobs':jobs,'beam_cases_completed':6*sum(j['status']=='RUN_COMPLETED' for j in jobs),'planned_beam_cases':600,'configuration_hash':cfg_hash},indent=2));print('END',tag,report['job_status'],round(report['seconds'],2),flush=True)
