"""Specified 90-beam independent model comparison, all fields retained."""
from pathlib import Path
import json,time,hashlib,traceback,gc,datetime
import numpy as np
from spectral_quadrature import HankelQuadrature
from observables import profiles,boundary_event
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/model_comparison';DEST.mkdir(exist_ok=True)
protocol={'R0':[16,24,56,96,192],'eta':[.2,.35,.7],'m':list(range(1,7)),'tau':[-3,5,.025],'ny':601,'xmax_at_zc':16,'forward_nlog':262144,'inverse_du':.0025,'spectral_umax_rule':'sqrt(24/alpha)','models':['fresnel','exact_kz'],'threads':1,'status':'requires_independent_grid_convergence'}
protocol['code_hashes']={f:hashlib.sha256((OUT/'code'/f).read_bytes()).hexdigest() for f in ['run_model_comparison.py','spectral_solver.py','spectral_quadrature.py','observables.py']}
(OUT/'config/MODEL_COMPARISON.json').write_text(json.dumps(protocol,indent=2));jobs=[]
for eta in protocol['eta']:
 for R in protocol['R0']:
  tag=f'R{R}_eta{eta:g}';fn=DEST/f'{tag}.json'
  if fn.exists():jobs.append(json.loads(fn.read_text()));continue
  t0=time.perf_counter();report={'case':tag,'R0':R,'eta':eta,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_status':'RUNNING'};print('START',tag,flush=True)
  try:
   sol=HankelQuadrature(R,eta,7,nlog=262144);tau=np.linspace(-3,5,321);both=sol.propagate_grid(tau,ny=601,xmax_at_zc=16,du=.0025,model=['fresnel','exact_kz']);report['spectrum_diagnostics']=sol.diagnostics;report['models']={}
   for model,d in both.items():
    if not np.isfinite(d['U']).all():raise FloatingPointError('nonfinite field')
    np.savez_compressed(DEST/f'{tag}_{model}_fields.npz',**d,R0=R,eta=eta,kw=sol.kw)
    records={};curves={'tau':tau};roots={}
    for m in range(1,7):
     pp,rr=profiles(d,m);records[str(m)]={label:boundary_event(tau,v['Q'],rr if label=='actual_root' else None) for label,v in pp.items()};roots[str(m)]=rr
     for label,v in pp.items():
      for key,arr in v.items():curves[f'm{m}__{label}__{key}']=arr
    np.savez_compressed(DEST/f'{tag}_{model}_curves.npz',**curves);report['models'][model]={'events':records,'root_tracking':roots,'spectral_nu':int(d['spectral_nu']),'spectral_du':float(d['spectral_du'])}
   report['job_status']='RUN_COMPLETED';del sol,both,curves,records,roots;gc.collect()
  except Exception as exc:report.update(job_status='FAILED',error=repr(exc),traceback=traceback.format_exc());print(report['traceback'],flush=True)
  report['seconds']=time.perf_counter()-t0;fn.write_text(json.dumps(report,indent=2));jobs.append(report)
  (DEST/'RUN_STATUS.json').write_text(json.dumps({'jobs':[{'case':j['case'],'status':j['job_status'],'seconds':j.get('seconds')} for j in jobs],'planned_beam_cases':90,'completed_beam_cases':sum(6 for j in jobs if j['job_status']=='RUN_COMPLETED')},indent=2));print('END',tag,report['job_status'],report['seconds'],flush=True)
