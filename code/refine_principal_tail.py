"""Convergence correction: add omitted source tail coherently to cached fields.
The original 600-case archive stays immutable. This is a quadrature refinement.
"""
from pathlib import Path
import json,time,datetime,hashlib,gc
import numpy as np
from scipy.integrate import cumulative_simpson
from revision_solver import Solver,Grid
from observables import profiles,boundary_event
OUT=Path(__file__).resolve().parents[1];SRC=OUT/'data/principal';DEST=OUT/'data/principal_refined';DEST.mkdir(exist_ok=True)
checks=json.loads((OUT/'baseline/tail_correction_check.json').read_text())
if len(checks)!=3 or max(abs(v) for r in checks for v in r['single_event_errors'])>1e-7:raise RuntimeError('Tail method has not passed the declared comparison')
protocol={'reason':'source-cutoff convergence failed for large R0 and strong eta in v1','change_type':'numerical_quadrature_refinement','original_archive':'data/principal','source_cutoff_rule':'ceil(16 + eta*sqrt(R0)/2)','source_ds':.025,'ny':601,'ymax':16,'phase_terms':4,'tau':[-3,5,.025],'tail_added_as_complex_amplitudes':True,'crosscheck':'baseline/tail_correction_check.json','scientific_scope_unchanged':True,'remaining_check':'two further source cutoffs at critical points'}
protocol['code_hashes']={f:hashlib.sha256((OUT/'code'/f).read_bytes()).hexdigest() for f in ['refine_principal_tail.py','revision_solver.py','observables.py']};(OUT/'config/PRINCIPAL_REFINEMENT.json').write_text(json.dumps(protocol,indent=2));jobs=[]
for fn in sorted(SRC.glob('R*_eta*.json')):
 old=json.loads(fn.read_text());R=old['R0'];eta=old['eta'];tag=old['case'];dest=DEST/f'{tag}.json'
 if dest.exists():jobs.append(json.loads(dest.read_text()));continue
 start=time.perf_counter();base=dict(np.load(SRC/f'{tag}_fields.npz'));cut=float(np.ceil(16+eta*np.sqrt(R)/2));sol=Solver(R,eta,7,Grid(ds=.025,cutoff=cut,ny=601,ymax=16,phase_terms=4,source_start=float(base['source_smax'])));tail=sol.propagate(base['tau']);d=dict(tail,U=base['U']+tail['U'],Pin=float(base['Pin'])+float(tail['Pin']),source_smin=0.,source_n=int(base['source_n'])+int(tail['source_n'])-1)
 np.savez_compressed(DEST/f'{tag}_fields.npz',**d,R0=R,eta=eta,alpha=sol.alpha,kw=sol.kw,w_mm=sol.w)
 events={};roots={};curves={'tau':d['tau']}
 for m in range(1,7):
  pp,rr=profiles(d,m);events[str(m)]={key:boundary_event(d['tau'],v['Q'],rr if key=='actual_root' else None) for key,v in pp.items()};roots[str(m)]=rr
  for key,v in pp.items():
   for name,arr in v.items():curves[f'm{m}__{key}__{name}']=arr
 np.savez_compressed(DEST/f'{tag}_curves.npz',**curves);cum=cumulative_simpson(np.pi*abs(d['U'])**2*d['rho_mm'][:,None,:],x=np.broadcast_to(d['rho_mm'][:,None,:],d['U'].shape),axis=2,initial=0)/d['Pin'];np.savez_compressed(DEST/f'{tag}_cumulative.npz',tau=d['tau'],rho_mm=d['rho_mm'],channel_power=cum)
 report={'case':tag,'R0':R,'eta':eta,'job_status':'RUN_COMPLETED','source_cutoff':cut,'source_start_tail':float(base['source_smax']),'source_smax':float(d['source_smax']),'events':events,'root_tracking':roots,'Pin':float(d['Pin']),'tail_relative_field_L2':float(np.linalg.norm(tail['U'])/np.linalg.norm(d['U'])),'seconds':time.perf_counter()-start,'code_hashes':protocol['code_hashes'],'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};dest.write_text(json.dumps(report,indent=2));jobs.append(report)
 (DEST/'RUN_STATUS.json').write_text(json.dumps({'jobs':[{'case':j['case'],'status':j['job_status'],'seconds':j.get('seconds')} for j in jobs],'beam_cases_completed':6*len(jobs),'planned_beam_cases':600},indent=2));print(tag,'cutoff',cut,'tail L2',report['tail_relative_field_L2'],'seconds',report['seconds'],flush=True)
 del sol,base,tail,d,curves,cum,events,roots;gc.collect()
