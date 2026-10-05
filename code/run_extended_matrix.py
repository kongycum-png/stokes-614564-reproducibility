"""Configured high-order/small-ring/P/fixed-alpha/kw controls, sequential."""
from pathlib import Path
import numpy as np,json,time,datetime,traceback,gc,hashlib
from sparse_channels import SparseSolver
from revision_solver import Grid
from observables import profiles,boundary_event
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/extended';DEST.mkdir(exist_ok=True)
Ptheory=json.loads((OUT/'theory/general_P_preregistration.json').read_text())
def savecase(sol,tau,tag,beams,kind,metadata):
 fn=DEST/f'{tag}.json'
 if fn.exists():return json.loads(fn.read_text())
 start=time.perf_counter();rec={'tag':tag,'kind':kind,'R0':sol.R0,'eta':sol.eta,'alpha':sol.alpha,'kw':sol.kw,'w_mm':sol.w,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metadata':metadata,'job_status':'RUNNING'};print('START',tag,flush=True)
 try:
  chunks=[sol.propagate(tau[j:j+64]) for j in range(0,len(tau),64)];d=dict(chunks[0]);d.update({key:np.concatenate([part[key] for part in chunks],axis=0) for key in ['tau','x','rho_mm','z_mm','U']});del chunks;np.savez_compressed(DEST/f'{tag}_fields.npz',**d,R0=sol.R0,eta=sol.eta,kw=sol.kw,w_mm=sol.w);records={};curves={'tau':tau}
  for m,P in beams:
   root=None if P==1 else Ptheory['domains'][str(P)][str(m)]['root'];common=None if P==1 else Ptheory['domains'][str(P)][str(P)]['root'];pp,rr=profiles(d,m,P,root,common);bid=f'P{P}_m{m}';records[bid]={'m':m,'P':P,'events':{key:boundary_event(tau,v['Q'],rr if key=='actual_root' else None) for key,v in pp.items()},'root_tracking':rr}
   for key,v in pp.items():
    for name,arr in v.items():curves[f'{bid}__{key}__{name}']=arr
  np.savez_compressed(DEST/f'{tag}_curves.npz',**curves);rec.update(job_status='RUN_COMPLETED',beams=records,source_n=int(d['source_n']),source_smax=float(d['source_smax']),phase_bound=float(d['phase_remainder_absolute_bound']));del d,curves;gc.collect()
 except Exception as exc:rec.update(job_status='FAILED',error=repr(exc),traceback=traceback.format_exc());print(rec['traceback'],flush=True)
 rec['seconds']=time.perf_counter()-start;fn.write_text(json.dumps(rec,indent=2));print('END',tag,rec['job_status'],rec['seconds'],flush=True);return rec
if __name__=='__main__':
 jobs=[];protocol={'high_order_m':[1,8,12,16],'high_order_R':[24,56,96,192,256],'eta':[.35,.7],'small_R':[8,12],'P_generalization':'all preregistered P=2,3 domains at R24,56,96,192','fixed_alpha':[.02,.04,.07],'kw':[200,400,800,1600],'source_ds':.025,'source_cutoff':'ceil(16+eta*sqrt(R0)/2)','threads':1,'max_kernel_GiB':5,'high_order_tau':[-6,8,.025],'ordinary_tau':[-3,5,.025],'note':'test levels, not certified validity or instrument tolerances'}
 protocol['code_hashes']={f:hashlib.sha256((OUT/'code'/f).read_bytes()).hexdigest() for f in ['run_extended_matrix.py','sparse_channels.py','revision_solver.py','observables.py']};(OUT/'config/EXTENDED_RUN.json').write_text(json.dumps(protocol,indent=2))
 specs=[]
 for eta in [.35,.7]:
  for R in [24,56,96,192,256]:specs.append(('high_order',R,eta,[(m,1) for m in [1,8,12,16]],32,801,-6,8))
  for R in [8,12]:specs.append(('small_ring',R,eta,[(m,1) for m in range(1,5)],16,601,-3,5))
  for R in [24,56,96,192]:specs.append(('P_generalization',R,eta,[(int(m),int(P)) for P,rows in Ptheory['domains'].items() for m in rows],18,601,-3,5))
 for alpha in [.02,.04,.07]:
  for R in [24,56,96,192,256]:specs.append(('fixed_alpha',R,2*alpha*np.sqrt(R),[(m,1) for m in [1,2,4,6]],16,601,-3,5))
 for kind,R,eta,beams,ymax,ny,tmin,tmax in specs:
  tag=f'{kind}_R{R}_eta{eta:.12g}'
  if (DEST/f'{tag}.json').exists():jobs.append(json.loads((DEST/f'{tag}.json').read_text()));continue
  orders=sorted(set(abs(m+s*P) for m,P in beams for s in [-1,1]));cut=float(np.ceil(16+eta*np.sqrt(R)/2));ns=np.ceil((R+max(45,cut/(eta/(2*np.sqrt(R)))))/.025)+3
  if len(orders)*ny*ns*8>5*1024**3:raise MemoryError(f'planned kernel exceeds resource bound: {tag}')
  sol=SparseSolver(R,eta,orders,Grid(ds=.025,cutoff=cut,ny=ny,ymax=ymax,phase_terms=4));jobs.append(savecase(sol,np.linspace(tmin,tmax,round((tmax-tmin)/.025)+1),tag,beams,kind,{'physical_path':kind}));del sol;gc.collect()
  (DEST/'RUN_STATUS.json').write_text(json.dumps({'jobs':[{'tag':j['tag'],'status':j['job_status']} for j in jobs],'planned_parameter_jobs':len(specs)+24},indent=2))
 for eta in [.35,.7]:
  for R in [24,96,256]:
   sol=SparseSolver(R,eta,[0,1,2,3,5],Grid(ds=.025,cutoff=float(np.ceil(16+eta*np.sqrt(R)/2)),ny=601,ymax=16,phase_terms=4));input_integral=sol.Pin/sol.w**2
   for kw in [200,400,800,1600]:
    sol.kw=kw;sol.w=kw/sol.k;sol.c=np.sqrt(1-(sol.a/kw)**2);sol.dc=-(sol.a/kw)**2/(1+sol.c);sol.Pin=input_integral*sol.w**2;tag=f'kw{kw}_R{R}_eta{eta:g}';jobs.append(savecase(sol,np.linspace(-3,5,321),tag,[(m,1) for m in [1,2,4]],'kw_control',{'w_derived_from_kw':True,'input_kernel_reused':True}))
   del sol;gc.collect();(DEST/'RUN_STATUS.json').write_text(json.dumps({'jobs':[{'tag':j['tag'],'status':j['job_status']} for j in jobs],'planned_parameter_jobs':len(specs)+24},indent=2))
