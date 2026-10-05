from pathlib import Path
import numpy as np,json,time,gc,datetime,hashlib
from sparse_channels import SparseSolver
from revision_solver import Grid
from refined_spectral_grid import RefinedSpectral
from run_convergence_extended import extract
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/convergence_final';DEST.mkdir(exist_ok=True)
protocol={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_tail_cases':[[24,.7],[192,.2],[256,.9],[256,1.]],'extra_cutoffs':[2,4],'tail_ds':.0125,'spectral_tasks':[[192,.2,'nlog2M',2097152,.0003125,401,.025],[192,.2,'nlog8M',8388608,.0003125,401,.025],[192,.2,'du000625',4194304,.000625,401,.025],[192,.2,'du00015625',4194304,.00015625,401,.025],[192,.2,'ny201',4194304,.0003125,201,.025],[192,.2,'ny801',4194304,.0003125,801,.025],[24,.7,'ny201',2097152,.00125,201,.025],[24,.7,'ny801',2097152,.00125,801,.025],[256,.9,'nlog8M',8388608,.0003125,401,.025],[256,.9,'dt00125',4194304,.0003125,401,.0125]],'comparison':'corresponding production refined spectral grid; all changes isolated','event_absolute_target':1e-6,'small_difference_target':'report actual differences against the order effect and higher-order residual; no fixed uniform bound claimed'}
protocol['code_hashes']={f:hashlib.sha256((OUT/'code'/f).read_bytes()).hexdigest() for f in ['final_convergence.py','refined_spectral_grid.py','spectral_solver.py','revision_solver.py']};(OUT/'config/FINAL_CONVERGENCE.json').write_text(json.dumps(protocol,indent=2));jobs=[]
for R,eta in protocol['source_tail_cases']:
 old=dict(np.load(OUT/'data/principal_refined'/f'R{R}_eta{eta:g}_fields.npz'));cut=json.loads((OUT/'data/principal_refined'/f'R{R}_eta{eta:g}.json').read_text())['source_cutoff'];orders=[0,2,3,5,7]
 for extra in protocol['extra_cutoffs']:
  tag=f'source_R{R}_eta{eta:g}_cutplus{extra}';fn=DEST/f'{tag}.json'
  if fn.exists():jobs.append(json.loads(fn.read_text()));continue
  start=time.perf_counter();print('START',tag,flush=True);sol=SparseSolver(R,eta,orders,Grid(ds=.0125,cutoff=cut+extra,ny=601,ymax=16,phase_terms=4,source_start=float(old['source_smax'])));tail=sol.propagate(old['tau']);d=dict(tail,U=old['U'][:,orders]+tail['U'],Pin=float(old['Pin'])+float(tail['Pin']));rec={'tag':tag,'R0':R,'eta':eta,'cutoff':cut+extra,'events':extract(d),'seconds':time.perf_counter()-start,'job_status':'RUN_COMPLETED'};fn.write_text(json.dumps(rec,indent=2));np.savez_compressed(DEST/f'{tag}_tail_fields.npz',**tail);jobs.append(rec);print('END',tag,rec['seconds'],flush=True);del sol,d,tail;gc.collect()
 del old
for R,eta,axis,nlog,du,ny,dt in protocol['spectral_tasks']:
 tag=f'spectral_R{R}_eta{eta:g}_{axis}';fn=DEST/f'{tag}.json'
 if fn.exists():jobs.append(json.loads(fn.read_text()));continue
 start=time.perf_counter();print('START',tag,flush=True);sol=RefinedSpectral(R,eta,nlog=nlog);d=sol.propagate_grid(np.linspace(-3,5,round(8/dt)+1),ny=ny,du=du,models=('exact_kz',))['exact_kz'];rec={'tag':tag,'R0':R,'eta':eta,'axis':axis,'nlog':nlog,'du':du,'ny':ny,'dt':dt,'events':extract(d),'seconds':time.perf_counter()-start,'job_status':'RUN_COMPLETED'};fn.write_text(json.dumps(rec,indent=2));np.savez_compressed(DEST/f'{tag}_fields.npz',**d);jobs.append(rec);(DEST/'RUN_STATUS.json').write_text(json.dumps({'completed':len(jobs),'planned':18,'jobs':[{'tag':r['tag'],'status':r['job_status']} for r in jobs]},indent=2));print('END',tag,rec['seconds'],flush=True);del sol,d;gc.collect()
