from pathlib import Path
import numpy as np,json,time,datetime,gc,hashlib
from scipy.interpolate import CubicSpline
from refined_spectral_grid import RefinedSpectral
from observables import profiles,boundary_event
from unexpanded_rs import FullRS
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/model_comparison_refined';DEST.mkdir(exist_ok=True)
protocol={'type':'numerical refinement following observed aliasing and exact-logstep bug fix','grid':[[R,e] for R in [16,24,56,96,192] for e in [.2,.35,.7]],'extra_cases':[[256,.9],[256,1.]],'nlog_rule':'2097152 for R<192;4194304 for R>=192','du_rule':'.00125 for R<=24;.000625 for R<=96;.0003125 for R>=192','radial_grid':[401,16],'tau':[-3,5,.025],'model_names':['fresnel','exact_kz'],'reference':'unexpanded scalar RS at three shared planes and four radii; separate event grid convergence retained','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'memory_bounded_phase_columns':48,'code_hashes':{f:hashlib.sha256((OUT/'code'/f).read_bytes()).hexdigest() for f in ['run_refined_spectrum.py','refined_spectral_grid.py','spectral_solver.py','unexpanded_rs.py']}}
(OUT/'config/REFINED_SPECTRAL_RUN.json').write_text(json.dumps(protocol,indent=2));jobs=[]
for R,eta in protocol['grid']+protocol['extra_cases']:
 tag=f'R{R}_eta{eta:g}';fn=DEST/f'{tag}.json'
 if fn.exists():jobs.append(json.loads(fn.read_text()));continue
 start=time.perf_counter();print('START',tag,flush=True);nlog=2097152 if R<192 else 4194304;du=.00125 if R<=24 else (.000625 if R<=96 else .0003125);sol=RefinedSpectral(R,eta,nlog=nlog);ds=sol.propagate_grid(np.linspace(-3,5,321),du=du);rec={'tag':tag,'R0':R,'eta':eta,'nlog':nlog,'du':du,'job_status':'RUNNING','models':{}}
 for mod,d in ds.items():
  np.savez_compressed(DEST/f'{tag}_{mod}_fields.npz',**d,R0=R,eta=eta,kw=sol.kw,w_mm=sol.w);ev={};curves={'tau':d['tau']};roots={}
  for m in range(1,7):
   pp,rr=profiles(d,m);ev[str(m)]={key:boundary_event(d['tau'],v['Q'],rr if key=='actual_root' else None) for key,v in pp.items()};roots[str(m)]=rr
   for key,v in pp.items():
    for name,ar in v.items():curves[f'm{m}__{key}__{name}']=ar
  np.savez_compressed(DEST/f'{tag}_{mod}_curves.npz',**curves);rec['models'][mod]={'events':ev,'root_tracking':roots,'retained_power_relative_input':d['retained_channel_power_relative_input'].tolist()}
 # An independent unexpanded spatial integral at exactly matched coordinates.
 t=np.array([-.5,0.,1.]);x=np.array([0.,1.,3.,6.]);ref=FullRS(R,eta,ds=.0125,cutoff=np.ceil(20+eta*np.sqrt(R)/2)).propagate(t,x);d=ds['exact_kz'];cmp=[]
 for j,tt in enumerate(t):
  it=int(np.argmin(abs(d['tau']-tt)));cmp.append(np.array([CubicSpline(d['x'][it],d['U'][it,q])(x) for q in range(8)]))
 cmp=np.array(cmp);rec['unexpanded_RS_spotcheck']={'relative_complex_L2':float(np.linalg.norm(cmp-ref['U'])/np.linalg.norm(ref['U'])),'includes_radial_interpolation_error':True,'tau':t.tolist(),'x':x.tolist(),'source_ds':.0125};np.savez_compressed(DEST/f'{tag}_unexpanded_RS_spots.npz',**ref,spectral_interpolated=cmp);rec.update(job_status='RUN_COMPLETED',seconds=time.perf_counter()-start);fn.write_text(json.dumps(rec,indent=2));jobs.append(rec);(DEST/'RUN_STATUS.json').write_text(json.dumps({'completed_parameter_sets':len(jobs),'planned':17,'jobs':[{'tag':r['tag'],'status':r['job_status'],'seconds':r.get('seconds')} for r in jobs]},indent=2));print('END',tag,rec['seconds'],rec['unexpanded_RS_spotcheck'],flush=True);del sol,ds,d,ref,cmp,curves;gc.collect()
