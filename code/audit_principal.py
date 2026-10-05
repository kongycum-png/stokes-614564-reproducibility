"""Audit every retained case; no case selection based on agreement."""
from pathlib import Path
import json,csv,collections
import numpy as np
from scipy.special import jnp_zeros
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/analysis';DEST.mkdir(exist_ok=True)
coeff=json.loads((OUT/'theory/formal_event_coefficients.json').read_text())['results']
pre=json.loads((OUT/'theory/analytic_preregistration.json').read_text())
D={m:m*m+.75-float(jnp_zeros(m,1)[0])**2 for m in range(1,7)}
rows=[];shapes=[];status=collections.Counter();exceptions=[]
for fn in sorted((OUT/'data/principal').glob('R*_eta*.json')):
 d=json.loads(fn.read_text());R=d['R0'];eta=d['eta']
 if d['job_status']!='RUN_COMPLETED':exceptions.append(d);continue
 cv=np.load(fn.with_name(fn.stem+'_curves.npz'));tau=cv['tau']
 for mstr,events in d['events'].items():
  m=int(mstr)
  for label,e in events.items():
   status[(label,e['status'])]+=1;ref=d['events']['1'][label]
   r={'R0':R,'eta':eta,'m':m,'boundary':label,'event_status':e['status'],'tau_peak':e['tau_peak'],'root_jumps':e.get('root_jumps',0),'missing_planes':e.get('missing_plane_count',0),'fwhm':e.get('fwhm'),'centroid':e.get('centroid'),'centroid_window':str(e.get('window'))}
   for g in np.arange(1,10)/10:
    k=f'{g:.1f}';h=e['thresholds'].get(k,{}).get('tau');h1=ref['thresholds'].get(k,{}).get('tau');r[f'tau_{k}']=h;r[f'advance_{k}']=h1-h if h is not None and h1 is not None else None
   r['peak_advance']=ref['tau_peak']-e['tau_peak'] if ref['tau_peak'] is not None and e['tau_peak'] is not None else None
   base=d['events'][mstr]['scaled_1']['thresholds'].get('0.5',{}).get('tau');base1=d['events']['1']['scaled_1']['thresholds'].get('0.5',{}).get('tau')
   delta0=base1-base if base1 is not None and base is not None else None
   r['boundary_change_absolute']=r['advance_0.5']-delta0 if r['advance_0.5'] is not None and delta0 is not None else None
   r['boundary_change_relative']=r['boundary_change_absolute']/delta0 if r['boundary_change_absolute'] is not None and abs(delta0)>1e-8 else None
   key=f'm{m}__{label}__';Q=cv[key+'Q'];T=cv[key+'T'];pb=cv[key+'pbar'];finite=np.isfinite(Q)&np.isfinite(T)&np.isfinite(pb)
   r['Q_equals_Tpbar_max_absolute']=float(np.max(abs(Q[finite]-T[finite]*pb[finite]))) if finite.any() else None
   if e['tau_peak'] is not None:
    ix=int(np.argmin(abs(tau-e['tau_peak'])));r['T_at_nearest_peak_plane']=float(T[ix]);r['pbar_at_nearest_peak_plane']=float(pb[ix]);r['negative_S3_at_nearest_peak_plane']=float(cv[key+'negative_S3_contribution'][ix])
   else:r.update(T_at_nearest_peak_plane=None,pbar_at_nearest_peak_plane=None,negative_S3_at_nearest_peak_plane=None)
   r.update(A=None,B=None,C=None,leading_prediction=None,residual_R1p5=None,residual_after_B_R2=None)
   if label=='scaled_1' and m>1 and f'{eta:g}' in coeff and r['advance_0.5'] is not None:
    A,B,C=coeff[f'{eta:g}'][mstr]['advance_coefficients'][2:5];delta=r['advance_0.5'];r.update(A=A,B=B,C=C,leading_prediction=A/R,residual_R1p5=R**1.5*(delta-A/R),residual_after_B_R2=R**2*(delta-A/R-B/R**1.5))
   rows.append(r)
  if m>1:
   y=cv[f'm{m}__scaled_1__Q'];y1=cv['m1__scaled_1__Q'];s=CubicSpline(tau,y);s1=CubicSpline(tau,y1);tt=np.linspace(-.5,.5,51);star=0.;floor=1e-8*max(y.max(),y1.max());valid=np.all(s(tt)>floor) and np.all(s1(tt)>floor) and s(star)>floor and s1(star)>floor
   shape={'R0':R,'eta':eta,'m':m,'double_ratio_status':'defined' if valid else 'below_positive_floor','double_ratio_predicted_slope':D[m]-D[1]}
   if valid:
    v=R*np.log(s(tt)*s1(star)/(s(star)*s1(tt)));slope=np.dot(tt,v)/np.dot(tt,tt);shape.update(double_ratio_measured_slope=float(slope),double_ratio_prediction_RMSE=float(np.sqrt(np.mean((v-(D[m]-D[1])*tt)**2))),double_ratio_affine_RMSE=float(np.sqrt(np.mean((v-slope*tt)**2))))
   tr=np.linspace(-1,2,151);ym=s(tr)
   def registration(delta,full=False):
    yr=s1(tr+delta);a=np.dot(ym,yr)/np.dot(yr,yr);err=np.linalg.norm(ym-a*yr)/np.linalg.norm(ym)
    return (a,err) if full else err
   opt=minimize_scalar(registration,bounds=(-1.9,1.9),method='bounded');a,err=registration(opt.x,True)
   shape.update(registration_shift=float(opt.x),registration_amplitude=float(a),registration_shape_residual=float(err),registration_window='[-1,2]',registration_search='[-1.9,1.9]')
   shapes.append(shape)
def writecsv(name,records):
 keys=list(dict.fromkeys(k for r in records for k in r))
 with (DEST/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
writecsv('principal_all_events.csv',rows);writecsv('principal_shape_tests.csv',shapes)
summary={'beam_cases':len(rows)//17,'boundary_records':len(rows),'run_failures':len(exceptions),'status_counts':[{'boundary':b,'status':s,'n':n} for (b,s),n in sorted(status.items())],'proxy_halfheight_undefined':[{'R0':r['R0'],'eta':r['eta'],'m':r['m'],'status':r['event_status']} for r in rows if r['boundary']=='scaled_1' and r['tau_0.5'] is None],'proxy_nonpositive_advances':[r for r in rows if r['boundary']=='scaled_1' and r['m']>1 and r['advance_0.5'] is not None and r['advance_0.5']<=0],'Q_equals_Tpbar_max_absolute':max(r['Q_equals_Tpbar_max_absolute'] or 0 for r in rows),'scientific_status':'PROPAGATION_AND_POSTPROCESSING_COMPLETE_VALIDATION_IN_PROGRESS'}
(DEST/'principal_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['proxy_nonpositive_advances','status_counts']},indent=2))
for r in rows:
 if r['R0']==24 and r['eta']==.7 and r['m'] in [2,3,4] and r['boundary'] in ['scaled_0.9','scaled_1','scaled_1.1','actual_root']:print({k:r[k] for k in ['m','boundary','advance_0.5','boundary_change_relative','event_status']})
