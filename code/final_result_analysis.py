"""Evidence tables, with prediction, fitting, model and numerical errors separated."""
from pathlib import Path
import json,csv,collections,numpy as np
from scipy.interpolate import CubicSpline,make_interp_spline
from scipy.optimize import brentq
from scipy.integrate import simpson
from scipy.special import jnp_zeros
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/final_analysis';DEST.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def write(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (DEST/name).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def half(e):return e.get('thresholds',{}).get('0.5',{}).get('tau')
def delta(events,m):
 a=half(events['1']);b=half(events[str(m)]);return None if a is None or b is None else a-b
def diff(a,b):return None if a is None or b is None else a-b
def loadcsv(p):
 rr=list(csv.DictReader(p.open()))
 for r in rr:
  for k,v in r.items():
   try:r[k]=float(v) if v else None
   except ValueError:pass
 return rr
pre=read(OUT/'theory/analytic_preregistration.json');coeff=read(OUT/'theory/formal_event_coefficients.json')['results'];D={m:m*m+.75-float(jnp_zeros(m,1)[0])**2 for m in range(1,17)}
principal=loadcsv(OUT/'data/analysis_refined/principal_all_events.csv');pred=[]
for r in principal:
 if r['boundary'] not in ['scaled_1','actual_root'] or r['m']==1:continue
 R=r['R0'];eta=r['eta'];m=int(r['m']);ae=pre['airy_events'].get(f'{eta:g}',{});kv=ae.get('thresholds',{}).get('0.5',{}).get('K_proxy' if r['boundary']=='scaled_1' else 'K_actual_root_candidate');A=None if kv is None else kv*(D[1]-D[m]);dv=r['advance_0.5'];B=C=None
 if r['boundary']=='scaled_1' and f'{eta:g}' in coeff:B,C=coeff[f'{eta:g}'][str(m)]['advance_coefficients'][3:5]
 pr={'R0':R,'eta':eta,'m':m,'boundary':r['boundary'],'event_status':r['event_status'],'advance':dv,'A':A,'B':B,'C':C,'analytic_scope':'simple first Airy peak/crossing, fixed finite m; actual root also requires simple continuous root','holdout':R in [128,192,256]}
 for n in [1,2,3]:
  v=None if A is None else A/R
  if n>=2:v=None if v is None or B is None else v+B/R**1.5
  if n>=3:v=None if v is None or C is None else v+C/R**2
  pr[f'prediction_{n}']=v;pr[f'error_{n}']=diff(v,dv)
  pr[f'relative_error_{n}']=None if v is None or dv is None or abs(dv)<1e-8 else (v-dv)/dv
 pred.append(pr)
write('analytic_predictions.csv',pred)
fits=[]
for eta in [.2,.35,.7,.9]:
 for m in range(2,7):
  rr=sorted([r for r in pred if r['eta']==eta and r['m']==m and r['boundary']=='scaled_1'],key=lambda r:r['R0']);train=[r for r in rr if r['R0']<=96];R=np.array([r['R0'] for r in train]);Y=np.array([r['advance']-r['A']/r['R0'] for r in train]);X=np.column_stack([R**-1.5,R**-2]);bc=np.linalg.lstsq(X,Y,rcond=None)[0]
  for r in rr:
   v=r['A']/r['R0']+bc[0]/r['R0']**1.5+bc[1]/r['R0']**2;fits.append({'eta':eta,'m':m,'R0':r['R0'],'role':'holdout' if r['holdout'] else 'training','A_fixed_analytic':r['A'],'B_fitted':bc[0],'C_fitted':bc[1],'B_analytic':r['B'],'C_analytic':r['C'],'fit_prediction':v,'fit_error':v-r['advance'],'analytic_third_error':r['error_3'],'training_range':'16<=R0<=96; no holdout data fit'})
write('higher_order_fit_holdout.csv',fits)
# Common-window moments are computed independently of peak existence. The old
# full-window FWHM is retained, but cannot cross a later local minimum/peak.
moments=[];locator=[]
for fn in sorted((OUT/'data/principal_refined').glob('R[0-9]*.json')):
 d=read(fn);cv=np.load(fn.with_name(fn.stem+'_curves.npz'));tau=cv['tau'];local=[]
 for ms,evs in d['events'].items():
  for boundary,e in evs.items():
   y=cv[f'm{ms}__{boundary}__Q'];use=(tau>=-1)&(tau<=2);finite=np.isfinite(y[use]).all();centroid=None;positive=None;branchsafe=True
   if boundary=='actual_root':branchsafe=not any(-1<tau[j+1]<2 for j in d['root_tracking'][ms]['jump_indices'])
   if finite and branchsafe:
    cs=CubicSpline(tau[use],y[use]);den=cs.integrate(-1,2);nodes=np.linspace(-1,2,1201);centroid=float(simpson(nodes*cs(nodes),x=nodes)/den) if abs(den)>1e-12 else None;positive=bool(np.min(y[use])>=0)
   p=e.get('tau_peak');h=half(e);fall=e.get('thresholds',{}).get('0.5',{}).get('falling_tau');same=None;reason='no_pair_of_half_crossings';nextmin=None
   if p is not None and h is not None and fall is not None:
    use2=(tau>=e['window'][0])&(tau<=e['window'][1]);ss=CubicSpline(tau[use2],y[use2]);mins=[z for z in ss.derivative().roots(extrapolate=False) if z>p and ss(z,2)>0];nextmin=float(min(mins)) if mins else e['window'][1]
    if fall<=nextmin:same=fall-h;reason='defined'
    else:reason='half_crossing_beyond_next_local_minimum'
   peakc=e.get('peak_curvature');slope=e.get('thresholds',{}).get('0.5',{}).get('slope');height=e.get('q_peak');r={'R0':d['R0'],'eta':d['eta'],'m':int(ms),'boundary':boundary,'common_window':'[-1,2]','common_centroid':centroid,'centroid_positive_weight':positive,'common_window_branch_continuous':branchsafe,'centroid_status':'defined' if centroid is not None else 'missing_or_branch_change_or_zero_integral','fwhm_old_search':e.get('fwhm'),'fwhm_same_peak':same,'fwhm_status':reason,'next_local_minimum':nextmin,'peak_condition':height/abs(peakc) if height is not None and peakc else None,'half_condition':height/abs(slope) if height is not None and slope else None};local.append(r)
   if boundary=='scaled_1' and int(ms) in [1,4,6] and d['R0'] in [24,192,256] and d['eta'] in [.2,.7,.9]:
    sp=make_interp_spline(tau,y,k=5);cp=sp.derivative();crit=[]
    for a,b in zip(tau[:-1],tau[1:]):
     if cp(a)>0 and cp(b)<0:crit.append(brentq(cp,a,b))
    pp=next((v for v in crit if tau[1]<v<tau[-2] and sp(v)>1e-12),None);hh=None
    if pp is not None:
     yy=sp(tau)-.5*sp(pp);roots=[brentq(lambda t:sp(t)-.5*sp(pp),a,b) for a,b,ya,yb in zip(tau[:-1],tau[1:],yy[:-1],yy[1:]) if ya<0<yb and b<pp];hh=roots[-1] if roots else None
    locator.append({'R0':d['R0'],'eta':d['eta'],'m':int(ms),'quintic_half':hh,'cubic_half':h,'quintic_minus_cubic_half':diff(hh,h),'quintic_minus_cubic_peak':diff(pp,p)})
 for r in local:
  ref=next(v for v in local if v['m']==1 and v['boundary']==r['boundary']);r['common_centroid_advance']=diff(ref['common_centroid'],r['common_centroid'])
 moments+=local
write('common_centroids_conditioning_and_widths.csv',moments);write('event_locator_comparison.csv',locator)
conv=[]
for fn in (OUT/'data/convergence_final').glob('*_eta*.json'):
 d=read(fn);R=d['R0'];eta=d['eta'];source=fn.stem.startswith('source');base=read(OUT/('data/principal_refined' if source else 'data/model_comparison_refined')/f'R{R}_eta{eta:g}.json');old={m:ev['scaled_1'] for m,ev in (base['events'] if source else base['models']['exact_kz']['events']).items()}
 for m,e in d['events'].items():
  v=delta(d['events'],m);bv=delta(old,m);conv.append({'tag':d['tag'],'R0':R,'eta':eta,'m':int(m),'axis':'source_tail' if source else d['axis'],'reference_half':half(old[m]),'refined_half':half(e),'single_half_difference':diff(half(e),half(old[m])),'advance_difference':diff(v,bv),'peak_difference':diff(e.get('tau_peak'),old[m].get('tau_peak'))})
write('final_convergence.csv',conv)
inputs=[]
for fn in (OUT/'data/input_mismatch').glob('R[0-9]*.json'):
 d=read(fn)
 if 'events' not in d:continue
 meta=d['metadata'];R=meta['R0'];eta=meta['eta'];base=read(OUT/'data/input_mismatch'/f'R{R}_eta{eta:g}_balanced.json')
 for m,e in d['events'].items():
  dv=delta(d['events'],m);basev=delta(base['events'],m);inputs.append(dict(meta,tag=d['tag'],m=int(m),status=e['status'],peak_status=e['status'],half_status=e.get('thresholds',{}).get('0.5',{}).get('status','no_selected_peak'),paired_advance_status='defined' if dv is not None else 'missing_reference_half' if half(e) is not None else 'missing_order_half',half=half(e),peak=e.get('tau_peak'),advance=dv,balanced_advance=basev,advance_change=diff(dv,basev),relative_advance_change=(dv-basev)/basev if dv is not None and basev is not None and abs(basev)>1e-8 else None,all_maxima=str(e.get('all_stationary_maxima')),Pin=d['Pin']))
write('input_tolerances.csv',inputs)
two=[]
for fn in (OUT/'data/input_2d').glob('R[0-9]*.json'):
 d=read(fn)
 for m,e in d['events'].items():two.append(dict(tag=fn.stem,m=int(m),status=e['status'],half=half(e),advance=delta(d['events'],m)))
write('translation_events.csv',two)
ell=[]
for fn in (OUT/'data/input_2d/ellipticity').glob('ellipticity*.json'):
 d=read(fn)
 for key,evs in d['events'].items():
  for m,e in evs.items():ell.append({'ellipticity':d['elliptic_log_axis_ratio'],'key':key,'m':int(m),'status':e['status'],'half':half(e),'advance':delta(evs,m)})
write('ellipticity_events.csv',ell)
general=[];pcoeff=read(OUT/'theory/general_P_preregistration.json')['predictions']
for r in loadcsv(OUT/'data/analysis_refined/extended_events.csv'):
 if r['kind']!='P_generalization' or r['boundary']!='scaled_1':continue
 a=next(v['A'] for v in pcoeff if v['P']==r['P'] and v['m']==r['m'] and v['eta']==r['eta']);general.append(dict(r,A_analytic=a,prediction=a/r['R0'],error=diff(a/r['R0'],r['advance'])))
write('general_P_predictions.csv',general)
branches=[]
for fn in (OUT/'data/hankel_branches').glob('R[0-9]*.json'):
 d=read(fn)
 for partition,rr in d['partitions'].items():
  ref=delta(rr['events']['full'],4)
  for name,evs in rr['events'].items():branches.append({'R0':d['R0'],'eta':d['eta'],'partition_divisor':partition,'variant':name,'m4_advance':delta(evs,4),'change_from_full':diff(delta(evs,4),ref),'component_field_L2':rr['field_relative_L2'].get(name)})
write('Hankel_branch_effects.csv',branches)
maxwell=[]
for fn in (OUT/'data/maxwell').glob('R[0-9]*.json'):
 for r in read(fn)['records']:
  maxwell.append({k:r[k] for k in ['R0','eta','m','kw','fullplane_longitudinal_fraction','retained_transverse_power_relative_input','retained_fullplane_S3_relative_input']}|{'max_sampled_ROI_longitudinal_fraction':max(v['longitudinal_fraction_ROI'] for v in r['planes']),'max_sampled_flux_correction':max(abs(v['flux_relative_correction']) for v in r['planes'])})
write('Maxwell_summary.csv',maxwell)
summary={'principal_prediction_rows':len(pred),'common_moment_rows':len(moments),'input_beams':len(inputs),'translation_beams':len(two),'ellipticity_event_rows':len(ell),'generalP_beams':len(general),'final_convergence_rows':len(conv),'same_peak_FWHM_statuses':dict(collections.Counter(r['fwhm_status'] for r in moments)),'maximum_final_single_half_change':max(abs(r['single_half_difference']) for r in conv if r['single_half_difference'] is not None),'maximum_final_order_difference_change':max(abs(r['advance_difference']) for r in conv if r['advance_difference'] is not None),'input_statuses':dict(collections.Counter(r['status'] for r in inputs))}
(DEST/'analysis_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
