"""Input mismatch tests; channels are separately propagated and normalized.
Signed perturbations reuse their swapped radial-input pair. No fitting or deletion.
"""
from pathlib import Path
import numpy as np,json,time,datetime,gc,hashlib
from scipy.special import airy,jnp_zeros
from scipy.interpolate import CubicSpline
from sparse_channels import SparseSolver
from revision_solver import Grid
from observables import event
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/input_mismatch';DEST.mkdir(exist_ok=True)
protocol={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'groups':[[24,.35],[24,.7],[96,.35],[96,.7]],'m':[1,2,4],'tau':[-8,8,.025],'source_cut':'ceil(20+eta sqrtR/2)','source_ds':.025,'ny':601,'xmax_at_zc':16,'power_imbalance':[-.05,-.02,-.01,-.005,0,.005,.01,.02,.05],'differential_scale':[-.02,-.01,0,.01,.02],'differential_ring_fraction':[-.01,-.005,0,.005,.01],'differential_quadratic_phase_rad_at_r0':[-.3,-.1,0,.1,.3],'finite_aperture_r0_factors':[1.5,2,3],'finite_aperture_transition_w':.5,'common_scale':[.99,1.01],'common_quadratic_phase_rad_at_r0':.3,'definitions':{'scale':'f(s/(1+delta/2)) versus f(s/(1-delta/2)); both r0 and w scale; alpha dimensionless unchanged','ring':'r0 plus/minus delta*r0/2; w and alpha fixed','phase':'opposite phases plus/minus beta*(r/r0)^2/2','normalization':'separate equal channel input powers for scale/ring/phase; actual transmitted Pin for finite common aperture','power':'intensity weights 1+epsilon and 1-epsilon, total incident power unchanged','event':'first interior positive stationary maximum in declared broad window; all maxima and crossings retained. Also report largest interior maximum as a labelled diagnostic, without replacing the first-peak definition.'},'test_levels_not_instrument_tolerances':True}
(OUT/'config/INPUT_MISMATCH_RUN.json').write_text(json.dumps(protocol,indent=2));jobs=[]
def curves(d,up,um,pin,tag,meta):
 rows={};save={'tau':d['tau']};orders=list(d['q_orders'])
 for m in protocol['m']:
  ip=.5*abs(up[:,orders.index(m-1)])**2;im=.5*abs(um[:,orders.index(m+1)])**2;qq=[];tt=[];b=float(jnp_zeros(m,1)[0])
  for i in range(len(d['tau'])):
   r=2*np.pi*d['rho_mm'][i]*np.gradient(d['rho_mm'][i],d['x'][i])/pin
   for y,out in [(r*(ip[i]-im[i]),qq),(r*(ip[i]+im[i]),tt)]:
    s=CubicSpline(d['x'][i],y).antiderivative();out.append(float(s(b)-s(0)))
  qq=np.array(qq);tt=np.array(tt);ev=event(d['tau'],qq);cs=CubicSpline(d['tau'],qq);allp=ev.get('all_stationary_maxima',[]);largest=max(allp,key=lambda p:float(cs(p))) if allp else None;ev['largest_stationary_peak_tau_diagnostic']=largest;ev['first_peak_is_largest']=bool(largest is not None and abs(largest-ev['tau_peak'])<1e-8);rows[str(m)]=ev;save[f'm{m}_Q']=qq;save[f'm{m}_T']=tt
 np.savez_compressed(DEST/f'{tag}_curves.npz',**save);rec={'tag':tag,'metadata':meta,'events':rows,'Pin':pin,'job_status':'RUN_COMPLETED'};(DEST/f'{tag}.json').write_text(json.dumps(rec,indent=2));return rec
for R,eta in protocol['groups']:
 group=f'R{R}_eta{eta:g}';start=time.perf_counter();print('START GROUP',group,flush=True);sol=SparseSolver(R,eta,[0,1,2,3,5],Grid(ds=.025,cutoff=float(np.ceil(20+eta*np.sqrt(R)/2)),ny=601,ymax=16,phase_terms=4));tau=np.linspace(-8,8,641);fields={};power={};input_meta={}
 def obtain(name,source,equalize=True):
  fn=DEST/f'{group}_input_{name}_fields.npz'
  pin=2*np.pi*sol.w**2*np.dot(sol.weights,abs(source)**2*sol.s);ratio=pin/sol.Pin;power[name]=pin;mult=np.divide(np.asarray(source,dtype=complex),sol.f,out=np.zeros_like(source,dtype=complex),where=abs(sol.f)>1e-280);omitted=float(np.max(abs(source[abs(sol.f)<=1e-280]))) if np.any(abs(sol.f)<=1e-280) else 0
  input_meta[name]={'raw_input_power_relative_baseline':ratio,'normalization_multiplier':float(np.sqrt(sol.Pin/pin)) if equalize else 1.,'omitted_source_max_amplitude_under_base_floor':omitted}
  if fn.exists():return dict(np.load(fn))
  pieces=[sol.propagate(tau[j:j+64],source_multiplier=mult) for j in range(0,len(tau),64)];d=dict(pieces[0]);d.update({k:np.concatenate([v[k] for v in pieces],axis=0) for k in ['tau','x','rho_mm','z_mm','U']});del pieces
  if equalize:d['U']*=np.sqrt(sol.Pin/pin)
  d['Pin']=sol.Pin if equalize else pin;np.savez_compressed(fn,**d,R0=R,eta=eta,raw_source_power=pin);return d
 base=obtain('balanced',sol.f);fields['balanced']=base['U'];tag=group+'_balanced';rec=curves(base,base['U'],base['U'],sol.Pin,tag,{'R0':R,'eta':eta,'kind':'balanced','source':'balanced'});jobs.append(rec)
 for imbalance in protocol['power_imbalance']:
  if imbalance==0:continue
  tag=group+f'_power{imbalance:+g}';rec=curves(base,base['U']*np.sqrt(1+imbalance),base['U']*np.sqrt(1-imbalance),sol.Pin,tag,{'R0':R,'eta':eta,'kind':'power','value':imbalance,'source':'balanced'});jobs.append(rec)
 for kind,values in [('scale',[.01,.02]),('ring',[.005,.01]),('phase',[.1,.3])]:
  for value in values:
   pair=[];ids=[]
   for sign in [1,-1]:
    name=f'{kind}{sign*value:+g}';ids.append(name)
    if kind=='scale':arg=R-sol.s/(1+sign*value/2);f=airy(arg)[0]*np.exp(sol.alpha*arg)
    elif kind=='ring':arg=R*(1+sign*value/2)-sol.s;f=airy(arg)[0]*np.exp(sol.alpha*arg)
    else:f=sol.f*np.exp(.5j*sign*value*(sol.s/R)**2)
    pair.append(obtain(name,f)['U'])
   for sign in [1,-1]:
    tag=group+f'_{kind}{sign*value:+g}';pp,mm=pair if sign==1 else pair[::-1];rec=curves(base,pp,mm,sol.Pin,tag,{'R0':R,'eta':eta,'kind':kind,'value':sign*value,'source_ids':ids if sign==1 else ids[::-1],'separate_channel_power_normalization':True});jobs.append(rec);print('CASE',tag,flush=True)
   del pair;gc.collect()
 for factor in protocol['finite_aperture_r0_factors']:
  edge=R*factor;width=.5;window=np.where(sol.s<edge-width,1.,np.where(sol.s>=edge,0.,.5*(1+np.cos(np.pi*(sol.s-edge+width)/width))));name=f'aperture{factor:g}';d=obtain(name,sol.f*window,False);tag=group+'_'+name;jobs.append(curves(d,d['U'],d['U'],float(d['Pin']),tag,{'R0':R,'eta':eta,'kind':'finite_aperture','radius_r0':factor,'transition_w':width,'transmitted_input_fraction':power[name]/sol.Pin,'source_ids':[name,name]}));del d
 for a in protocol['common_scale']:
  arg=R-sol.s/a;name=f'common_scale{a:g}';d=obtain(name,airy(arg)[0]*np.exp(sol.alpha*arg));tag=group+'_'+name;jobs.append(curves(d,d['U'],d['U'],sol.Pin,tag,{'R0':R,'eta':eta,'kind':'common_scale','scale':a,'source_ids':[name,name]}));del d
 beta=protocol['common_quadratic_phase_rad_at_r0'];name='common_phase';d=obtain(name,sol.f*np.exp(1j*beta*(sol.s/R)**2));jobs.append(curves(d,d['U'],d['U'],sol.Pin,group+'_'+name,{'R0':R,'eta':eta,'kind':'common_phase','beta':beta,'source_ids':[name,name]}));del d
 (DEST/f'{group}_sources.json').write_text(json.dumps(input_meta,indent=2));(DEST/'RUN_STATUS.json').write_text(json.dumps({'parameter_groups_completed':len(set(j['metadata']['R0'].__str__()+'_'+str(j['metadata']['eta']) for j in jobs)),'planned_groups':4,'beam_cases_completed':len(jobs)*3,'records':[j['tag'] for j in jobs]},indent=2));print('END GROUP',group,time.perf_counter()-start,flush=True);del sol,base,fields;gc.collect()
