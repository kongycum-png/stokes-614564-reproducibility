"""Declared observation-domain extension; old event definitions/results retained.
Reuses every available inner field plane. New propagation supplies the outer
annulus and previously unobserved axial planes, not duplicate baseline fields.
"""
from pathlib import Path
import json,numpy as np,time,gc,datetime,math
from scipy.special import jnp_zeros
from scipy.interpolate import CubicSpline
from scipy.linalg.blas import dgemm
from sparse_channels import SparseSolver
from revision_solver import Grid
from desingularized_roots import resolve
from observables import event,boundary_event
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/adaptive_roots';DEST.mkdir(exist_ok=True)
protocol={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'distinguish finite observation extent, first-root branch changes and event disappearance','change_type':'explicit observation-domain extension; previous window results preserved','cases':[[24,.35],[24,.7],[24,.9],[96,.85],[96,.9],[96,.925],[96,.95],[256,1.]],'m':[1,4,6],'tau':[-8,12,.025],'y':[0,32,1201],'source_ds':.025,'cutoff':'ceil(16+alpha R)','root':'desingularized Uq/x^q; no signal floor; all candidates retained','branch_jump_diagnostic':.3,'claimed_scope':'finite search windows only; no global absence inferred'}
(OUT/'config/ADAPTIVE_ROOT_EXTENT.json').write_text(json.dumps(protocol,indent=2));records=[]
for R,eta in protocol['cases']:
 tag=f'R{R}_eta{eta:g}';fn=DEST/f'{tag}.json'
 if fn.exists():records.append(json.loads(fn.read_text()));continue
 start=time.perf_counter();print('START',tag,flush=True);orders=[0,2,3,5,7];tau=np.linspace(-8,12,801);cut=np.ceil(16+eta*np.sqrt(R)/2);sol=SparseSolver(R,eta,orders,Grid(ds=.025,cutoff=cut,ny=1201,ymax=32,phase_terms=4));oldfn=OUT/'data/principal_refined'/f'{tag}_fields.npz';cache=oldfn.exists();keys=['y','a','c','dc','kernel'];full={k:getattr(sol,k) for k in keys}
 def region(sl):
  for k,v in full.items():setattr(sol,k,v[:,sl] if k=='kernel' else v[sl])
 def propagate(tt):
  chunks=[sol.propagate(t) for t in np.array_split(tt,max(1,int(np.ceil(len(tt)/48))))];d=dict(chunks[0])
  for k in ['tau','x','rho_mm','z_mm','U']:d[k]=np.concatenate([c[k] for c in chunks],axis=0)
  return d
 if cache:
  old=dict(np.load(oldfn));region(slice(600,None));outer=propagate(tau);region(slice(0,601));sel=(tau>=-3-1e-9)&(tau<=5+1e-9);extra=propagate(tau[~sel]);d=dict(outer);d['U']=np.zeros((len(tau),len(orders),1201),complex);d['x']=np.zeros((len(tau),1201));d['rho_mm']=np.zeros_like(d['x'])
  for k in ['U','x','rho_mm']:
   if k=='U':d[k][sel,:,:601]=old[k][:,orders];d[k][~sel,:,:601]=extra[k];d[k][:,:,601:]=outer[k][:,:,1:]
   else:d[k][sel,:601]=old[k];d[k][~sel,:601]=extra[k];d[k][:,601:]=outer[k][:,1:]
  seam=float(np.linalg.norm(outer['U'][sel,:,0]-old['U'][:,orders,-1])/max(np.linalg.norm(old['U'][:,orders,-1]),1e-30));del old,outer,extra
 else:region(slice(None));d=propagate(tau);seam=None
 d.update(R0=R,eta=eta,kw=sol.kw,w_mm=sol.w);zet=d['z_mm']/(sol.kw*sol.w);mom=np.zeros((len(tau),len(orders)),complex);M=np.array([sol.weights*sol.f*sol.s**(q+1) for q in orders])
 for j in range(0,len(tau),48):
  phase=np.exp(1j*sol.s[:,None]**2/(2*zet[None,j:j+48]));mom[j:j+48]=(dgemm(1.,M,np.asfortranarray(phase.real))+1j*dgemm(1.,M,np.asfortranarray(phase.imag))).T
 for i,q in enumerate(orders):mom[:,i]*=(-1j)**(q+1)/(math.factorial(q)*zet**(2*q+1))
 cv={'tau':tau,'axis_coefficients':mom};report={'R0':R,'eta':eta,'reused_inner_cache':cache,'seam_relative_L2':seam,'events':{},'root_records':{},'source_cutoff':float(cut)}
 for m in protocol['m']:
  v=resolve(d,m,mom);rr={'jump_indices':np.flatnonzero(abs(np.diff(v['roots']))>.3).tolist()};Q=[]
  for it in range(len(tau)):
   den=np.pi*(abs(d['U'][it,orders.index(m-1)])**2-abs(d['U'][it,orders.index(m+1)])**2)*d['rho_mm'][it]/float(d['Pin']);cs=CubicSpline(d['x'][it],den*(d['rho_mm'][it,-1]/d['x'][it,-1])).antiderivative();Q.append(float(cs(jnp_zeros(m,1)[0])-cs(0)))
  report['events'][str(m)]={'actual_root':boundary_event(tau,v['Q'],rr),'proxy':event(tau,Q)};report['root_records'][str(m)]=dict(rr,all_candidates=v['all_candidates'],missing_planes=int(np.isnan(v['roots']).sum()),minimum_root=float(np.nanmin(v['roots'])),maximum_root=float(np.nanmax(v['roots'])));cv[f'm{m}_proxy_Q']=Q
  for k in ['roots','Q','T','root_relative_S0']:cv[f'm{m}_{k}']=v[k]
 np.savez_compressed(DEST/f'{tag}_fields.npz',**d);np.savez_compressed(DEST/f'{tag}_curves.npz',**cv);report.update(seconds=time.perf_counter()-start,job_status='RUN_COMPLETED');fn.write_text(json.dumps(report,indent=2));records.append(report);(DEST/'RUN_STATUS.json').write_text(json.dumps({'completed':len(records),'planned':len(protocol['cases'])},indent=2));print('END',tag,report['seconds'],[(m,v['actual_root']['status'],v['actual_root'].get('tau_peak')) for m,v in report['events'].items()],flush=True);del sol,full,d,cv;gc.collect()
