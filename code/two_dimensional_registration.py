from pathlib import Path
import numpy as np,json,time,datetime
from scipy.special import jnp_zeros
from scipy.interpolate import CubicSpline
from numpy.polynomial.legendre import leggauss
from observables import event
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/input_2d';DEST.mkdir(exist_ok=True)
protocol={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'groups':[[24,.35],[24,.7],[96,.35],[96,.7]],'m':[1,2,4],'relative_translation_w':[0,.01,.05,.1],'families':['plus_channel_only','opposite_half_shifts','common_half_shift'],'model':'exact-kz scalar AS; translation invariance applied to saved complex fields','aperture':'fixed scaled proxy centered on nominal axis; no claim it follows the actual zero contour','quadrature':[128,128],'validation_levels':[64,128,256],'no_new_propagation_for_translations':True}
(OUT/'config/TWO_DIMENSIONAL_REGISTRATION.json').write_text(json.dumps(protocol,indent=2));jobs=[]
def evaluate(d,dx,w,family,nr=128,nphi=128):
 tau=d['tau'];z=d['z_mm'];k=2*np.pi/.0006328;xi,wi=leggauss(nr);rr=(xi+1)/2;wr=wi/2;phi=2*np.pi*np.arange(nphi)/nphi;cp=np.cos(phi);sp=np.sin(phi);shifts={'plus_channel_only':(dx,0),'opposite_half_shifts':(dx/2,-dx/2),'common_half_shift':(dx/2,dx/2)}[family];curves={'tau':tau};records={}
 for m in [1,2,4]:
  bound=2*k*w**3*float(jnp_zeros(m,1)[0])/z;P=[]
  for iq,q in enumerate([m-1,m+1]):
   vals=[]
   for i,b in enumerate(bound):
    x=b*rr[:,None]*cp[None,:]-shifts[iq];y=b*rr[:,None]*sp[None,:];r=np.hypot(x,y);U=CubicSpline(d['rho_mm'][i],d['U'][i,q])(r);vals.append(float(np.sum(wr*rr*np.mean(abs(U)**2,axis=1))*np.pi*b*b/float(d['Pin'])))
   P.append(np.array(vals))
  Q=P[0]-P[1];T=P[0]+P[1];records[str(m)]=event(tau,Q);curves[f'm{m}_Q']=Q;curves[f'm{m}_T']=T
 return records,curves
for R,eta in protocol['groups']:
 group=f'R{R}_eta{eta:g}';d=dict(np.load(OUT/'data/model_comparison_refined'/f'{group}_exact_kz_fields.npz'));w=float(d['w_mm'])
 for frac in protocol['relative_translation_w']:
  for family in protocol['families']:
   tag=f'{group}_{family}_d{frac:g}';fn=DEST/f'{tag}.json'
   if fn.exists():jobs.append(json.loads(fn.read_text()));continue
   st=time.perf_counter();ev,curves=evaluate(d,frac*w,w,family);np.savez_compressed(DEST/f'{tag}_curves.npz',**curves);rec={'tag':tag,'R0':R,'eta':eta,'fraction_w':frac,'shift_mm':frac*w,'family':family,'events':ev,'job_status':'RUN_COMPLETED','seconds':time.perf_counter()-st};fn.write_text(json.dumps(rec,indent=2));jobs.append(rec);print(tag,rec['seconds'],flush=True)
 (DEST/'RUN_STATUS.json').write_text(json.dumps({'translation_cases_completed':len(jobs),'planned':48},indent=2))
# Resolution check at an intentionally finite displacement.
d=dict(np.load(OUT/'data/model_comparison_refined/R24_eta0.7_exact_kz_fields.npz'));check=[]
for n in [64,128,256]:
 ev,cv=evaluate(d,.1*float(d['w_mm']),float(d['w_mm']),'plus_channel_only',n,n);check.append({'quadrature':n,'events':ev})
(OUT/'baseline/translation_quadrature_check.json').write_text(json.dumps(check,indent=2))
# Exact symmetry controls under the same radial propagator.
d=dict(np.load(OUT/'data/principal_refined/R24_eta0.7_fields.npz'));U=d['U'];s3zero=.5*(abs(U[:,1])**2-abs(U[:,1])**2);s3=.5*(abs(U[:,0])**2-abs(U[:,2])**2);swapped=.5*(abs(U[:,2])**2-abs(U[:,0])**2);phase=.5*(abs(U[:,0]*np.exp(.73j))**2-abs(U[:,2])**2);rec={'m0_P1_max_abs_S3':float(abs(s3zero).max()),'P0_same_q_max_abs_S3':float(abs(s3zero).max()),'swapped_channel_sign_max_residual':float(abs(swapped+s3).max()),'uniform_phase_max_relative_S3_difference':float(abs(phase-s3).max()/abs(s3).max()),'degenerate_event_status':event(d['tau'],np.zeros(len(d['tau'])))['status'],'note':'m=0 and P=0 compare equal-magnitude orders; they do not have a positive local-domain event'};(OUT/'data/physics/symmetry_controls.json').write_text(json.dumps(rec,indent=2))
