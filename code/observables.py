"""Boundary/event postprocessing of stored complex channels; no propagation."""
import numpy as np
from scipy.special import jnp_zeros
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.integrate import simpson
FACTORS=[.8,.85,.9,.925,.95,.975,1.,1.025,1.05,1.075,1.1,1.15,1.2]
def event(t,q,gammas=np.arange(1,10)/10):
 t=np.asarray(t);q=np.asarray(q)
 base={'status':'no_peak_in_valid_interval','window':[float(t[0]),float(t[-1])],'tau_peak':None,'q_peak':None,'thresholds':{},'fwhm':None,'centroid':None}
 if not np.isfinite(q).all():base['status']='numerical_failure';return base
 if np.max(abs(q))<1e-14:base['status']='degenerate_or_ill_conditioned_event';return base
 s=CubicSpline(t,q);critical=s.derivative().roots(extrapolate=False);maxima=[p for p in critical if t[1]<p<t[-2] and s(p,2)<0 and s(p)>1e-12]
 # Interior stationary maxima only. Tiny near-tail ripples must be recorded, not discarded.
 base['all_stationary_maxima']=[float(p) for p in maxima]
 if not maxima:return base
 p=float(maxima[0]);peak=float(s(p));base.update(status='defined',tau_peak=p,q_peak=peak,peak_curvature=float(s(p,2)))
 for gamma in gammas:
  roots=s.solve(float(gamma*peak),extrapolate=False);rise=[h for h in roots if t[0]<h<p and s(h,1)>0];fall=[h for h in roots if p<h<t[-1] and s(h,1)<0]
  # nearest preceding rising crossing of the selected peak, but retain every crossing
  h=float(rise[-1]) if rise else None
  base['thresholds'][f'{gamma:.1f}']={'tau':h,'status':'defined' if h is not None else 'no_valid_rising_crossing','all_rising_crossings':[float(v) for v in rise],'slope':float(s(h,1)) if h is not None else None,'falling_tau':float(fall[0]) if fall else None}
  if abs(gamma-.5)<1e-8 and h is not None and fall:base['fwhm']=float(fall[0]-h)
 den=float(s.integrate(t[0],t[-1]));base['window_signal_integral']=den
 if abs(den)>1e-12:base['centroid']=float(simpson(t*q,x=t)/den)
 base['centroid_is_positive_weight']=bool(np.min(q)>=0)
 return base

def profiles(d,m,P=1,base_root=None,common_root=None):
 U=d['U'];x=d['x'];rho=d['rho_mm'];tau=d['tau'];Pin=float(d['Pin'])
 orders=list(d.get('q_orders',range(U.shape[1])))
 ip=.5*abs(U[:,orders.index(abs(m-P))])**2;im=.5*abs(U[:,orders.index(abs(m+P))])**2
 if P!=1 and base_root is None:raise ValueError('P != 1 requires its independently derived boundary')
 if base_root is None:base_root=float(jnp_zeros(m,1)[0])
 z=d['z_mm'];zc=float(np.interp(0,tau,z));b1=float(jnp_zeros(1,1)[0]) if common_root is None else common_root; b=base_root
 boundaries={f'scaled_{f:g}':np.full(len(tau),f*b) for f in FACTORS}
 boundaries.update(physical_frozen=b*z/zc,physical_common=b1*z/zc)
 roots=[];candidates=[]
 for i,xx in enumerate(x):
  s=ip[i]-im[i]; floor=1e-10*np.max(abs(s)); inds=np.where((s[:-1]>0)&(s[1:]<0)&(xx[:-1]>.01)&(np.maximum(abs(s[:-1]),abs(s[1:]))>floor))[0]
  spline=CubicSpline(xx,s); r=[brentq(spline,xx[j],xx[j+1]) for j in inds]
  candidates.append(r);roots.append(r[0] if r else np.nan)
 roots=np.array(roots);boundaries['actual_root']=roots
 out={}; cumulative=[]
 for i in range(len(tau)):
  sp=CubicSpline(x[i],2*np.pi*ip[i]*rho[i]*np.gradient(rho[i],x[i],edge_order=2)/Pin).antiderivative()
  sm=CubicSpline(x[i],2*np.pi*im[i]*rho[i]*np.gradient(rho[i],x[i],edge_order=2)/Pin).antiderivative()
  cumulative.append((sp,sm))
 for label,bs in boundaries.items():
  pp=[];pm=[];neg=[]
  for i,bound in enumerate(bs):
   if not np.isfinite(bound) or bound>x[i,-1]:pp.append(np.nan);pm.append(np.nan);neg.append(np.nan);continue
   sp,sm=cumulative[i];pp.append(float(sp(bound)-sp(0)));pm.append(float(sm(bound)-sm(0)))
   # Negative S3 area within this ROI; full radial source arrays retained for refinement.
   s=ip[i]-im[i];dens=2*np.pi*np.minimum(s,0)*rho[i]*np.gradient(rho[i],x[i],edge_order=2)/Pin
   ni=CubicSpline(x[i],dens).antiderivative();neg.append(float(ni(bound)-ni(0)))
  pp=np.array(pp);pm=np.array(pm);Q=pp-pm;T=pp+pm
  out[label]={'Q':Q,'T':T,'pbar':np.divide(Q,T,out=np.full_like(Q,np.nan),where=T>1e-14),'Pplus':pp,'Pminus':pm,'negative_S3_contribution':np.array(neg),'boundary_x':bs}
 # soft window: 1 below .95b, raised-cosine transition over [.95b,1.05b], zero above
 w=np.where(x<=.95*b,1.,np.where(x>=1.05*b,0.,.5*(1+np.cos(np.pi*(x-.95*b)/(.1*b)))))
 jac=np.array([2*np.pi*rho[i]*np.gradient(rho[i],x[i],edge_order=2) for i in range(len(tau))])
 pp=np.array([simpson(w[i]*ip[i]*jac[i]/Pin,x=x[i]) for i in range(len(tau))]);pm=np.array([simpson(w[i]*im[i]*jac[i]/Pin,x=x[i]) for i in range(len(tau))])
 out['soft_cosine']={'Q':pp-pm,'T':pp+pm,'pbar':(pp-pm)/(pp+pm),'Pplus':pp,'Pminus':pm,'negative_S3_contribution':np.array([simpson(w[i]*np.minimum(ip[i]-im[i],0)*jac[i]/Pin,x=x[i]) for i in range(len(tau))]),'boundary_x':np.full(len(tau),b)}
 return out,{'roots':roots.tolist(),'all_candidates':candidates,'jump_indices':np.flatnonzero(abs(np.diff(roots))>.3).tolist(),'selection':'first nontrivial positive-to-negative sign change; all candidates retained','root_floor_relative':1e-10}

def boundary_event(t,q,root_record=None):
 """Never bridge a missing/root-jump interval with an interpolating spline."""
 t=np.asarray(t);q=np.asarray(q);finite=np.isfinite(q);breaks=np.zeros(len(q),bool)
 if root_record is not None:
  for j in root_record['jump_indices']:breaks[j+1]=True
 segments=[];start=None
 for i,valid in enumerate(finite):
  if start is not None and (not valid or breaks[i]):
   if i-start>=5:segments.append((start,i))
   start=None
  if valid and start is None:start=i
 if start is not None and len(q)-start>=5:segments.append((start,len(q)))
 results=[event(t[a:b],q[a:b]) for a,b in segments]
 selected=next((r for r in results if r['status']=='defined'),None)
 if selected is None:
  selected={'status':'root_branch_change' if root_record is not None and (not finite.all() or breaks.any()) else 'no_peak_in_valid_interval','tau_peak':None,'thresholds':{},'fwhm':None,'centroid':None}
 selected=dict(selected);selected['full_search_window']=[float(t[0]),float(t[-1])];selected['valid_segments']=[{'start':float(t[a]),'end':float(t[b-1]),'status':r['status']} for (a,b),r in zip(segments,results)];selected['missing_plane_count']=int((~finite).sum());selected['root_jumps']=int(breaks.sum())
 return selected
