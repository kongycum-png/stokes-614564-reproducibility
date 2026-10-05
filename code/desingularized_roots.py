"""Resolve the first Stokes zero using Uq/x^q near the vortex axis.
The axis coefficient is integrated independently from the Bessel small-x limit.
No relative intensity floor removes a mathematically resolved root.
"""
from pathlib import Path
import numpy as np,json,math
from scipy.special import airy
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.linalg.blas import dgemm
from numpy.polynomial.legendre import leggauss
OUT=Path(__file__).resolve().parents[1]
def axis_coefficients(R,eta,d,base,info):
 alpha=eta/(2*np.sqrt(R));zet=d['z_mm']/(float(d['kw'])*float(d['w_mm']));orders=list(d.get('q_orders',range(d['U'].shape[1])));mom=np.zeros((len(zet),len(orders)),complex)
 grids=[(0.,float(base['source_smax']),int(base['source_n'])),(float(base['source_smax']),float(d['source_smax']),int(d['source_n'])-int(base['source_n'])+1)]
 for lo,hi,n in grids:
  s=np.linspace(lo,hi,n);weights=np.ones(n);weights[1:-1:2]=4;weights[2:-1:2]=2;weights*=(s[1]-s[0])/3;f=airy(R-s)[0]*np.exp(alpha*(R-s));M=np.array([weights*f*s**(q+1) for q in orders])
  for j in range(0,len(zet),48):
   phase=np.exp(1j*s[:,None]**2/(2*zet[None,j:j+48]));mom[j:j+48]+=(dgemm(1.,M,np.asfortranarray(phase.real))+1j*dgemm(1.,M,np.asfortranarray(phase.imag))).T
 for i,q in enumerate(orders):mom[:,i]*=(-1j)**(q+1)/(math.factorial(q)*zet**(2*q+1))
 return mom

def resolve(d,m,axis):
 orders=list(d.get('q_orders',range(d['U'].shape[1])));a=m-1;b=m+1;ia=orders.index(a);ib=orders.index(b);nodes,weights=leggauss(128);roots=[];Q=[];T=[];intensity_at_root=[];axis_extrap_error=[];allroots=[]
 for i,x in enumerate(d['x']):
  up=d['U'][i,ia];um=d['U'][i,ib];fa=np.empty_like(up);fb=np.empty_like(um);fa[0]=axis[i,ia];fb[0]=axis[i,ib];fa[1:]=up[1:]/x[1:]**a;fb[1:]=um[1:]/x[1:]**b;sa=CubicSpline(x*x,fa);sb=CubicSpline(x*x,fb)
  # Dense near-axis sampling prevents a root inside the first observation cell
  # being confused with the trivial x=0 intensity zero.
  xx=np.unique(np.r_[0,np.geomspace(max(x[1]*1e-8,1e-12),x[1],81),x[1:]])
  def g(v):return abs(sa(v*v))**2-v**(2*(b-a))*abs(sb(v*v))**2
  yy=g(xx);ids=np.where((yy[:-1]>0)&(yy[1:]<0))[0];rr=[float(brentq(g,xx[j],xx[j+1],xtol=1e-12)) for j in ids];allroots.append(rr)
  if not rr:roots.append(np.nan);Q.append(np.nan);T.append(np.nan);intensity_at_root.append(np.nan);continue
  root=rr[0];roots.append(root);xxi=(nodes+1)*root/2;ip=.5*xxi**(2*a)*abs(sa(xxi*xxi))**2;im=.5*xxi**(2*b)*abs(sb(xxi*xxi))**2;ratio=d['rho_mm'][i,-1]/x[-1];jac=2*np.pi*ratio**2*xxi/float(d['Pin']);Q.append(float(np.dot(weights,jac*(ip-im))*root/2));T.append(float(np.dot(weights,jac*(ip+im))*root/2));S0max=.5*np.max(abs(up)**2+abs(um)**2);intensity_at_root.append(float(.5*(root**(2*a)*abs(sa(root*root))**2+root**(2*b)*abs(sb(root*root))**2)/S0max))
 return {'roots':np.array(roots),'Q':np.array(Q),'T':np.array(T),'root_relative_S0':np.array(intensity_at_root),'all_candidates':allroots}
if __name__=='__main__':
 from observables import boundary_event
 dest=OUT/'data/root_audit';dest.mkdir(exist_ok=True);records=[]
 for R,eta in [(24,.35),(24,.9),(96,.9),(256,1.)]:
  tag=f'R{R}_eta{eta:g}';d=dict(np.load(OUT/'data/principal_refined'/f'{tag}_fields.npz'));base=dict(np.load(OUT/'data/principal'/f'{tag}_fields.npz'));info=json.loads((OUT/'data/principal_refined'/f'{tag}.json').read_text());axis=axis_coefficients(R,eta,d,base,info);save={'tau':d['tau'],'axis_coefficients':axis};report={'R0':R,'eta':eta,'q0_axis_max_absolute_check':float(np.max(abs(axis[:,0]-d['U'][:,0,0]))),'m':{}}
  for m in [1,4,6]:
   v=resolve(d,m,axis);rr={'jump_indices':np.flatnonzero(abs(np.diff(v['roots']))>.3).tolist()};ev=boundary_event(d['tau'],v['Q'],rr);old=info['events'][str(m)]['actual_root'];report['m'][str(m)]={'new_event':ev,'old_event':old,'new_missing_planes':int(np.isnan(v['roots']).sum()),'old_missing_planes':old.get('missing_plane_count'),'root_below_original_sampling_cell_count':int(np.sum(v['roots']<d['x'][:,1])),'root_min':float(np.nanmin(v['roots']))};save.update({f'm{m}_{k}':v[k] for k in ['roots','Q','T','root_relative_S0']})
  np.savez_compressed(dest/f'{tag}_desingularized.npz',**save);(dest/f'{tag}_desingularized.json').write_text(json.dumps(report,indent=2));records.append(report);print(tag,'axis error',report['q0_axis_max_absolute_check'],[(m,v['new_missing_planes'],v['old_missing_planes'],v['root_below_original_sampling_cell_count']) for m,v in report['m'].items()],flush=True)
 (OUT/'baseline/desingularized_root_pilot.json').write_text(json.dumps(records,indent=2))
