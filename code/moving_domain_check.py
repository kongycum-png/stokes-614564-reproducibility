"""Independent fixed-physical-radius derivative and paraxial boundary current."""
from pathlib import Path
import numpy as np,json
from scipy.interpolate import CubicSpline
from scipy.special import jnp_zeros
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/physics';DEST.mkdir(exist_ok=True)
d=np.load(OUT/'data/model_comparison/R24_eta0.7_fresnel_fields.npz');cv=np.load(OUT/'data/model_comparison/R24_eta0.7_fresnel_curves.npz');z=d['z_mm'];tau=d['tau'];rho=d['rho_mm'][0];U=d['U'];Pin=float(d['Pin']);k=2*np.pi/.0006328;Lc=(z[-1]-z[0])/(tau[-1]-tau[0]);zc=np.interp(0,tau,z);records=[];data={'tau':tau};fig,axs=plt.subplots(1,3,figsize=(10,3.2));window=(tau>=-.5)&(tau<=1.25)
for col,m in enumerate([1,2,4]):
 s=.5*(abs(U[:,m-1])**2-abs(U[:,m+1])**2);dsdz=CubicSpline(z,s,axis=0).derivative()(z);du=CubicSpline(rho,U[:,[m-1,m+1]],axis=2).derivative()(rho);j=(np.imag(np.conj(U[:,m-1])*du[:,0])-np.imag(np.conj(U[:,m+1])*du[:,1]))/(2*k)
 for label in ['scaled_1','physical_frozen','actual_root']:
  Q=cv[f'm{m}__{label}__Q'];bx=cv[f'm{m}__{label}__boundary_x'];b=bx*rho[-1]/d['x'][:,-1];finite=np.isfinite(b)&np.isfinite(Q)
  # The comparison interval is fixed in advance. No bridging of missing roots.
  if not finite[window].all():records.append({'m':m,'boundary':label,'status':'undefined_in_predeclared_comparison_interval'});continue
  indices=np.flatnonzero(finite);lo,hi=indices[0],indices[-1]+1
  if not finite[lo:hi].all():lo=np.flatnonzero(window)[0]-2;hi=np.flatnonzero(window)[-1]+3
  bp=np.zeros_like(z)
  if label=='scaled_1':bp=-b/z
  elif label=='actual_root':bp[lo:hi]=CubicSpline(z[lo:hi],b[lo:hi]).derivative()(z[lo:hi])
  bulk=np.full_like(z,np.nan);move=bulk.copy();flux=bulk.copy();total=bulk.copy();total[lo:hi]=CubicSpline(z[lo:hi],Q[lo:hi]).derivative()(z[lo:hi])*Lc
  for i in range(lo,hi):
   integral=CubicSpline(rho,dsdz[i]*rho).antiderivative();bulk[i]=2*np.pi/Pin*(integral(b[i])-integral(0))*Lc;move[i]=2*np.pi/Pin*b[i]*bp[i]*CubicSpline(rho,s[i])(b[i])*Lc;flux[i]=-2*np.pi/Pin*b[i]*CubicSpline(rho,j[i])(b[i])*Lc
  norm=np.linalg.norm(total[window]);record={'m':m,'boundary':label,'status':'checked','tau_window':[-.5,1.25],'leibniz_relative_L2':float(np.linalg.norm((total-bulk-move)[window])/norm),'paraxial_current_relative_L2':float(np.linalg.norm((total-flux-move)[window])/norm),'max_absolute_error_dQdtau':float(np.max(abs((total-flux-move)[window]))),'moving_contribution_relative_L2':float(np.linalg.norm(move[window])/norm),'moving_term_at_tau0':float(move[np.argmin(abs(tau))]),'current_term_at_tau0':float(flux[np.argmin(abs(tau))])};records.append(record)
  for name,v in [('total',total),('bulk',bulk),('moving',move),('current',flux)]:data[f'm{m}_{label}_{name}']=v
  if label=='scaled_1':
   ax=axs[col];ax.plot(tau,total,label=r'$dQ/d\tau$',color='k');ax.plot(tau,flux,label='radial current',color='#2563eb');ax.plot(tau,move,label='moving aperture',color='#ea580c');ax.plot(tau,flux+move,'--',label='sum',color='#15803d');ax.set(xlim=(-.5,1.25),xlabel=r'$\tau$',ylabel=r'$dQ/d\tau$',title=f'm = {m}');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2)
axs[0].legend(fontsize=7);fig.tight_layout()
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figures/moving_domain_decomposition.{ext}',dpi=220)
np.savez_compressed(DEST/'moving_domain_terms.npz',**data);(DEST/'moving_domain_check.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
