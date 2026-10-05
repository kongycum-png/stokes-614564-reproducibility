from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.special import jnp_zeros
OUT=Path(__file__).resolve().parents[1];d=np.load(OUT/'data/principal_refined/R24_eta0.7_fields.npz');cv=np.load(OUT/'data/principal_refined/R24_eta0.7_curves.npz');ev=json.loads((OUT/'data/principal_refined/R24_eta0.7.json').read_text())['events'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,2,figsize=(7.3,4.7),layout='constrained');ax=axs.ravel();tau=d['tau'];use=(tau>=-1.25)&(tau<=1.5);x=np.linspace(0,8.5,401);u=np.array([CubicSpline(d['x'][i],d['U'][i,2])(x) for i in np.flatnonzero(use)]);v=np.array([CubicSpline(d['x'][i],d['U'][i,4])(x) for i in np.flatnonzero(use)]);S0=(abs(u)**2+abs(v)**2)/2;S3=(abs(u)**2-abs(v)**2)/2;norm=S0.max()
im=ax[0].pcolormesh(tau[use],x,(S0/norm).T,cmap='viridis',shading='auto',rasterized=True);fig.colorbar(im,ax=ax[0],label=r'$S_0/S_{0,\max}$');im=ax[1].pcolormesh(tau[use],x,(S3/norm).T,cmap='RdBu_r',vmin=-1,vmax=1,shading='auto',rasterized=True);fig.colorbar(im,ax=ax[1],label=r'$S_3/S_{0,\max}$')
for a in ax[:2]:a.plot(tau[use],cv['m3__actual_root__boundary_x'][use],color='#D62728',lw=1,label='actual zero');a.axhline(jnp_zeros(3,1)[0],color='k',ls='--',lw=1,label='proxy');a.set(xlabel=r'$\tau$',ylabel=r'$x=z\rho/(2kw^3)$')
ax[0].set_title('$m=3$: transverse intensity');ax[1].set_title('$m=3$: signed Stokes signal');ax[1].legend(frameon=True,facecolor='white',framealpha=.85,loc='upper left')
for color,m in zip(['#0072B2','#D55E00','#009E73'],[1,2,4]):
 q=cv[f'm{m}__scaled_1__Q'];e=ev[str(m)]['scaled_1'];p=e['tau_peak'];h=e['thresholds']['0.5']['tau'];qp=e['q_peak'];ax[2].plot(tau,q,color=color,label=f'$m={m}$');ax[2].plot(p,qp,'o',mfc='white',mec=color,ms=4);ax[3].plot(tau,q/qp,color=color);ax[3].plot(h,.5,'o',mfc='white',mec=color,ms=4)
ax[2].legend(frameon=False);ax[3].axhline(.5,color='.5',ls='--',lw=.8)
for a in ax[2:]:a.set(xlim=(-1.5,2),xlabel=r'$\tau$')
ax[2].set(ylabel=r'$Q_m$',title='Input-normalized proxy signal');ax[3].set(ylabel=r'$Q_m/Q_m(p_m)$',title='Rising half-height marker')
for i,a in enumerate(ax):a.text(-.16,1.08,'abcd'[i],transform=a.transAxes,fontweight='bold',fontsize=10)
for ext in ['pdf','svg','png']:fig.savefig(OUT/'figures'/f'revised_domain_and_event.{ext}',dpi=280,bbox_inches='tight')
np.savez_compressed(OUT/'data/final_analysis/revised_domain_figure_data.npz',tau=tau[use],x=x,S0=S0,S3=S3,common_normalization=norm)
