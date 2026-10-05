"""Publication plots from retained evidence tables; no fitted analytic coefficients."""
from pathlib import Path
import json,csv,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from scipy.interpolate import CubicSpline
from scipy.special import jnp_zeros
OUT=Path(__file__).resolve().parents[1];F=OUT/'figures';A=OUT/'data/final_analysis'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.1,'lines.markersize':3.5})
C=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
ETA={.2:'#CC79A7',.35:'#0072B2',.7:'#D55E00',.9:'#009E73'}
MC={1:'#0072B2',2:'#D55E00',4:'#009E73',6:'#CC79A7'}
def read(name):
 rows=list(csv.DictReader(name.open()))
 for r in rows:
  for k,v in r.items():
   try:r[k]=float(v) if v else np.nan
   except ValueError:pass
 return rows
def series(rows,select,x,y):
 rr=sorted([r for r in rows if select(r)],key=lambda r:r[x]);return np.array([r[x] for r in rr]),np.array([r[y] for r in rr])
def finish(fig,axs,name):
 for i,ax in enumerate(np.ravel(axs)):ax.text(-.18,1.08,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=10);ax.tick_params(direction='out',length=3)
 for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',dpi=280,bbox_inches='tight')
 plt.close(fig)
pr=read(A/'analytic_predictions.csv');events=read(OUT/'data/analysis_refined/principal_all_events.csv');shapes=read(OUT/'data/analysis_refined/principal_shape_tests.csv');pre=json.loads((OUT/'theory/analytic_preregistration.json').read_text())
fig,axs=plt.subplots(2,3,figsize=(7.3,4.8),layout='constrained');ax=axs.ravel()
for i,eta in enumerate([.35,.7]):
 rr=sorted([r for r in pr if r['eta']==eta and r['m']==4 and r['boundary']=='scaled_1'],key=lambda r:r['R0']);R=np.array([r['R0'] for r in rr]);dv=np.array([r['advance'] for r in rr]);ax[0].plot(1/R,dv,'o',color=ETA[eta],label=fr'$\eta={eta:g}$');ax[0].plot(1/R,[r['prediction_1'] for r in rr],'--',color=ETA[eta]);ax[0].plot(1/R,[r['prediction_3'] for r in rr],'-',color=ETA[eta]);B=rr[0]['B'];CC=rr[0]['C'];A0=rr[0]['A'];ax[2].plot(R,R**1.5*(dv-A0/R),'o-',color=ETA[eta],label=fr'$\eta={eta:g}$');ax[2].axhline(B,ls='--',color=ETA[eta]);ax[3].plot(R,R**2*(dv-A0/R-B/R**1.5),'o-',color=ETA[eta]);ax[3].axhline(CC,ls='--',color=ETA[eta])
for i,m in enumerate([2,4,6]):
 rr=sorted([r for r in pr if r['eta']==.7 and r['m']==m and r['boundary']=='scaled_1'],key=lambda r:r['R0']);ax[1].plot([r['R0'] for r in rr],[r['advance']*r['R0']/r['A'] for r in rr],'o-',color=MC[m],label=f'$m={m}$')
ax[1].axhline(1,color='.4',ls=':');ax[1].legend(frameon=False);ax[0].legend(frameon=False);ax[2].legend(frameon=False)
for i,R in enumerate([24,96,256]):
 d=np.load(OUT/'data/principal_refined'/f'R{R}_eta0.7_curves.npz');t=np.linspace(-.5,.5,101);q1=CubicSpline(d['tau'],d['m1__scaled_1__Q']);qm=CubicSpline(d['tau'],d['m4__scaled_1__Q']);ax[4].plot(t,R*np.log(qm(t)*q1(0)/(q1(t)*qm(0))),color=C[i],label=f'$R_0={R}$')
predslope=next(v['D'] for v in pre['orders'] if v['m']==4)-pre['orders'][0]['D'];ax[4].plot(t,predslope*t,'k--',label='predicted');ax[4].legend(frameon=False,fontsize=6.5)
for i,eta in enumerate([.35,.7,.9]):
 xx,yy=series(pr,lambda r:r['m']==4 and r['eta']==eta and r['boundary']=='scaled_1','R0','relative_error_3');ax[5].plot(xx,100*yy,'o-',color=ETA[eta],label=fr'$\eta={eta:g}$')
ax[5].axhline(0,color='.5',lw=.6);ax[5].axvspan(128,256,color='.92',zorder=-1);ax[5].legend(frameon=False)
labels=[(r'$1/R_0$',r'$\Delta\tau_{50}(4)$','$m=4$: fixed analytic predictions'),(r'$R_0$',r'$R_0\Delta\tau_{50}/A$','$\eta=0.7$: finite orders'),(r'$R_0$',r'$R_0^{3/2}(\Delta-A/R_0)$','Third-order residual'),(r'$R_0$',r'$R_0^2(\Delta-A/R_0-B/R_0^{3/2})$','Fourth-order residual'),(r'$\tau$',r'$\mathcal{T}_4(\tau)$','No peak normalization'),(r'$R_0$','Prediction error (%)','Through $R_0^{-2}$; shaded holdout')]
for a,(x,y,tit) in zip(ax,labels):a.set(xlabel=x,ylabel=y,title=tit)
finish(fig,axs,'revision_scaling')
fig,axs=plt.subplots(2,3,figsize=(7.3,4.8),layout='constrained');ax=axs.ravel()
for i,(R,eta) in enumerate([(24,.35),(24,.7),(96,.35),(96,.7)]):
 rr=[r for r in events if r['R0']==R and r['eta']==eta and r['m']==4 and r['boundary'].startswith('scaled_')];rr.sort(key=lambda r:float(r['boundary'].split('_')[1]));ax[0].plot([float(r['boundary'].split('_')[1]) for r in rr],[100*r['boundary_change_relative'] for r in rr],'o-',color=C[i],label=fr'${R},{eta:g}$')
ax[0].axvspan(.9,1.1,color='.94',zorder=-1);ax[0].axhline(0,color='.5',lw=.6);ax[0].legend(frameon=False,title=r'$R_0,\eta$',ncol=2,fontsize=6)
fd=np.load(OUT/'data/principal_refined/R24_eta0.7_fields.npz');fc=np.load(OUT/'data/principal_refined/R24_eta0.7_curves.npz');it=int(np.argmin(abs(fd['tau'])));x=fd['x'][it];ip=.5*abs(fd['U'][it,3])**2;im=.5*abs(fd['U'][it,5])**2;s3=ip-im;norm=np.max(ip+im);show=x<=8
for values,label,color,ls in [(ip,r'$I_+$','#0072B2','-'),(im,r'$I_-$','#D55E00','--'),(s3,r'$S_3$','#009E73','-')]:ax[1].plot(x[show],values[show]/norm,ls,color=color,label=label)
ax[1].axvline(float(jnp_zeros(4,1)[0]),color='k',ls=':',lw=1,label='proxy')
ax[1].axvline(float(fc['m4__actual_root__boundary_x'][it]),color='#CC79A7',ls='-.',lw=1,label='actual zero')
ax[1].axhline(0,color='.65',lw=.6);ax[1].legend(frameon=False,fontsize=6.5,ncol=2)
d=np.load(OUT/'data/principal_refined/R24_eta0.7_curves.npz');t=d['tau'];use=(t>=-1)&(t<=2)
for i,(k,name) in enumerate([('Q','$Q/Q(0)$'),('T','$T/T(0)$'),('pbar',r'$\bar p_3/\bar p_3(0)$')]):
 v=d[f'm4__scaled_1__{k}'];ax[2].plot(t[use],v[use]/np.interp(0,t,v),color=C[i],label=name)
ax[2].legend(frameon=False);d=np.load(OUT/'data/principal_refined/R24_eta0.9_curves.npz')
for i,m in enumerate([1,4,6]):
 t=d['tau'];v=d[f'm{m}__actual_root__boundary_x'].copy();v[1:][abs(np.diff(v))>.3]=np.nan;ax[3].plot(t,v,color=MC[m],label=f'$m={m}$');ax[3].axhline(next(z['b0'] for z in pre['orders'] if z['m']==m),ls=':',color=MC[m])
ax[3].legend(frameon=False);ax[3].set_ylim(0,18)
Rs=sorted(set(r['R0'] for r in events));etas=sorted(set(r['eta'] for r in events));codes={'defined':0,'no_peak_in_valid_interval':1,'root_branch_change':2};arr=np.array([[codes[next(r['event_status'] for r in events if r['m']==4 and r['R0']==R and r['eta']==eta and r['boundary']=='actual_root')] for R in Rs] for eta in etas]);cm=ListedColormap(['#56B4E9','#E69F00','#CC79A7']);im=ax[4].imshow(arr,origin='lower',aspect='auto',cmap=cm,norm=BoundaryNorm([-.5,.5,1.5,2.5],3));ax[4].set_xticks([0,3,6,9],[f'{Rs[j]:g}' for j in [0,3,6,9]]);ax[4].set_yticks([0,2,6,9],[f'{etas[j]:g}' for j in [0,2,6,9]]);cb=fig.colorbar(im,ax=ax[4],ticks=[0,1,2],shrink=.75);cb.ax.set_yticklabels(['event','no peak','branch/gap'],fontsize=6)
for i,R in enumerate([24,96,256]):
 r=next(v for v in events if v['R0']==R and v['eta']==.7 and v['m']==4 and v['boundary']=='scaled_1');gg=np.arange(1,10)/10;ax[5].plot(gg,[R*r[f'advance_{g:.1f}'] for g in gg],'o-',color=C[i],label=f'$R_0={R}$')
ax[5].plot(gg,[-predslope*pre['airy_events']['0.7']['thresholds'][f'{g:.1f}']['K_proxy'] for g in gg],'k--',label='leading');ax[5].legend(frameon=False,fontsize=6.5)
labels=[('Boundary factor',r'Change in $\Delta\tau_{50}$ (%)','$m=4$: aperture sensitivity'),(r'$x$',r'Intensity or $S_3$ / max $S_0$',r'$R_0=24,\eta=0.7,m=4,\tau=0$'),(r'$\tau$','Value / value at $\tau=0$',r'$R_0=24,\eta=0.7,m=4$'),(r'$\tau$','First zero in $x$',r'$R_0=24,\eta=0.9$'),(r'$R_0$',r'$\eta$','$m=4$: actual-root event status'),(r'Rising fraction $\gamma$',r'$R_0\Delta\tau_\gamma(4)$',r'$\eta=0.7$: event family')]
for a,(x,y,tit) in zip(ax,labels):a.set(xlabel=x,ylabel=y,title=tit)
finish(fig,axs,'revision_boundaries')
models=read(OUT/'data/analysis_refined/model_comparison.csv');mx=read(A/'Maxwell_summary.csv');gp=read(A/'general_P_predictions.csv');fig,axs=plt.subplots(2,2,figsize=(7.3,4.6),layout='constrained');ax=axs.ravel()
for i,eta in enumerate([.2,.35,.7]):
 for aa,key in [(ax[0],'exact_minus_reduced_RS_half_tau'),(ax[1],'exact_minus_reduced_RS_advance')]:
  xx,yy=series(models,lambda r:r['m']==4 and r['eta']==eta,'R0',key);aa.plot(xx,yy,'o-',color=ETA[eta],label=fr'$\eta={eta:g}$')
for i,(R,eta) in enumerate([(24,.7),(192,.2),(256,.9)]):
 xx,yy=series(mx,lambda r:r['R0']==R and r['eta']==eta and r['m']==4,'kw','max_sampled_ROI_longitudinal_fraction');ax[2].loglog(xx,yy,'o-',color=['#D55E00','#9467BD','#56B4E9'][i],label=fr'${R},{eta:g}$')
for i,(P,m) in enumerate([(2,1),(2,4),(3,1),(3,5)]):
 rr=sorted([r for r in gp if r['P']==P and r['m']==m and r['eta']==.7],key=lambda r:r['R0']);ax[3].plot([r['R0'] for r in rr],[r['R0']*r['advance']/r['A_analytic'] for r in rr],'o-',color=C[i],label=fr'$P={P},m={m}$')
ax[3].axhline(1,color='.4',ls=':')
labels=[(r'$R_0$',r'$h_4^{\rm exact}-h_4^{\rm reduced}$','Single-order model difference'),(r'$R_0$',r'$\Delta_4^{\rm exact}-\Delta_4^{\rm reduced}$','Order-difference model effect'),(r'$kw$','Longitudinal electric fraction','$m=4$: largest sampled ROI fraction'),(r'$R_0$',r'$R_0\Delta\tau_{50}/A_{P,m}$',r'$\eta=0.7$: general polarization index')]
for a,(x,y,tit) in zip(ax,labels):a.set(xlabel=x,ylabel=y,title=tit);a.legend(frameon=False,fontsize=7)
finish(fig,axs,'revision_models_and_P')
inp=read(A/'input_tolerances.csv');fig,axs=plt.subplots(2,3,figsize=(7.3,4.8),layout='constrained');ax=axs.ravel()
for i,(R,eta) in enumerate([(24,.35),(24,.7),(96,.35),(96,.7)]):
 for j,kind in enumerate(['power','scale','ring','phase']):
  xx,yy=series(inp,lambda r:r['R0']==R and r['eta']==eta and r['m']==4 and r['kind']==kind,'value','advance');base=next(r['advance'] for r in inp if r['R0']==R and r['eta']==eta and r['m']==4 and r['kind']=='balanced');ax[j].plot(xx*100 if j<3 else xx,yy/base,'o-',color=C[i],label=fr'${R},{eta:g}$')
 rr=[r for r in inp if r['R0']==R and r['eta']==eta and r['m']==4 and r['kind']=='finite_aperture'];rr.sort(key=lambda r:r['radius_r0']);ax[4].plot([r['radius_r0'] for r in rr],[r['advance']/r['balanced_advance'] for r in rr],'o-',color=C[i]);
 for r in rr:
  if not np.isfinite(r['advance']):ax[4].plot(r['radius_r0'],.05+i*.12,'x',color=C[i],ms=6)
 tr=read(A/'translation_events.csv');rr=[r for r in tr if r['tag'].startswith(f'R{R}_eta{eta:g}_plus_channel_only_') and r['m']==4];rr.sort(key=lambda r:float(r['tag'].split('_d')[-1]));baseline=rr[0]['advance'];ax[5].plot([float(r['tag'].split('_d')[-1]) for r in rr],[r['advance']/baseline for r in rr],'o-',color=C[i])
for j in range(6):ax[j].axhline(1,color='.5',ls=':',lw=.7)
ax[1].set_yscale('symlog',linthresh=2);fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,-.01),ncol=4,frameon=False,title=r'$R_0,\eta$')
labels=[('Signed power imbalance (%)','Power imbalance'),('Differential scale (%)','Radial scale (symmetric log $y$)'),('Differential ring radius (%)','Ring radius'),('Differential phase at $r_0$ (rad)','Quadratic input phase'),('Source aperture / $r_0$','Finite source aperture'),('Plus-channel displacement / $w$','Two-dimensional displacement')]
for a,(x,tit) in zip(ax,labels):a.set(xlabel=x,ylabel=r'$\Delta_4/\Delta_{4,\rm balanced}$',title=tit)
finish(fig,axs,'input_sensitivity')
print('four figure sets written from final_analysis tables')
