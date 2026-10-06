from pathlib import Path
import json,csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
OUT=Path(__file__).resolve().parents[1];D=OUT/'data/measurement';plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False});colors=['#0072B2','#D55E00','#009E73','#CC79A7'];fig,axs=plt.subplots(2,3,figsize=(7.3,4.6),layout='constrained');ax=axs.ravel();raw=np.load(D/'R24_eta0.7_baseline_raw_pixels.npz');xs=np.unique(raw['x_mm']);ys=np.unique(raw['y_mm']);ix=np.searchsorted(xs,raw['x_mm']);iy=np.searchsorted(ys,raw['y_mm']);vmin=min(-3,float(raw['raw_counts'].min()));vmax=float(raw['raw_counts'].max())
for c in range(2):
 image=np.full((len(ys),len(xs)),np.nan);image[iy,ix]=raw['raw_counts'][c];im=ax[c].imshow(image,origin='lower',extent=[xs[0]*1e3,xs[-1]*1e3,ys[0]*1e3,ys[-1]*1e3],cmap='viridis',vmin=vmin,vmax=vmax);ax[c].set(title=['Raw $I_+$ counts','Raw $I_-$ counts'][c],xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)')
fig.colorbar(im,ax=ax[:2],label='electrons / pixel',shrink=.8,pad=.02)
s=np.load(D/'R24_eta0.7_baseline_samples.npz');tau=s['tau'];cnt=s['raw_aggregate_Iplus_Iminus'][0,2];Q=(cnt[0]-cnt[1])/s['reference_monitor'][0];step=tau[1]-tau[0];win=max(5,int(round(.25/step)));win+=1-win%2;expect=s['expected_raw_aggregate'][0,2];ax[2].plot(tau,Q,color='.7',lw=.6,label='count difference');ax[2].plot(tau,savgol_filter(Q,win,3),color=colors[0],lw=1,label='smoothed');ax[2].plot(tau,(expect[0]-expect[1])/1e6,'--',color=colors[1],lw=1,label='conditional mean');ax[2].set(xlabel=r'$\tau$',ylabel=r'$\widehat Q$',title='$m=4$, one synthetic scan');ax[2].legend(frameon=False,loc='upper left')
rows=[]
for ic,(R,e) in enumerate([(24,.35),(24,.7),(96,.35),(96,.7)]):
 Ns=[1e4,1e5,1e6,1e7];rates=[]
 for N in Ns:
  name='baseline' if N==1e6 else f'Nincident_{N:g}';v=json.loads((D/f'R{R}_eta{e:g}_{name}.json').read_text());rates.append(v['orders']['4']['advance_sign_recovery_rate'])
 ax[3].semilogx(Ns,rates,'o-',ms=3,color=colors[ic],label=fr'$R_0={R},\eta={e:g}$')
ax[3].set(xlabel='Incident reference counts / plane',ylabel='Correct advance sign fraction',ylim=(-.03,1.04),title='$m=4$: count sweep');fig.legend(*ax[3].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,-.01),ncol=4,frameon=False,fontsize=7)
for i,(name,label) in enumerate([('baseline','$10^6$ counts'),('Nincident_1e+07','$10^7$ counts')]):
 s=np.load(D/f'R96_eta0.7_{name}_samples.npz');delta=s['half_z_minus_zc_mm'][:,0]-s['half_z_minus_zc_mm'][:,2];delta=np.sort(delta[np.isfinite(delta)]);ax[4].step(delta,np.arange(1,len(delta)+1)/len(delta),color=colors[i],label=label)
v=json.loads((D/'R96_eta0.7_baseline.json').read_text());ax[4].axvline(v['orders']['4']['delta_ideal_mm'],color='k',ls=':',lw=1,label='ideal');ax[4].axvline(0,color='.5',lw=.6);ax[4].set(xlabel=r'Estimated $\Delta z_4$ (mm)',ylabel='Cumulative fraction',title=r'$R_0=96,\eta=0.7$');ax[4].legend(frameon=False,loc='lower right')
for ic,(R,e) in enumerate([(24,.35),(24,.7),(96,.35),(96,.7)]):
 gs=[0,.001,.005,.01];means=[]
 for g in gs:
  name='baseline' if g==.001 else f'gain_sd_{g:g}';v=json.loads((D/f'R{R}_eta{e:g}_{name}.json').read_text());means.append(v['orders']['4']['delta_CI_width_mm'])
 ax[5].plot(np.array(gs)*100,means,'o-',ms=3,color=colors[ic])
ax[5].set(xlabel='Relative gain SD (%)',ylabel='95% interval width (mm)',title='$m=4$: gain covariance')
for i,a in enumerate(ax):a.text(-.18,1.06,'abcdef'[i],transform=a.transAxes,fontweight='bold',fontsize=10)
for ext in ['pdf','svg','png']:fig.savefig(OUT/'figures'/f'synthetic_measurement.{ext}',dpi=250,bbox_inches='tight')
# Summarize every retained scenario.
for fn in sorted(D.glob('R[0-9]*.json')):
 if '_pilot' in fn.stem or '_stability' in fn.stem:continue
 v=json.loads(fn.read_text())
 for m,d in v['orders'].items():
  if m=='1':continue
  rows.append(dict(tag=v['tag'],m=int(m),**v['config'],**d))
keys=list(dict.fromkeys(k for r in rows for k in r))
with (OUT/'data/analysis/measurement_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
print('figure and',len(rows),'paired measurement records written')
