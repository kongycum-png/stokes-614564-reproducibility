import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];d=json.loads((R/'data/radius_study.json').read_text())['rows'];c=np.array(json.loads((R/'data/analytic_coefficients_N14.json').read_text())[-1]['delta'])
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':11,'pdf.fonttype':42,'svg.fonttype':'none','axes.grid':False,'xtick.direction':'in','ytick.direction':'in'})
f,ax=plt.subplots(2,2,figsize=(7.8,6.4));r=np.linspace(16,100,401)
for N,color in [(4,'#999999'),(6,'#dd8844'),(10,'#5588bb'),(14,'#993355')]:
 y=sum(c[k]*r**(1-k/2) for k in range(2,N+1));ax[0,0].plot(r,y,color=color,label=f'Order {N}')
rr=np.array([x['R0'] for x in d]);v=np.array([x['scaled'] for x in d]);sel=rr<=100
ax[0,0].plot(rr[sel],v[sel],'ko',ms=3.5,label='Reduced R-S');ax[0,0].set(xlabel='$R_0$',ylabel='$R_0\Delta$',ylim=(4.21,4.35));ax[0,0].legend(fontsize=8,ncol=2,loc='upper right')
for N,col in [(4,'#999999'),(6,'#dd8844'),(10,'#5588bb'),(14,'#993355')]:
 errs=[abs(x['series'][str(N)]-x['fresnel_scaled']) for x in d];ax[0,1].loglog(rr,errs,'o-',ms=3,color=col,label=f'Order {N}')
ax[0,1].loglog(rr,[x['geometry_scaled_correction'] for x in d],'k--',label='R-S / Fresnel gap');ax[0,1].set(xlabel='$R_0$',ylabel='Absolute error in $R_0\Delta$');ax[0,1].legend(fontsize=8)
for name,style in [('reference','-'),('target','--')]:
 ax[1,0].plot(rr[sel],[x[name]['p'] for x in d if x['R0']<=100],style,color='#226699',label=f"Peak, m={1 if name=='reference' else 3}")
 ax[1,0].plot(rr[sel],[x[name]['h'] for x in d if x['R0']<=100],style,color='#aa5533',label=f"Half-height, m={1 if name=='reference' else 3}")
ax[1,0].set(xlabel='$R_0$',ylabel='Event position $\tau$');ax[1,0].legend(fontsize=8)
for k,col in [(3,'#4477aa'),(4,'#66aa88'),(5,'#dd8844'),(6,'#993355')]:
 ax[1,1].plot(r,c[k]*r**(1-k/2),color=col,label=f'$c_{k} R_0^{{{1-k/2:g}}}$')
ax[1,1].axhline(0,c='black',lw=.7);ax[1,1].set(xlabel='$R_0$',ylabel='Contributions to $R_0\Delta-A_3$');ax[1,1].legend(fontsize=8,ncol=2)
for a,l in zip(ax.ravel(),'abcd'):a.text(-.17,1.04,l,transform=a.transAxes,weight='bold',fontsize=13)
f.tight_layout(pad=1.4)
for ext in ['png','pdf','svg']:f.savefig(R/f'figures/residual_mechanism.{ext}',dpi=220)
