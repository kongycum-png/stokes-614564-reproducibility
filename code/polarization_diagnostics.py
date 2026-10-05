"""Jones/Stokes algebra checked against actual cached complex fields."""
from pathlib import Path
import numpy as np,json
from scipy.interpolate import CubicSpline
from scipy.special import jnp_zeros
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/physics';DEST.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
def stokes(ex,ey):
 return np.array([abs(ex)**2+abs(ey)**2,abs(ex)**2-abs(ey)**2,2*np.real(np.conj(ex)*ey),2*np.imag(np.conj(ex)*ey)])
tests={}
for name,j,expect in [('horizontal',[1,0],[1,1,0,0]),('vertical',[0,1],[1,-1,0,0]),('diagonal',np.array([1,1])/np.sqrt(2),[1,0,1,0]),('antidiagonal',np.array([1,-1])/np.sqrt(2),[1,0,-1,0]),('eplus',np.array([1,1j])/np.sqrt(2),[1,0,0,1]),('eminus',np.array([1,-1j])/np.sqrt(2),[1,0,0,-1])]:tests[name]=float(np.max(abs(stokes(*j)-expect)))
d=np.load(OUT/'data/principal_refined/R24_eta0.7_fields.npz');curves=np.load(OUT/'data/principal_refined/R24_eta0.7_curves.npz');report={'basis_tests':tests,'phasor':'exp(-i omega t)','S3_convention':'2 Im(Ex* Ey)','mask_intensity_floors_relative':[1e-12,1e-9,1e-6],'orientation_mask_linear_fraction':1e-8,'checks':[]}
fig,axs=plt.subplots(2,3,figsize=(7.3,4.8));xx=np.linspace(-6.5,6.5,241);X,Y=np.meshgrid(xx,xx);rr=np.hypot(X,Y);phi=np.arctan2(Y,X);data={}
for col,m in enumerate([1,2,4]):
 for tau in [-.5,0,1.]:
  i=int(np.argmin(abs(d['tau']-tau)));u=CubicSpline(d['x'][i],d['U'][i,m-1])(rr);v=CubicSpline(d['x'][i],d['U'][i,m+1])(rr);up=u*np.exp(1j*(m-1)*phi);um=v*np.exp(1j*(m+1)*phi);ex=(up+um)/2;ey=1j*(up-um)/2;S=stokes(ex,ey);s3=np.divide(S[3],S[0],out=np.full_like(S[0],np.nan),where=S[0]>0);psi=.5*np.arctan2(S[2],S[1]);chi=.5*np.arcsin(np.clip(s3,-1,1));delta=np.angle(u*np.conj(v));prediction=phi-delta/2;lin=np.divide(np.hypot(S[1],S[2]),S[0],out=np.full_like(S[0],np.nan),where=S[0]>0);V=np.divide(abs(u)*abs(v),S[0],out=np.full_like(S[0],np.nan),where=S[0]>0)
  testphase=.73;ex2=(up*np.exp(1j*testphase)+um)/2;ey2=1j*(up*np.exp(1j*testphase)-um)/2;Sphase=stokes(ex2,ey2)
  a=.31;ia=abs(ex*np.cos(a)+ey*np.sin(a))**2;ia2=S[0]/2*(1+V*np.cos(2*(a-phi)+delta));mask=(S[0]>1e-9*S[0].max())&(lin>1e-8)
  check={'m':m,'tau':tau,'S3_channel_vs_Jones_max_relative':float(np.max(abs(S[3]-(abs(u)**2-abs(v)**2)/2))/S[0].max()),'psi_mod_pi_max_error':float(np.max(abs(np.angle(np.exp(2j*(psi[mask]-prediction[mask]))))/2)),'phase_only_S3_max_relative':float(np.max(abs(Sphase[3]-S[3]))/S[0].max()),'analyzer_projection_max_relative':float(np.max(abs(ia[mask]-ia2[mask]))/S[0].max()),'visibility_identity_max_error':float(np.max(abs(V[mask]**2+s3[mask]**2-1))),'mask_sensitivity':[{'floor':floor,'masked_fraction':float(np.mean(S[0]<=floor*S[0].max()))} for floor in [1e-12,1e-9,1e-6]]};report['checks'].append(check)
  if tau==0:
   data.update({f'm{m}_S':S,f'm{m}_chi':chi,f'm{m}_psi':psi,f'm{m}_mask':mask});ax=axs[0,col];im=ax.pcolormesh(X,Y,np.ma.masked_where(~mask,s3),vmin=-1,vmax=1,cmap='RdBu_r',shading='auto',rasterized=True)
   for iy in range(10,len(xx)-10,15):
    for ix in range(10,len(xx)-10,15):
     if not mask[iy,ix]:continue
     ax.add_patch(Ellipse((X[iy,ix],Y[iy,ix]),.35,.35*abs(np.tan(chi[iy,ix])),angle=np.degrees(psi[iy,ix]),fill=False,lw=.6,edgecolor='k'))
   proxy=float(jnp_zeros(m,1)[0]);root=curves[f'm{m}__actual_root__boundary_x'][i];ax.add_patch(Circle((0,0),proxy,fill=False,ls='--',color='#222222',lw=1.2));ax.add_patch(Circle((0,0),root,fill=False,color='#dc2626',lw=1.2));ax.set(aspect='equal',xlim=(-6.5,6.5),ylim=(-6.5,6.5),xlabel=r'$x\cos\theta$',ylabel=r'$x\sin\theta$',title=f'{chr(97+col)}  m = {m}, common $\\tau=0$')
 t=curves['tau'];Q=curves[f'm{m}__scaled_1__Q'];T=curves[f'm{m}__scaled_1__T'];pb=curves[f'm{m}__scaled_1__pbar'];ax=axs[1,col];ax.plot(t,Q/Q.max(),label='$Q/Q_{max}$',color='#2563eb');ax.plot(t,T/T.max(),label='$T/T_{max}$',color='#ea580c');ax.plot(t,pb,label='$\\bar p_3$',color='#15803d');ax.set(xlim=(-2,3),xlabel='$\\tau$',ylabel='Normalized signal / imbalance',title=f'{chr(100+col)}  m = {m}, proxy ROI');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.subplots_adjust(left=.07,right=.88,bottom=.09,top=.9,hspace=.35,wspace=.32);fig.colorbar(im,cax=fig.add_axes([.915,.55,.015,.34]),label='$S_3/S_0$');fig.suptitle('$R_0=24$, $\\eta=0.70$: solid red = actual zero; dashed = proxy',fontsize=9)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'figures/polarization_and_observable.{ext}',dpi=220)
np.savez_compressed(DEST/'polarization_maps.npz',X=X,Y=Y,**data);(DEST/'polarization_checks.json').write_text(json.dumps(report,indent=2));print('max tests',max(tests.values()),'saved actual-field maps and algebra checks')
