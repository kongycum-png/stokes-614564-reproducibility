from pathlib import Path
import numpy as np,json,time
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from spectral_solver import SpectralSolver
OUT=Path(__file__).resolve().parents[1];R=24;eta=.7;kw=200;sol=SpectralSolver(R,eta,nlog=2097152,kw=kw);nu=int(np.ceil(sol.umax/.00125))+1;nu+=1-nu%2;u=np.linspace(0,sol.umax,nu);kap=np.sqrt(kw*kw-u*u);rate=-u*u*kw/(kap+kw);H={}
for q in [0,2,3,5,7]:
 _,ulog,Hlog=sol.spectra[q];H[q]=CubicSpline(ulog,Hlog)(u);H[q][0]=sol.h0 if q==0 else 0
stored=dict(np.load(OUT/'data/maxwell/R24_eta0.7_kw200_fields.npz'));rgrid=stored['rho_mm']/(kw/sol.k);it=2;zet=2*np.sqrt(R);records=[]
for m in [1,4,6]:
 for x,theta in [(1.,.23),(3.,1.12),(6.,2.1)]:
  r=x/np.sqrt(R);a=m-1;b=m+1;Ua=CubicSpline(rgrid,stored[f'm{m}_Ua'][it])(r)*np.exp(1j*a*theta);Ub=CubicSpline(rgrid,stored[f'm{m}_Ub'][it])(r)*np.exp(1j*b*theta);Va=CubicSpline(rgrid,stored[f'm{m}_Va'][it])(r)*np.exp(1j*a*theta);Vb=CubicSpline(rgrid,stored[f'm{m}_Vb'][it])(r)*np.exp(1j*b*theta);target=np.array([(Ua+Ub)/2,.5j*(Ua-Ub),CubicSpline(rgrid,stored[f'm{m}_Ez'][it])(r)*np.exp(1j*m*theta),.5j*(Vb-Va),(Va+Vb)/2,CubicSpline(rgrid,stored[f'm{m}_Hz'][it])(r)*np.exp(1j*m*theta)])
  for nphi in [64,128,256]:
   phi=2*np.pi*np.arange(nphi)/nphi;cp=np.cos(phi);sp=np.sin(phi);integ=np.zeros((6,nu),complex)
   for j in range(0,nu,1024):
    sl=slice(j,j+1024);U=u[sl,None];K=kap[sl,None];fp=(-1j)**a*H[a][sl,None]*np.exp(1j*a*phi)[None,:];fm=(-1j)**b*H[b][sl,None]*np.exp(1j*b*phi)[None,:];ex=(fp+fm)/2;ey=.5j*(fp-fm);ez=-(U*cp[None,:]*ex+U*sp[None,:]*ey)/K;hx=(U*sp[None,:]*ez-K*ey)/kw;hy=(K*ex-U*cp[None,:]*ez)/kw;hz=(U*cp[None,:]*ey-U*sp[None,:]*ex)/kw;phase=np.exp(1j*(U*r*np.cos(phi-theta)[None,:]+rate[sl,None]*zet));integ[:,sl]=np.array([np.mean(v*phase,axis=1)*u[sl] for v in [ex,ey,ez,hx,hy,hz]])
   actual=simpson(integ,x=u,axis=1);records.append({'m':m,'x':x,'theta':theta,'nphi':nphi,'relative_EH_L2_difference':float(np.linalg.norm(actual-target)/np.linalg.norm(target)),'relative_Ez_difference':float(abs(actual[2]-target[2])/max(abs(target[2]),1e-20)),'relative_Hz_difference':float(abs(actual[5]-target[5])/max(abs(target[5]),1e-20))})
(OUT/'baseline/maxwell_angular_check.json').write_text(json.dumps({'reference':'direct Cartesian k cross E, k dot E=0 and angular quadrature; compared to independent radial Bessel recurrence formulas','case':[R,eta,kw],'records':records,'max_EH_relative_difference':max(r['relative_EH_L2_difference'] for r in records),'max_Ez_relative_difference':max(r['relative_Ez_difference'] for r in records)},indent=2));print('max EH',max(r['relative_EH_L2_difference'] for r in records),'max Ez',max(r['relative_Ez_difference'] for r in records))
