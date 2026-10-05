"""Numerical separation of high-u Hankel branches and low-u endpoint remainder.
A declared smooth partition is necessary: separate Hankel functions are nonuniform
at u=0 and at the source axis. Coherent fields, not incoherent powers, are compared.
"""
from pathlib import Path
import numpy as np,json,time,gc,datetime
from scipy.special import airy,jv,hankel1
from scipy.signal import ZoomFFT
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.linalg.blas import dgemm
from spectral_solver import SpectralSolver
from observables import event
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/hankel_branches_eta035';DEST.mkdir(exist_ok=True)
protocol={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':[[24,.35],[32,.35],[40,.35],[56,.35]],'q':[0,2,4],'m':[1,3],'source_axis_partition':'zero below R-10; raised cosine to one at R-8; omitted core remains in remainder','spectral_partition':'zero below ua=sqrtR/divisor; raised cosine to one at 2ua','divisors':[6,4,3],'Debye_terms':[8,12],'source_ds':.0125,'source_cut':'ceil(22+alpha R)','inverse_du':.0003125,'tau':[-3,5,.025],'ny':301,'xmax_at_zc':12,'interpretation':'partition-dependent components; no globally uniform individual Hankel branch at endpoint; compare coherent sums','CZT_method':'ZoomFFT on a uniform source grid, conjugated for positive Fourier sign'}
(OUT/'config/HANKEL_BRANCHES_ETA035.json').write_text(json.dumps(protocol,indent=2));allrec=[]
for R,eta in protocol['cases']:
 start=time.perf_counter();tag=f'R{R}_eta{eta:g}';fn=DEST/f'{tag}.json'
 if fn.exists():allrec.append(json.loads(fn.read_text()));continue
 print('START',tag,flush=True);alpha=eta/(2*np.sqrt(R));smax=R+np.ceil(22+alpha*R)/alpha;ns=int(np.ceil(smax/.0125))+1;ns+=1-ns%2;s=np.linspace(0,smax,ns);sw=np.ones(ns);sw[1:-1:2]=4;sw[2:-1:2]=2;sw*=s[1]/3;f=airy(R-s)[0]*np.exp(alpha*(R-s));part=np.where(s<R-10,0,np.where(s>R-8,1,.5*(1-np.cos(np.pi*(s-R+10)/2))));ff=f*part;sol=SpectralSolver(R,eta,nlog=4194304 if R>=192 else 2097152);nu=int(np.ceil(sol.umax/.0003125))+1;nu+=1-nu%2;u=np.linspace(0,sol.umax,nu);du=u[1];uw=np.ones(nu);uw[1:-1:2]=4;uw[2:-1:2]=2;uw*=du/3;zoom=ZoomFFT(ns,[0,sol.umax/(2*np.pi)],m=nu,fs=1/s[1],endpoint=True);safe=np.maximum(s,1e-12);FT=np.array([np.conj(zoom(sw*ff*safe**(.5-n))) for n in range(12)]);valid=u>=np.sqrt(R)/6;hp={};errors=[]
 for q in protocol['q']:
  sums=[];v=np.zeros(nu,complex);coef=1.
  for n in range(12):
   if n:coef*= (4*q*q-(2*n-1)**2)/(8*n)
   v[valid]+=1j**n*coef*FT[n,valid]/u[valid]**n
   if n+1 in protocol['Debye_terms']:
    vv=np.zeros(nu,complex);vv[valid]=np.exp(-1j*(np.pi*q/2+np.pi/4))/np.sqrt(2*np.pi*u[valid])*v[valid];sums.append(vv)
  hp[q]=sums[-1];spots=np.array([np.sqrt(R)/6,np.sqrt(R)/3,np.sqrt(R),1.5*np.sqrt(R)]);checks=[]
  for us in spots:
   good=s>0;direct=.5*np.sum(sw[good]*ff[good]*s[good]*hankel1(q,us*s[good]));pred=CubicSpline(u,hp[q])(us) if us>=u[np.flatnonzero(valid)[0]] else None
   checks.append({'u':float(us),'direct_real':float(direct.real),'direct_imag':float(direct.imag),'relative_error':float(abs(pred-direct)/max(abs(direct),1e-30)) if pred is not None else None})
  errors.append({'q':q,'Debye_8_to_12_relative_spectrum_L2':float(np.linalg.norm((sums[1]-sums[0])[valid])/np.linalg.norm(sums[1][valid])),'direct_Hankel_checks':checks})
 tau=np.linspace(-3,5,321);zet=2*np.sqrt(R)+tau/np.sqrt(R);rr=np.linspace(0,12/np.sqrt(R),301);rate=-u*u/2;fields={div:{name:np.zeros((321,3,301),complex) for name in ['full','inward','outward','endpoint']} for div in protocol['divisors']}
 for iq,q in enumerate(protocol['q']):
  _,ulog,Hlog=sol.spectra[q];H=CubicSpline(ulog,Hlog)(u);H[0]=sol.h0 if q==0 else 0.;J=jv(q,rr[:,None]*u[None,:])*(uw*u)[None,:]
  for div in protocol['divisors']:
   ua=np.sqrt(R)/div;chi=np.where(u<=ua,0,np.where(u>=2*ua,1,.5*(1-np.cos(np.pi*(u-ua)/ua))));parts={'full':H,'inward':chi*hp[q],'outward':chi*np.conj(hp[q]),'endpoint':(1-chi)*H}
   for name,A in parts.items():
    if name=='full' and div!=protocol['divisors'][0]:fields[div][name][:,iq]=fields[protocol['divisors'][0]][name][:,iq];continue
    for j in range(0,len(tau),64):
     op=A[:,None]*np.exp(1j*rate[:,None]*zet[None,j:j+64]);fields[div][name][j:j+64,iq]=(dgemm(1.,J,np.asfortranarray(op.real))+1j*dgemm(1.,J,np.asfortranarray(op.imag))).T
  del J
 report={'R0':R,'eta':eta,'method_checks':errors,'partitions':{}}
 for div,F in fields.items():
  F['high_u_remainder']=F['full']-F['inward']-F['outward']-F['endpoint'];norms={name:float(np.linalg.norm(v)/np.linalg.norm(F['full'])) for name,v in F.items()};ev={};saved={'tau':tau,'rho_w':rr,'q_orders':np.array(protocol['q'])};variants={'full':F['full'],'inward':F['inward'],'without_outward':F['full']-F['outward'],'without_endpoint':F['full']-F['endpoint'],'without_high_u_remainder':F['full']-F['high_u_remainder']}
  for name,U in variants.items():
   ev[name]={}
   for m in protocol['m']:
    a=protocol['q'].index(m-1);b=protocol['q'].index(m+1);dens=np.pi*sol.w**2*rr[None,:]*(abs(U[:,a])**2-abs(U[:,b])**2)/sol.Pin;root=2*float(__import__('scipy').special.jnp_zeros(m,1)[0])/zet;Q=[]
    for i in range(len(tau)):
     cs=CubicSpline(rr,dens[i]).antiderivative();Q.append(float(cs(root[i])-cs(0)))
    ev[name][str(m)]=event(tau,Q);saved[f'{name}_m{m}_Q']=Q
  np.savez_compressed(DEST/f'{tag}_div{div}_fields.npz',**F,**saved);report['partitions'][str(div)]={'field_relative_L2':norms,'events':ev}
 report['seconds']=time.perf_counter()-start;fn.write_text(json.dumps(report,indent=2));allrec.append(report);(DEST/'RUN_STATUS.json').write_text(json.dumps({'completed':len(allrec),'planned':4},indent=2));print('END',tag,report['seconds'],flush=True);del sol,FT,hp,fields;gc.collect()
