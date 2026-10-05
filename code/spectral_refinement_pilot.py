from pathlib import Path
import numpy as np,json,time,gc,datetime
from scipy.interpolate import CubicSpline
from scipy.special import jv
from scipy.integrate import simpson
from spectral_solver import SpectralSolver
from unexpanded_rs import FullRS
OUT=Path(__file__).resolve().parents[1]
protocol={'reason':'inverse spectral aliasing and forward log-grid error detected at R192 eta=.2','type':'numerical_refinement_not_model_change','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'levels':[[262144,.0025],[1048576,.000625],[2097152,.0003125]],'tau':[-.5,0,1],'x':[0,1,3,6],'cases':[[24,.7],[192,.2],[192,.7]],'target_relative_complex_field':1e-7,'reference':'unexpanded scalar RS independently integrated; three source resolutions at the difficult case'}
(OUT/'config/SPECTRAL_REFINEMENT_PILOT.json').write_text(json.dumps(protocol,indent=2));records=[]
for R,eta in protocol['cases']:
 tau=np.array(protocol['tau']);x=np.array(protocol['x']);rr=2*x[None,:]/(2*np.sqrt(R)+tau[:,None]/np.sqrt(R))
 ref=FullRS(R,eta,ds=.00625 if R==192 and eta==.2 else .0125,cutoff=np.ceil(20+eta*np.sqrt(R)/2)).propagate(tau,x)
 np.savez_compressed(OUT/'data/convergence'/f'unexpanded_R{R}_eta{eta:g}_refined.npz',**ref)
 for nlog,du in protocol['levels']:
  t=time.perf_counter();sol=SpectralSolver(R,eta,nlog=nlog);nu=int(np.ceil(sol.umax/du))+1;nu+=1-nu%2;u=np.linspace(0,sol.umax,nu);rate=-u*u*sol.kw/(np.sqrt(sol.kw**2-u*u)+sol.kw);U=np.zeros((len(tau),8,len(x)),complex)
  for q in range(8):
   _,ulog,Hlog=sol.spectra[q];H=CubicSpline(ulog,Hlog)(u);H[0]=sol.h0 if q==0 else 0
   for it,tt in enumerate(tau):U[it,q]=simpson(jv(q,rr[it,:,None]*u[None,:])*(u*H*np.exp(1j*rate*(2*np.sqrt(R)+tt/np.sqrt(R))))[None,:],x=u,axis=1)
  rec={'R0':R,'eta':eta,'nlog':nlog,'du':float(u[1]),'relative_L2_vs_unexpanded_RS':float(np.linalg.norm(U-ref['U'])/np.linalg.norm(ref['U'])),'per_order_relative_L2':[float(np.linalg.norm(U[:,q]-ref['U'][:,q])/np.linalg.norm(ref['U'][:,q])) for q in range(8)],'seconds':time.perf_counter()-t,'spectral_diagnostics':sol.diagnostics};records.append(rec);print({k:v for k,v in rec.items() if k not in ['spectral_diagnostics','per_order_relative_L2']},flush=True)
  np.savez_compressed(OUT/'data/convergence'/f'spectral_point_R{R}_eta{eta:g}_n{nlog}.npz',U=U,tau=tau,x=x)
  (OUT/'baseline/spectral_refinement_pilot.json').write_text(json.dumps(records,indent=2));del sol;gc.collect()
