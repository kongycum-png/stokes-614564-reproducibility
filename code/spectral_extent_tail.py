"""Independent spectral-extent levels via coherent annular tail integration."""
from pathlib import Path
import json,numpy as np,time,gc
from scipy.interpolate import CubicSpline
from scipy.special import jv
from scipy.linalg.blas import dgemm
from spectral_solver import SpectralSolver
from run_convergence_extended import extract
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/spectral_extent';DEST.mkdir(exist_ok=True);records=[]
for R,eta in [(24,.7),(192,.2),(256,.9)]:
 tag=f'R{R}_eta{eta:g}';fn=DEST/f'{tag}.json'
 if fn.exists():records.append(json.loads(fn.read_text()));continue
 start=time.perf_counter();print('START',tag,flush=True);sol=SpectralSolver(R,eta,nlog=2097152 if R<192 else 4194304);base=dict(np.load(OUT/'data/model_comparison_refined'/f'{tag}_exact_kz_fields.npz'));rr=base['rho_mm'][0]/sol.w;zet=base['z_mm']/(sol.kw*sol.w);orders=[0,2,3,5,7];reports={};save={'tau':base['tau'],'rho_mm':base['rho_mm'],'q_orders':orders}
 for cut,lo,hi,sign in [(20,20,24,-1),(28,24,28,1)]:
  du=.00125 if R==24 else .0003125;u0=np.sqrt(lo/sol.alpha);u1=np.sqrt(hi/sol.alpha);nu=int(np.ceil((u1-u0)/du))+1;nu+=1-nu%2;u=np.linspace(u0,u1,nu);uw=np.ones(nu);uw[1:-1:2]=4;uw[2:-1:2]=2;uw*=(u[1]-u[0])/3;kap=np.sqrt(sol.kw**2-u*u);rate=-u*u*sol.kw/(kap+sol.kw);tail=np.zeros((len(zet),len(orders),len(rr)),complex)
  for iq,q in enumerate(orders):
   _,ulog,Hlog=sol.spectra[q];H=CubicSpline(ulog,Hlog)(u);J=jv(q,rr[:,None]*u[None,:])*(uw*u)[None,:]
   for j in range(0,len(zet),48):
    op=H[:,None]*np.exp(1j*rate[:,None]*zet[None,j:j+48]);tail[j:j+48,iq]=(dgemm(1.,J,np.asfortranarray(op.real))+1j*dgemm(1.,J,np.asfortranarray(op.imag))).T
  d=dict(base,U=base['U'][:,orders]+sign*tail,q_orders=np.array(orders));reports[str(cut)]={'events':extract(d),'tail_relative_field_L2':float(np.linalg.norm(tail)/np.linalg.norm(d['U'])),'annulus_spectral_cut':[lo,hi],'annulus_samples':nu};save[f'cut{cut}_coherent_tail']=tail
 np.savez_compressed(DEST/f'{tag}_tails.npz',**save);report={'R0':R,'eta':eta,'reference_cut':24,'levels':reports,'seconds':time.perf_counter()-start,'job_status':'RUN_COMPLETED'};fn.write_text(json.dumps(report,indent=2));records.append(report);(DEST/'RUN_STATUS.json').write_text(json.dumps({'completed':len(records),'planned':3},indent=2));print('END',tag,report['seconds'],flush=True);del sol,base,d,tail;gc.collect()
