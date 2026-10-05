from pathlib import Path
import numpy as np,json,time,gc,datetime
from scipy.special import jv,jnp_zeros
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.linalg.blas import dgemm
from spectral_solver import SpectralSolver
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/maxwell';DEST.mkdir(exist_ok=True)
protocol={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input':'given transverse field preserved on common stated spectral band','cases':[[24,.7],[192,.2],[256,.9]],'kw':[200,800,1600],'m':[1,4,6],'tau':[-1,-.5,0,.5,1,1.5,2],'ny':401,'xmax_at_zc':12,'spectral_cut':24,'nlog':'2097152 if R<192 else 4194304','du':'0.000625 if R<192 else 0.0003125','mask_relative_S0':[1e-9,1e-6],'formulas':'theory/maxwell_completion.md','quantity':'electric longitudinal fraction and normalized axial Poynting flux; not a redefinition of transverse Stokes'}
(OUT/'config/MAXWELL_COMPLETION.json').write_text(json.dumps(protocol,indent=2));records=[]
for R,eta in protocol['cases']:
 start=time.perf_counter();tag=f'R{R}_eta{eta:g}';fn=DEST/f'{tag}.json'
 if fn.exists():records+=json.loads(fn.read_text())['records'];continue
 print('START',tag,flush=True);sol=SpectralSolver(R,eta,nlog=2097152 if R<192 else 4194304);du=.000625 if R<192 else .0003125;nu=int(np.ceil(sol.umax/du))+1;nu+=1-nu%2;u=np.linspace(0,sol.umax,nu);weights=np.ones(nu);weights[1:-1:2]=4;weights[2:-1:2]=2;weights*=u[1]/3;H={}
 for q in range(8):
  _,ulog,Hlog=sol.spectra[q];H[q]=CubicSpline(ulog,Hlog)(u);H[q][0]=sol.h0 if q==0 else 0
 rr=np.linspace(0,12/np.sqrt(R),401);tau=np.array(protocol['tau']);zet=2*np.sqrt(R)+tau/np.sqrt(R);kwdata={kw:{} for kw in protocol['kw']};needs={q:[] for q in range(8)}
 for m in protocol['m']:
  a=m-1;b=m+1;needs[a]+=[(m,'Ua'),(m,'Va')];needs[b]+=[(m,'Ub'),(m,'Vb')];needs[m]+=[(m,'Ez'),(m,'Hz')]
 for q,ops in needs.items():
  if not ops:continue
  J=jv(q,rr[:,None]*u[None,:])*(weights*u)[None,:]
  for kw in protocol['kw']:
   kap=np.sqrt(kw*kw-u*u);rate=-u*u*kw/(kap+kw);phase=np.exp(1j*rate[:,None]*zet[None,:])
   for m,name in ops:
    a=m-1;b=m+1
    if name=='Ua':c=H[a]
    elif name=='Ub':c=H[b]
    elif name=='Va':c=kap/kw*H[a]+u*u/(2*kw*kap)*(H[a]-H[b])
    elif name=='Vb':c=kap/kw*H[b]+u*u/(2*kw*kap)*(H[b]-H[a])
    elif name=='Ez':c=.5j*u/kap*(H[b]-H[a])
    else:c=-u/(2*kw)*(H[a]+H[b])
    op=c[:,None]*phase;val=(dgemm(1.,J,np.asfortranarray(op.real))+1j*dgemm(1.,J,np.asfortranarray(op.imag))).T;kwdata[kw][f'm{m}_{name}']=val
  del J
 local=[]
 for kw,data in kwdata.items():
  k=sol.k;w=kw/k;Pin=2*np.pi*w*w*sol.input_integral;kap=np.sqrt(kw*kw-u*u);rho=rr*w;z=k*w*w*zet;save=dict(data,tau=tau,rho_mm=rho,z_mm=z,kw=kw,R0=R,eta=eta,umax=sol.umax,Pin=Pin)
  np.savez_compressed(DEST/f'{tag}_kw{kw}_fields.npz',**save)
  for m in protocol['m']:
   a=m-1;b=m+1;Ua=data[f'm{m}_Ua'];Ub=data[f'm{m}_Ub'];Va=data[f'm{m}_Va'];Vb=data[f'm{m}_Vb'];Ez=data[f'm{m}_Ez'];S0=(abs(Ua)**2+abs(Ub)**2)/2;S3=(abs(Ua)**2-abs(Ub)**2)/2;EL=abs(Ez)**2;flux=np.real(Ua*np.conj(Va)+Ub*np.conj(Vb))/2;bound=2*float(jnp_zeros(m,1)[0])/zet;planes=[]
   for i in range(len(tau)):
    def integ(y):cs=CubicSpline(rr,2*np.pi*w*w*rr*y/Pin).antiderivative();return float(cs(bound[i])-cs(0))
    t=integ(S0[i]);el=integ(EL[i]);qv=integ(S3[i]);fl=integ(flux[i]);roi=rr<=bound[i];mx={}
    for floor in protocol['mask_relative_S0']:
     good=roi&(S0[i]>floor*S0[i].max());mx[str(floor)]=float(np.max(EL[i,good]/(S0[i,good]+EL[i,good])))
    planes.append({'tau':float(tau[i]),'Q':qv,'T':t,'longitudinal_fraction_ROI':el/(t+el),'axial_flux_ROI':fl,'flux_relative_correction':fl/t-1,'max_bright_local_longitudinal_fraction':mx})
   den=sol.input_integral;et=simpson(u*(H[a]**2+H[b]**2),x=u)/(2*den);ez=simpson(u**3*(H[b]-H[a])**2/(4*kap**2),x=u)/den;fl=simpson(u*(kap/kw*(H[a]**2+H[b]**2)+u*u/(2*kw*kap)*(H[a]-H[b])**2),x=u)/(2*den);s3all=simpson(u*(H[a]**2-H[b]**2),x=u)/(2*den)
   rec={'R0':R,'eta':eta,'m':m,'kw':kw,'spectral_umax':sol.umax,'max_retained_sin_theta':sol.umax/kw,'retained_transverse_power_relative_input':float(et),'retained_fullplane_S3_relative_input':float(s3all),'fullplane_longitudinal_fraction':float(ez/(et+ez)),'fullplane_flux_relative_transverse_input':float(fl),'planes':planes};local.append(rec);records.append(rec)
 (DEST/f'{tag}.json').write_text(json.dumps({'records':local,'seconds':time.perf_counter()-start},indent=2));(DEST/'RUN_STATUS.json').write_text(json.dumps({'beam_kw_cases_completed':len(records),'planned':27},indent=2));print('END',tag,'seconds',time.perf_counter()-start,flush=True);del sol,H,kwdata;gc.collect()
