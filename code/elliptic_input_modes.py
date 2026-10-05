"""Two-dimensional elliptical inputs, propagated by an angular Fourier expansion.
Circular aperture integrals use exact mode orthogonality; coherent 2D maps are saved.
"""
from pathlib import Path
import numpy as np,json,time,gc,datetime
from scipy.special import airy,jv,jnp_zeros
from scipy.interpolate import CubicSpline
from scipy.linalg.blas import dgemm
from observables import event
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/input_2d/ellipticity';DEST.mkdir(exist_ok=True)
protocol={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'R0':24,'eta':.7,'m':[1,4],'q':[0,2,3,5],'elliptic_log_axis_ratio':[.005,.01],'coordinate_map':'sprime=s sqrt(cos(phi)^2/a^2+a^2 sin(phi)^2); a=exp(e/2); phase q atan2(a sin(phi),cos(phi)/a)','families':['plus_channel_only','common_ellipticity','opposite_axis_ellipticity'],'model':'paraxial Fresnel, compared to matched circular-input Fresnel fields','tau':[-3,5,.025],'ny':401,'xmax_at_zc':12,'ds':.025,'source_cutoff':24,'angular_samples':256,'mode_offset_levels':[16,24,32],'criterion':'report input modal power outside retained offsets and event changes; do not drop parameter cases','aperture':'nominal circular proxy, not a distorted actual zero contour'}
(OUT/'config/ELLIPTICAL_INPUT_RUN.json').write_text(json.dumps(protocol,indent=2));R=24;eta=.7;alpha=eta/(2*np.sqrt(R));kw=2*np.pi*.08/.0006328;w=.08;K=kw/w;smax=R+24/alpha;ns=int(np.ceil(smax/.025))+1;ns+=1-ns%2;s=np.linspace(0,smax,ns);sw=np.ones(ns);sw[1:-1:2]=4;sw[2:-1:2]=2;sw*=s[1]/3;phi=2*np.pi*np.arange(256)/256;rr=np.linspace(0,12/np.sqrt(R),401);tau=np.linspace(-3,5,321);zet=2*np.sqrt(R)+tau/np.sqrt(R);source=airy(R-s)[0]*np.exp(alpha*(R-s));Pin=2*np.pi*w*w*np.dot(sw,source**2*s);fres=dict(np.load(OUT/'data/model_comparison_refined/R24_eta0.7_fresnel_fields.npz'));circular=np.array([np.array([CubicSpline(fres['rho_mm'][i]/w,fres['U'][i,q])(rr) for i in range(len(tau))]) for q in protocol['q']]);kernel={};records=[]
# The source-coordinate kernel is cached by the actual signed-mode magnitude.
for e in protocol['elliptic_log_axis_ratio']:
 start=time.perf_counter();tag=f'ellipticity{e:g}';fn=DEST/f'{tag}.json'
 if fn.exists():records.append(json.loads(fn.read_text()));continue
 print('START',tag,flush=True);a=np.exp(e/2);scale=np.sqrt(np.cos(phi)**2/a**2+a*a*np.sin(phi)**2);theta=np.arctan2(a*np.sin(phi),np.cos(phi)/a);arg=R-s[:,None]*scale[None,:];amp=airy(arg)[0]*np.exp(alpha*arg);inputpower=2*np.pi*w*w*np.dot(sw,s*np.mean(amp**2,axis=1));modes_data={};mode_lists={};inputcheck=[]
 for iq,q in enumerate(protocol['q']):
  field=amp*np.exp(1j*q*theta)[None,:];coef=np.fft.fft(field,axis=1)/256;nsigned=np.arange(q-32,q+33,2);output=np.zeros((len(tau),len(nsigned),len(rr)),complex)
  for im,n in enumerate(nsigned):
   order=abs(int(n))
   radial=coef[:,n%256];power_mode=2*np.pi*w*w*np.dot(sw,s*abs(radial)**2)
   # Use the fixed scaled-kernel coordinate y, then interpolate to fixed physical rr.
   y=np.linspace(0,16,601);afreq=y/(2*R)
   if ('y',order) not in kernel:kernel[('y',order)]=jv(order,afreq[:,None]*s[None,:])*(sw*s)[None,:]
   J=kernel[('y',order)]
   for j in range(0,len(tau),48):
    zz=zet[j:j+48];operand=radial[:,None]*np.exp(1j*s[:,None]**2/(2*zz[None,:]));total=(dgemm(1.,J,np.asfortranarray(operand.real))+1j*dgemm(1.,J,np.asfortranarray(operand.imag))).T;rho=zz[:,None]*afreq[None,:];U=(-1j)**order*(-1j/zz[:,None])*np.exp(.5j*rho*rho/zz[:,None])*total
    for it in range(len(zz)):
     if rr[-1]>rho[it,-1]:raise ValueError('Elliptical output interpolation exceeds computed field radius')
     output[j+it,im]=CubicSpline(rho[it],U[it])(rr)
  mode_lists[q]=nsigned;modes_data[q]=output;levels=[]
  allpowers=np.array([2*np.pi*w*w*np.dot(sw,s*abs(coef[:,n])**2) for n in range(256)])
  for cap in protocol['mode_offset_levels']:
   selected=np.arange(q-cap,q+cap+1,2)%256;captured=float(allpowers[selected].sum()/inputpower);levels.append({'offset_cap':cap,'captured_power_fraction':captured,'missing_input_power_fraction':max(0,1-captured)})
  inputcheck.append({'q':q,'source_power_relative_circular':float(inputpower/Pin),'levels':levels});print('CHANNEL',q,'tail',levels[-1]['missing_input_power_fraction'],flush=True)
 curves={'tau':tau};events={};maps={};phis=2*np.pi*np.arange(128)/128
 for cap in protocol['mode_offset_levels']:
  for family in protocol['families']:
   key=f'cap{cap}_{family}';events[key]={}
   for m in protocol['m']:
    qi=protocol['q'].index(m-1);qj=protocol['q'].index(m+1);plus=modes_data[m-1][:,abs(mode_lists[m-1]-(m-1))<=cap];minus=modes_data[m+1][:,abs(mode_lists[m+1]-(m+1))<=cap];Ip=.5*np.sum(abs(plus)**2,axis=1);Im=.5*abs(circular[qj])**2 if family=='plus_channel_only' else .5*np.sum(abs(minus)**2,axis=1);dens=2*np.pi*w*w*rr[None,:]*(Ip-Im)/Pin;bs=2*float(jnp_zeros(m,1)[0])/zet;Q=[]
    for it in range(len(tau)):
     cs=CubicSpline(rr,dens[it]).antiderivative();Q.append(float(cs(bs[it])-cs(0)))
    events[key][str(m)]=event(tau,Q);curves[f'{key}_m{m}_Q']=Q
    if cap==32:
     for tv in [0.,1.]:
      it=int(np.argmin(abs(tau-tv)));nplus=mode_lists[m-1];nminus=mode_lists[m+1];Up=np.einsum('nr,np->rp',modes_data[m-1][it],np.exp(1j*nplus[:,None]*phis[None,:]));
      if family=='plus_channel_only':Um=circular[qj,it,:,None]*np.exp(1j*(m+1)*phis)[None,:]
      else:
       angle=phis-(np.pi/2 if family=='opposite_axis_ellipticity' else 0);Um=np.einsum('nr,np->rp',modes_data[m+1][it],np.exp(1j*nminus[:,None]*angle[None,:]))*np.exp(1j*(m+1)*(np.pi/2 if family=='opposite_axis_ellipticity' else 0))
      maps[f'{family}_m{m}_tau{tv:g}_Iplus']=.5*abs(Up)**2;maps[f'{family}_m{m}_tau{tv:g}_Iminus']=.5*abs(Um)**2
 np.savez_compressed(DEST/f'{tag}_mode_fields.npz',tau=tau,rho_w=rr,**{f'q{q}_U':v for q,v in modes_data.items()},**{f'q{q}_orders':v for q,v in mode_lists.items()});np.savez_compressed(DEST/f'{tag}_curves.npz',**curves);np.savez_compressed(DEST/f'{tag}_maps.npz',rho_mm=rr*w,phi=phis,**maps);rec={'elliptic_log_axis_ratio':e,'input_power_checks':inputcheck,'events':events,'seconds':time.perf_counter()-start,'job_status':'RUN_COMPLETED'};fn.write_text(json.dumps(rec,indent=2));records.append(rec);(DEST/'RUN_STATUS.json').write_text(json.dumps({'completed':len(records),'planned':2},indent=2));print('END',tag,rec['seconds'],flush=True);del modes_data,coef,amp;gc.collect()
