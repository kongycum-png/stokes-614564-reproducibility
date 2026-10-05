"""Count-level synthetic measurement sweeps, including correlated nuisance errors.
These are synthetic conditions, not specifications of an instrument.
"""
from pathlib import Path
import numpy as np,json,time,hashlib,datetime,sys,gc
from measurement_detector import detector_response,shifted_means
from measurement_estimator import estimate
from scipy.linalg.blas import dgemm
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/measurement';DEST.mkdir(exist_ok=True)
P=json.loads((OUT/'config/SYNTHETIC_MEASUREMENT.json').read_text());MS=[1,2,4]
def scenarios():
 all=[('baseline',dict(P['baseline']))]
 for key,vals in P['marginals'].items():
  for val in vals:
   if val==P['baseline'][key]:continue
   all.append((f'{key}_{val:g}',dict(P['baseline'],**{key:val})))
 for v in P['interactions']:
  d=dict(v);name=d.pop('name');all.append((name,dict(P['baseline'],**d)))
 return all

def covariance(response,cfg):
 mu=response['mu'];der=response['dmu_dz'];t=response['tau'];N=cfg['Nincident'];c=cfg['polarimetric_cross_talk_mean'];g=cfg.get('gain_mean',0.);T=mu.sum(axis=1);Q=(1-2*c)*(mu[:,0]-mu[:,1])+g*T
 G=[];means=[];condition=[]
 for m in range(3):
  e=estimate(t,Q[m],True);G.append(e['gradient_tau_per_Q']*response['Lc']);means.append(float(e['half_tau'][0]*response['Lc']));condition.append({'smoothing_samples':e['window_samples'],'extra_maxima':int(e['n_local_maxima'][0]-1),'defined':bool(e['valid'][0])})
 G=np.array(G);npix=response['npixels'];optical_T=T+g*(1-2*c)*(mu[:,0]-mu[:,1]);shot=(N*optical_T+2*cfg['background_e_per_pixel']*npix+2*cfg['read_noise_e_per_pixel']**2*npix)/N**2
 C=np.diag(np.sum(G*G*shot,axis=1));parts={'photon_background_read':C.copy()}
 mon=dgemm(1.,G*Q,(G*Q).T)/(cfg['monitor_reference_count_multiplier']*N);C+=mon;parts['monitor_shared_per_plane']=mon
 vgain=np.sum(G*T,axis=1);rho=cfg['gain_cross_order_correlation'];cg=cfg['gain_sd']**2*(rho*np.outer(vgain,vgain)+(1-rho)*np.diag(vgain*vgain));C+=cg;parts['gain']=cg
 vb=np.sum(G*npix/N,axis=1);cb=2*(1-cfg['background_cross_channel_correlation'])*cfg['background_calibration_sd_e_per_pixel']**2*np.outer(vb,vb);C+=cb;parts['background_calibration']=cb
 cp=cfg['polarimetric_cross_talk_asymmetry_sd']**2*np.outer(vgain,vgain);C+=cp;parts['polarimetric_asymmetry']=cp
 dq=(1-2*c)*(der[:,0]-der[:,1])+g*der.sum(axis=1);vz=np.sum(G*dq,axis=1);shape=(t-np.mean([t[0],t[-1]]))/((t[-1]-t[0])/2);vd=np.sum(G*dq*shape,axis=1)
 cz=cfg['common_axial_offset_sd_mm']**2*np.outer(vz,vz)+np.diag(cfg['order_axial_offset_sd_mm']**2*vz*vz+cfg['smooth_scan_drift_sd_mm']**2*vd*vd);C+=cz;parts['axial_drift']=cz
 return C,means,{k:v.tolist() for k,v in parts.items()},condition

def run_scenario(response,cfg,seed,nrep,tag,ideal):
 rng=np.random.default_rng(seed);mu=response['mu'];n=len(response['tau']);N=cfg['Nincident'];npix=response['npixels'];c=cfg['polarimetric_cross_talk_mean'];rho=cfg['gain_cross_order_correlation'];tau=response['tau'];shape=(tau-(tau[0]+tau[-1])/2)/((tau[-1]-tau[0])/2)
 C,mean_events,parts,cond=covariance(response,cfg);halves=[];peaks=[];maxima=[];raw_example=None
 for start in range(0,nrep,100):
  nr=min(100,nrep-start);g=cfg.get('gain_mean',0)+cfg['gain_sd']*(np.sqrt(rho)*rng.normal(size=(nr,1))+np.sqrt(1-rho)*rng.normal(size=(nr,3)));asym=rng.normal(0,cfg['polarimetric_cross_talk_asymmetry_sd'],size=(nr,1,1));a=c+asym/2;b=c-asym/2
  if np.min([a.min(),b.min()])<0:raise ValueError('Negative polarimeter probability')
  offset=rng.normal(0,cfg['common_axial_offset_sd_mm'],size=(nr,1,1))+rng.normal(0,cfg['order_axial_offset_sd_mm'],size=(nr,3,1))+rng.normal(0,cfg['smooth_scan_drift_sd_mm'],size=(nr,3,1))*shape[None,None,:]
  ph=shifted_means(response,offset);plus=((1-a)*ph[:,:,0]+b*ph[:,:,1])*(1+g[:,:,None]);minus=(a*ph[:,:,0]+(1-b)*ph[:,:,1])*(1-g[:,:,None]);optical=np.stack([plus,minus],axis=2)
  expected=N*optical+cfg['background_e_per_pixel']*npix[None,:,None,:];counts=rng.poisson(expected)+rng.normal(size=expected.shape)*np.sqrt(npix)[None,:,None,:]*cfg['read_noise_e_per_pixel']
  brho=cfg['background_cross_channel_correlation'];bg=cfg['background_calibration_sd_e_per_pixel']*(np.sqrt(brho)*rng.normal(size=(nr,1))+np.sqrt(1-brho)*rng.normal(size=(nr,2)))
  corrected=counts-(cfg['background_e_per_pixel']+bg[:,None,:,None])*npix[None,:,None,:]
  monitor=rng.poisson(N*cfg['monitor_reference_count_multiplier'],size=(nr,n))/cfg['monitor_reference_count_multiplier'];signal=(corrected[:,:,0]-corrected[:,:,1])/monitor[:,None,:]
  if raw_example is None:raw_example={'raw_aggregate_Iplus_Iminus':counts[:3], 'expected_raw_aggregate':expected[:3], 'reference_monitor':monitor[:3], 'gain_errors':g[:3],'axial_offset_mm':offset[:3],'polarimeter_asymmetry':asym[:3],'background_calibration_error':bg[:3]}
  hh=[];pp=[];nn=[]
  for j in range(3):
   est=estimate(tau,signal[:,j]);hh.append(est['half_tau']*response['Lc']);pp.append(est['peak_tau']*response['Lc']);nn.append(est['n_local_maxima'])
  halves.append(np.array(hh).T);peaks.append(np.array(pp).T);maxima.append(np.array(nn).T)
 h=np.concatenate(halves);pk=np.concatenate(peaks);nm=np.concatenate(maxima);pixel_target=np.array([estimate(tau,mu[j,0]-mu[j,1])['half_tau'][0]*response['Lc'] for j in range(3)]);rec={'tag':tag,'config':cfg,'seed':seed,'replicates':nrep,'job_status':'RUN_COMPLETED','CI_kind':'oracle first-order normal covariance at expected detector signal, fixed smoothing; systematic unknown gain mean excluded from uncertainty','mean_event_linearization_mm_relative_zc':mean_events,'pixel_blur_target_mm_relative_zc':pixel_target.tolist(),'ideal_target_mm_relative_zc':ideal.tolist(),'event_covariance_mm2':C.tolist(),'covariance_parts_mm2':parts,'condition':cond,'ROI_optical_counts_range':[[float((N*mu[j].sum(axis=0)).min()),float((N*mu[j].sum(axis=0)).max())] for j in range(3)],'ROI_pixel_counts_range':[[int(npix[j].min()),int(npix[j].max())] for j in range(3)],'orders':{}}
 for j,m in enumerate(MS):
  a=h[:,j];valid=np.isfinite(a);sd=float(np.sqrt(C[j,j]));row={'single_event_failure_rate':float(1-valid.mean()),'single_event_bias_ideal_mm':float(np.nanmean(a)-ideal[j]),'single_event_SD_mm':float(np.nanstd(a,ddof=1)),'single_event_CI_halfwidth_mm':1.95996398454*sd,'single_event_coverage_ideal':float(np.mean(abs(a[valid]-ideal[j])<=1.95996398454*sd)),'mean_extra_peak_count':float(np.mean(nm[:,j]-1))}
  if j:
   delta=h[:,0]-a;ok=np.isfinite(delta);target=float(ideal[0]-ideal[j]);targetpixel=float(pixel_target[0]-pixel_target[j]);se=float(np.sqrt(max(C[0,0]+C[j,j]-2*C[0,j],0)));half=1.95996398454*se
   row.update(delta_ideal_mm=target,delta_pixel_blur_mm=targetpixel,delta_mean_mm=float(np.nanmean(delta)),delta_bias_ideal_mm=float(np.nanmean(delta)-target),delta_bias_pixel_blur_mm=float(np.nanmean(delta)-targetpixel),delta_SD_mm=float(np.nanstd(delta,ddof=1)),delta_CI_width_mm=2*half,delta_CI_coverage_ideal=float(np.mean(abs(delta[ok]-target)<=half)),delta_CI_coverage_pixel_blur=float(np.mean(abs(delta[ok]-targetpixel)<=half)),advance_sign_recovery_rate=float(np.mean(np.sign(delta[ok])==np.sign(target))),paired_failure_rate=float(1-ok.mean()),undefined_count=int((~ok).sum()),ignoring_cross_order_covariance_CI_width_mm=float(2*1.95996398454*np.sqrt(C[0,0]+C[j,j])))
  rec['orders'][str(m)]=row
 np.savez_compressed(DEST/f'{tag}_samples.npz',tau=tau,half_z_minus_zc_mm=h,peak_z_minus_zc_mm=pk,local_peak_counts=nm,**raw_example)
 (DEST/f'{tag}.json').write_text(json.dumps(rec,indent=2));return rec

def save_raw_pixel_example(response,cfg,tag):
 # An independently seeded full detector image, before ROI aggregation.
 rng=np.random.default_rng(61456410);j=2;geo=response['geometry'][j];native=response['native_tau'];it=int(np.argmin(abs(native)));means=[]
 for q in [3,5]:
  I=response['intensity'][response['orders'].index(q),it];lo=geo['sub_left'];f=geo['sub_fraction'];means.append(np.sum((I[lo]*(1-f)+I[lo+1]*f)*geo['quadrature_area_weights'],axis=1))
 means=np.array(means);c=cfg['polarimetric_cross_talk_mean'];N=cfg['Nincident'];lam=N*np.array([(1-c)*means[0]+c*means[1],c*means[0]+(1-c)*means[1]])+cfg['background_e_per_pixel'];poisson=rng.poisson(lam);read=rng.normal(0,cfg['read_noise_e_per_pixel'],size=lam.shape);near=int(np.argmin(abs(response['tau'])));mask=geo['center_radius']<=response['bounds'][j,near]
 np.savez_compressed(DEST/f'{tag}_raw_pixels.npz',x_mm=geo['x'],y_mm=geo['y'],expected_counts=lam,poisson_counts=poisson,read_noise=read,raw_counts=poisson+read,ROI_mask=mask,field_tau=native[it],aperture_tau=response['tau'][near],m=4,pitch_mm=response['pitch'],blur_sigma_mm=response['blur'])

if __name__=='__main__':
 pilot='--pilot' in sys.argv;groups=P['groups'][:1] if pilot else P['groups'];scens=scenarios()[:1] if pilot else scenarios();records=[]
 amendment={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'covariance_conventions':{'gain':'common fraction rho=.8 plus independent per-order constant; no axial dependence','background':'two correlated per-channel calibration offsets shared by all orders and planes','polarimetric_asymmetry':'one shared per-scan channel mixing asymmetry; a=c+delta/2,b=c-delta/2','z':'common offset shared by orders; independent order offsets; independent order linear drift with stated RMS amplitude at scan endpoints','monitor':'independent Poisson per plane, shared across orders','CI':'simulation-known mean covariance (oracle linearization), not a claimed deployable calibration estimate'},'frozen_before_count_draws':True,'pilot':pilot,'code_hashes':{f:hashlib.sha256((OUT/'code'/f).read_bytes()).hexdigest() for f in ['measurement_detector.py','measurement_estimator.py','run_measurement.py']}}
 (OUT/'config'/('MEASUREMENT_PILOT_IMPLEMENTATION.json' if pilot else 'MEASUREMENT_IMPLEMENTATION.json')).write_text(json.dumps(amendment,indent=2))
 for ig,group in enumerate(groups):
  R=group['R0'];eta=group['eta'];prefix=f'R{R}_eta{eta:g}'
  deadline=time.monotonic()+3600
  while not (OUT/'data/principal_refined'/f'{prefix}.json').exists():
   if time.monotonic()>deadline:raise TimeoutError(f'Refined fields unavailable for {prefix}')
   time.sleep(10)
  fields=dict(np.load(OUT/'data/principal_refined'/f'{prefix}_fields.npz'));native=json.loads((OUT/'data/principal_refined'/f'{prefix}.json').read_text());Lc=float(fields['kw'])*float(fields['w_mm'])/np.sqrt(R);ideal=np.array([native['events'][str(m)]['scaled_1']['thresholds']['0.5']['tau']*Lc for m in MS]);last_key=None;response=None
  # Group identical image geometry to reuse deterministic count means.
  ordered=sorted(enumerate(scens),key=lambda it:(it[1][1]['pixel_pitch_mm'],it[1][1]['blur_sigma_mm'],it[1][1]['dz_mm']))
  for si,(name,cfg) in ordered:
   tag=f'{prefix}_{name}'+('_pilot' if pilot else '');fn=DEST/f'{tag}.json'
   if fn.exists():records.append(json.loads(fn.read_text()));continue
   start=time.perf_counter();key=(cfg['pixel_pitch_mm'],cfg['blur_sigma_mm'],cfg['dz_mm']);print('START',tag,flush=True)
   if key!=last_key:
    if response is not None:del response;gc.collect()
    a,b=P['scan_tau'];dz=cfg['dz_mm'];tau=a+np.arange(int(np.floor((b-a)*Lc/dz))+1)*dz/Lc;response=detector_response(fields,MS,tau,key[0],key[1]);last_key=key
   rec=run_scenario(response,cfg,P['seed']+10000*ig+si,200 if pilot else P['replicates'],tag,ideal);rec['seconds']=time.perf_counter()-start;fn.write_text(json.dumps(rec,indent=2));records.append(rec)
   if name=='baseline':save_raw_pixel_example(response,cfg,tag)
   (DEST/('PILOT_STATUS.json' if pilot else 'RUN_STATUS.json')).write_text(json.dumps({'completed':len(records),'planned':len(groups)*len(scens),'jobs':[{'tag':r['tag'],'status':r['job_status']} for r in records]},indent=2));print('END',tag,'seconds',rec['seconds'],'m4',rec['orders']['4'],flush=True)
