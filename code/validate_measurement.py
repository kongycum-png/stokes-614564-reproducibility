from pathlib import Path
import numpy as np,json
from scipy.interpolate import CubicSpline
from scipy.special import i0e
from scipy.linalg.blas import dgemm
from measurement_detector import detector_response,shifted_means,pixel_geometry
from measurement_estimator import estimate
OUT=Path(__file__).resolve().parents[1];fields=dict(np.load(OUT/'data/principal_refined/R24_eta0.35_fields.npz'));tau=np.linspace(-1.5,2.5,521);checks={};res=[]
for quadrature in [3,5,7]:
 d=detector_response(fields,[1,2,4],tau,.001,.001,quadrature);ev=np.array([estimate(tau,d['mu'][m,0]-d['mu'][m,1])['half_tau'][0] for m in range(3)]);res.append(ev)
 checks[f'pixel_q{quadrature}']={'half_tau':ev.tolist(),'pixel_area_sum_max_relative_error':float(max(np.max(abs(d['W'][m].sum(axis=1)-d['npixels'][m]*.001**2)/(d['npixels'][m]*.001**2)) for m in range(3)))}
checks['pixel_quadrature_max_event_changes_tau']=[float(np.max(abs(res[i]-res[i-1]))) for i in [1,2]]
y=d['mu'][2,0]-d['mu'][2,1];e=estimate(tau,y,True);rng=np.random.default_rng(44);v=rng.normal(size=len(y));eps=1e-9;fd=(estimate(tau,y+eps*v)['half_tau'][0]-estimate(tau,y-eps*v)['half_tau'][0])/(2*eps);anal=float(np.dot(e['gradient_tau_per_Q'],v));checks['event_gradient']={'finite_difference':float(fd),'analytic':anal,'relative_difference':float(abs(fd-anal)/max(abs(fd),1e-12))}
base=shifted_means(d,np.zeros((2,3,len(tau))));checks['zero_drift_mean_max_absolute_difference']=float(np.max(abs(base-d['mu'][None])))
h=1e-4;plus=shifted_means(d,np.full((1,3,len(tau)),h));minus=shifted_means(d,np.full((1,3,len(tau)),-h));checks['fixed_nominal_aperture_drift_derivative_relative_error']=float(np.linalg.norm((plus-minus)[0]/(2*h)-d['dmu_dz'])/np.linalg.norm(d['dmu_dz']))
# Independently known radial Gaussian convolution, with Simpson quadrature.
r=np.linspace(0,.08,1601);sigma=.002;a=.012;ww=np.ones(len(r));ww[1:-1:2]=4;ww[2:-1:2]=2;ww*=r[1]/3;K=np.exp(-.5*((r[:,None]-r[None,:])/sigma)**2)*i0e(r[:,None]*r[None,:]/sigma**2)*(r*ww/sigma**2)[None,:];I=np.exp(-r*r/a**2)/(np.pi*a*a);out=dgemm(1.,K,I[:,None])[:,0];width=a*a+2*sigma*sigma;expected=np.exp(-r*r/width)/(np.pi*width);checks['Gaussian_blur_relative_L2']=float(np.linalg.norm(out-expected)/np.linalg.norm(expected))
# Q_gain = Q + g T is an exact count-mean identity, not phase conversion.
pp=d['mu'][:,0];pm=d['mu'][:,1];gain=.01;checks['gain_leakage_identity_max_absolute']=float(np.max(abs(((1+gain)*pp-(1-gain)*pm)-((pp-pm)+gain*(pp+pm)))))
checks['status']='NUMERICAL_COMPONENT_CHECKS_COMPLETED';(OUT/'baseline/measurement_component_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
