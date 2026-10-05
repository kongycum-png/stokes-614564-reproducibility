from pathlib import Path
import sys,time,json,resource
import numpy as np
from scipy.special import jv
from revision_solver import Solver,Grid
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'revision_614564';sys.path.insert(0,str(ROOT/'scripts'))
import compute_formula_locked_sci_data as original
from oe_baseline import evaluate_channel_plane,_bessel_orders_0_to_n
start=time.perf_counter();sol=Solver(24,.7,qmax=4,grid=Grid(ds=.055,cutoff=9.,ny=301,ymax=8.5));d=sol.propagate(np.array([-.5,0.,1.]))
old=original.ExactCaseEvaluator(24,.7,original.PRODUCTION); rows=[]
for i,t in enumerate(d['tau']):
 p=evaluate_channel_plane(d['rho_mm'][i],d['z_mm'][i],q_orders=range(5),k_orders=(),envelope=old.envelope,k_rad_per_mm=old.case.k_rad_per_mm,options=old.options)
 carrier=np.exp(-1j*sol.k*d['z_mm'][i])
 for q in range(5):
  ref=p.u_channels[q]*carrier;u=d['U'][i,q];rows.append({'tau':float(t),'q':q,'relative_complex_L2':float(np.linalg.norm(u-ref)/np.linalg.norm(ref)),'relative_intensity_L2':float(np.linalg.norm(abs(u)**2-abs(ref)**2)/np.linalg.norm(abs(ref)**2))})
arg=np.geomspace(1e-6,40,2000); b=_bessel_orders_0_to_n(arg,17);rec=[{'q':q,'max_absolute_error':float(np.max(abs(b[q]-jv(q,arg))))} for q in [4,5,7,9,13,17]]
report={'comparison':rows,'original_forward_recurrence_stress':rec,'phase_M':d['phase_M'],'phase_series_absolute_field_error_bound':d['phase_remainder_absolute_bound'],'seconds':time.perf_counter()-start,'maxrss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(OUT/'baseline/independent_solver_pilot.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
