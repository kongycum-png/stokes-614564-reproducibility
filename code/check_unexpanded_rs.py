from pathlib import Path
import json,time
import numpy as np
from scipy.interpolate import CubicSpline
from unexpanded_rs import FullRS
OUT=Path(__file__).resolve().parents[1]
records=[]
for R,eta in [(24,.7),(192,.2)]:
 tau=np.array([-.5,0.,1.]);x=np.array([0.,1.,3.,6.]);a=None
 for label,ds,am in [('base',.025,1),('source_fine',.0125,1),('angular_double',.025,2)]:
  t=time.perf_counter();sol=FullRS(R,eta,ds=ds,cutoff=np.ceil(18+eta*np.sqrt(R)/2),angular_multiplier=am);d=sol.propagate(tau,x)
  np.savez_compressed(OUT/'data/convergence'/f'unexpanded_R{R}_eta{eta:g}_{label}.npz',**d)
  if a is None:a=d['U']
  rec={'R0':R,'eta':eta,'level':label,'relative_L2_to_base':float(np.linalg.norm(d['U']-a)/np.linalg.norm(a)),'seconds':time.perf_counter()-t,'max_angular_samples':d['max_angular_samples'],'source_n':d['source_n']};records.append(rec);print(rec,flush=True)
(OUT/'baseline/unexpanded_rs_quadrature_check.json').write_text(json.dumps(records,indent=2))
