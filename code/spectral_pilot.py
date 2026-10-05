from pathlib import Path
import numpy as np,json,time
from spectral_solver import SpectralSolver
from revision_solver import Solver,Grid
OUT=Path(__file__).resolve().parents[1]
start=time.perf_counter();direct=Solver(24,.7,4,Grid(ds=.025,cutoff=15,ny=201,ymax=8.5,phase_terms=4));taus=np.array([-.25,0.,.9]);ref=direct.propagate(taus,model='fresnel');x=ref['x'][1];out=[]
# Compare at each field's true observation x, not equal index on differing grids.
from scipy.interpolate import CubicSpline
for n in [65536,131072,262144]:
 s=SpectralSolver(24,.7,4,nlog=n);d=s.propagate(taus,x,model='fresnel');rows=[]
 for i,t in enumerate(taus):
  for q in range(5):
   v=CubicSpline(ref['x'][i],ref['U'][i,q])(x);u=d['U'][i,q];rows.append({'tau':float(t),'q':q,'relative_complex_L2':float(np.linalg.norm(u-v)/np.linalg.norm(v)),'relative_intensity_L2':float(np.linalg.norm(abs(u)**2-abs(v)**2)/np.linalg.norm(abs(v)**2))})
 out.append({'nlog':n,'comparison':rows,'spectrum':s.diagnostics});print(n,max(r['relative_complex_L2'] for r in rows),flush=True)
 (OUT/'baseline/spectral_pilot.json').write_text(json.dumps(out,indent=2))
print('seconds',time.perf_counter()-start,flush=True)
