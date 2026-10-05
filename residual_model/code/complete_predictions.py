from pathlib import Path
import json,numpy as np
from scipy.optimize import minimize_scalar
from study import advance
R=Path(__file__).resolve().parents[1];doc=json.loads((R/'data/radius_study.json').read_text());co=np.array(json.loads((R/'data/analytic_coefficients_N14.json').read_text())[-1]['delta'])
for row in doc['rows']:
 rad=row['R0'];row['series'].update({str(N):float(sum(co[k]*rad**(1-k/2) for k in range(2,N+1))) for N in [12,14]})
 row['fresnel_scaled']=advance(rad,model='fresnel')['scaled'];row['geometry_scaled_correction']=row['scaled']-row['fresnel_scaled'];row['N14_fresnel_scaled_error']=row['series']['14']-row['fresnel_scaled']
 print(rad,row['N14_fresnel_scaled_error'],flush=True)
 (R/'data/radius_study.json').write_text(json.dumps(doc,indent=2))
d=json.loads((R/'data/minima.json').read_text())
for N in [12,14]:
 o=minimize_scalar(lambda rad:sum(co[k]*rad**(1-k/2) for k in range(2,N+1)),bounds=(24,56),method='bounded');d[str(N)]={'R0':float(o.x),'scaled':float(o.fun)}
o=minimize_scalar(lambda rad:advance(rad,model='fresnel')['scaled'],bounds=(24,56),method='bounded',options={'xatol':2e-4});d['fresnel']={'R0':float(o.x),'scaled':float(o.fun)};(R/'data/minima.json').write_text(json.dumps(d,indent=2))
