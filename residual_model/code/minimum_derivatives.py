"""Five-point derivative controls at two finite-difference steps; no fit."""
from pathlib import Path
import json
from study import advance

root=Path(__file__).resolve().parents[1]
rows=[]
for radius in [24,32,36,36.3713,37,40,96]:
 for step in [.02,.05]:
  v=[advance(radius+k*step)['scaled'] for k in [-2,-1,0,1,2]]
  rows.append(dict(R0=radius,step=step,values=v,
   first_derivative=(v[0]-8*v[1]+8*v[3]-v[4])/(12*step),
   second_derivative=(-v[4]+16*v[3]-30*v[2]+16*v[1]-v[0])/(12*step**2)))
(root/'data/minimum_derivatives.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
