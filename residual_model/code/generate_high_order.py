from pathlib import Path
import json,numpy as np
from analytic_series import event_coefficients
R=Path(__file__).resolve().parents[1];rows=[]
for radius in [.7,.9]:
 e={str(m):event_coefficients(m,N=14,nx=32,nt=64,points=72,radius=radius) for m in [1,3]}
 c=np.array(e['1']['dh'])-e['3']['dh'];rows.append(dict(radius=radius,events=e,delta=c.tolist()))
 (R/'data/analytic_coefficients_N14.json').write_text(json.dumps(rows,indent=2));print(radius,c,flush=True)
