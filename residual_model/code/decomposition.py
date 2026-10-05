import json
from pathlib import Path
import numpy as np
from analytic_series import field,qcoeff,reference,grid
from scipy.special import jv,jnp_zeros
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1]
doc=json.loads((ROOT/'data/analytic_coefficients_N14.json').read_text())[-1];c=np.array(doc['delta']);rows=[]
for n in range(2,11):
 groups=[]
 for m in [1,3]:
  ev=doc['events'][str(m)];cp=np.array(ev['cp']);ch=np.array(ev['ch']);h=ev['h0'];p=ev['p0'];slope=ch[0,1]
  direct=(.5*cp[n,0]-ch[n,0])/slope
  def edge(t):
    F=field(t,.35,m,N=n,nx=48,nt=96);x,w,_,_=grid(m,48,96);b=jnp_zeros(m,1)[0];I0=2*m*jv(m,b)**2
    return 2*np.real(np.sum(x*w*(F[0,:,0]*F[n,:,0].conj()-F[0,:,1]*F[n,:,1].conj())))/I0
  newfield=(.5*edge(p)-edge(h))/slope
  groups.append(dict(new_field=newfield,lower_field_and_prefactor=direct-newfield,nonlinear_event=ev['dh'][n]-direct))
 row=dict(n=n,coefficient=c[n],**{key:groups[0][key]-groups[1][key] for key in groups[0]});rows.append(row)
 print(row)
controls=[]
for name,removed in [('full14',[]),('remove5',[5]),('remove6',[6]),('remove5and6',[5,6]),('through4',list(range(5,15))),('through6',list(range(7,15)))]:
 cc=c.copy();cc[removed]=0
 fun=lambda R:float(sum(cc[k]*R**(1-k/2) for k in range(2,len(cc))))
 o=minimize_scalar(fun,bounds=(12,96),method='bounded')
 controls.append(dict(name=name,minimum_R=o.x,minimum_value=o.fun,values={str(R):fun(R) for R in [24,40,96]}))
(ROOT/'data/coefficient_decomposition.json').write_text(json.dumps(dict(basis='compact-field epsilon series; algebraic grouping, not additive physical powers',rows=rows,controls=controls),indent=2))
