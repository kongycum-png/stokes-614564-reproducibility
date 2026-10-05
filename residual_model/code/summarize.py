from pathlib import Path
import json,csv,math,hashlib
import numpy as np
from decimal import Decimal, getcontext
getcontext().prec=60
R=Path(__file__).resolve().parents[1]
a=json.loads((R/'data/analytic_coefficients_N14.json').read_text());c=np.array(a[-1]['delta']);rows=json.loads((R/'data/radius_study.json').read_text())['rows']
for name,rs in [('radius_predictions', [{k:v for k,v in x.items() if k not in ['reference','target','series']}|{f'series_{k}':v for k,v in x['series'].items()} for x in rows]),('event_continuation',[dict(R0=x['R0'],m=m,**x[who]) for x in rows for m,who in [(1,'reference'),(3,'target')]])]:
 with (R/'data'/f'{name}.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
mpd=json.loads((R/'data/high_precision.json').read_text())['rows'];mperr=max(abs(float(list(x['mp_values'].values())[-1])-x['numpy_value']) for x in mpd)
mpref=str(max(abs(Decimal(list(x['mp_values'].values())[-1])-Decimal(list(x['mp_values'].values())[0])) for x in mpd))
neg=[]
for rad in [16,24,40,96]:
 al=.35/(2*np.sqrt(rad));logb=al*rad-2*rad**1.5/3-math.log(2*math.sqrt(math.pi)*rad**.25)-2*math.log(math.sqrt(rad)-al);neg.append(dict(R0=rad,source_integral_absolute_bound=math.exp(logb)))
held=[x for x in rows if x['R0'] in [20,28,36,44,48,64,80]]
summary=dict(coefficients=c.tolist(),coefficient_refinement_max=float(np.max(abs(np.array(a[0]['delta'])-c))),high_precision_numpy_signal_max_absolute=mperr,high_precision_refinement_absolute=mpref,negative_source_bounds=neg,heldout_max_scaled_error_14_fresnel=max(abs(x['N14_fresnel_scaled_error']) for x in held),sample24to96_max_scaled_error_14_fresnel=max(abs(x['N14_fresnel_scaled_error']) for x in rows if 24<=x['R0']<=96),geometry_scaled_max=max(x['geometry_scaled_correction'] for x in rows if 24<=x['R0']<=96),minimum_event_slope=min(x[y]['slope'] for x in rows for y in ['reference','target']),minimum_abs_peak_curvature=min(abs(x[y]['curvature']) for x in rows for y in ['reference','target']),minima=json.loads((R/'data/minima.json').read_text()))
(R/'data/summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
