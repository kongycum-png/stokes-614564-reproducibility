"""Re-run original Eq.7 at an existing m=1,2,3 representative point."""
from pathlib import Path
import sys,json,time,resource,hashlib,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'revision_614564'
sys.path.insert(0,str(ROOT/'scripts'))
import compute_formula_locked_sci_data as original
start=time.perf_counter()
old=json.loads((ROOT/'output/sci_formula_figures/data/cases/R0_24_eta_0.70.json').read_text())
evaluator=original.ExactCaseEvaluator(24.,.7,original.PRODUCTION)
print('Recomputing original Eq.7; R0=24 eta=.70; one process',flush=True)
z,rho,plane=evaluator.plane(0.)
np.savez_compressed(OUT/'baseline/pilot_complex_fields_tau0.npz',rho_mm=rho,x=evaluator.x,z_mm=z,**{f'U{q}':u for q,u in plane.u_channels.items()})
actual=original.direct_events(evaluator,{int(m):v for m,v in old['events']['RS_production'].items()})
comparison={str(m):{'actual':v,'archived':old['events']['RS_production'][str(m)],'difference':{k:v[k]-old['events']['RS_production'][str(m)][k] for k in v}} for m,v in actual.items()}
report={'status':'RUN_COMPLETED','scope':'REPRODUCTION_NOT_INDEPENDENT_VALIDATION','R0':24.,'eta':.7,'comparison':comparison,'seconds':time.perf_counter()-start,'maxrss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'field_evaluations':len(evaluator.cache),'source_diagnostics':original.source_diagnostics(evaluator.case,original.PRODUCTION),'python':platform.python_version(),'code_sha256':hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()}
(OUT/'baseline/pilot_result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
