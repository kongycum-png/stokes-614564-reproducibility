from pathlib import Path
import numpy as np,json,time
from sparse_channels import SparseSolver
from revision_solver import Grid
from run_convergence_extended import extract
OUT=Path(__file__).resolve().parents[1];rows=[]
for R,eta in [(24,.7),(192,.2),(256,.9)]:
 start=time.perf_counter();base=dict(np.load(OUT/f'data/principal/R{R}_eta{eta:g}_fields.npz'));ref=np.load(OUT/f'data/convergence/source_R{R}_eta{eta:g}_cutoff15_fields.npz');orders=np.array([0,2,3,5,7]);sol=SparseSolver(R,eta,orders,Grid(ds=.025,cutoff=15,ny=601,ymax=16,phase_terms=4,source_start=float(base['source_smax'])));tail=sol.propagate(base['tau']);d=dict(tail,U=base['U'][:,orders]+tail['U'],Pin=float(base['Pin'])+float(tail['Pin']));e=extract(d);er=extract(ref);diff=[]
 for m in [1,4,6]:diff.append(e[str(m)]['thresholds']['0.5']['tau']-er[str(m)]['thresholds']['0.5']['tau'])
 row={'R0':R,'eta':eta,'tail_cutoff':15,'complex_field_relative_L2':float(np.linalg.norm(d['U']-ref['U'])/np.linalg.norm(ref['U'])),'single_event_errors':diff,'advance_errors':[diff[0]-diff[j] for j in [1,2]],'seconds':time.perf_counter()-start};rows.append(row);print(row,flush=True)
 (OUT/'baseline/tail_correction_check.json').write_text(json.dumps(rows,indent=2))
if max(abs(v) for r in rows for v in r['single_event_errors'])>1e-7:raise RuntimeError('Tail-addition crosscheck failed declared 1e-7 target')
