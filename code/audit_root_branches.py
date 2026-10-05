"""Continuation identifiers and gate sensitivity, without changing first-root ROI.
Branch IDs are numerical continuation labels, not topological invariants.
"""
from pathlib import Path
import json,numpy as np,csv
from scipy.optimize import linear_sum_assignment
from observables import boundary_event
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/root_continuation';DEST.mkdir(exist_ok=True)
def track(t,allroots,gate):
 previous=[];prevprev={};nextid=0;first=[];allids=[];changes=[]
 for i,roots in enumerate(allroots):
  ids=[None]*len(roots)
  if previous and roots:
   pred=np.array([v+(v-prevprev[k][1])*(t[i]-t[i-1])/(t[i-1]-prevprev[k][0]) if k in prevprev else v for k,v in previous]);cost=abs(pred[:,None]-np.array(roots)[None,:]);rows,cols=linear_sum_assignment(cost)
   for a,b in zip(rows,cols):
    if cost[a,b]<=gate:ids[b]=previous[a][0]
  for j in range(len(ids)):
   if ids[j] is None:ids[j]=nextid;nextid+=1
  first.append(ids[0] if ids else None);allids.append(ids)
  if i and first[-1]!=first[-2]:changes.append(i-1)
  prevprev={k:(t[i-1],v) for k,v in previous} if i else {};previous=list(zip(ids,roots))
 return {'first_root_branch_id':first,'all_root_branch_ids':allids,'jump_indices':changes,'matching_gate_x':gate,'continuation':'minimum total distance to linear prediction, with gate; birth/death retained'}
rows=[]
for fn in sorted((OUT/'data/principal_refined').glob('R[0-9]*.json')):
 d=json.loads(fn.read_text());cv=np.load(fn.with_name(fn.stem+'_curves.npz'));t=cv['tau'];save={'R0':d['R0'],'eta':d['eta'],'m':{}}
 for ms,rr in d['root_tracking'].items():
  save['m'][ms]={}
  for gate in [.15,.3,.6]:
   tr=track(t,rr['all_candidates'],gate);ev=boundary_event(t,cv[f'm{ms}__actual_root__Q'],tr);old=d['events'][ms]['actual_root'];h=ev.get('thresholds',{}).get('0.5',{}).get('tau');ho=old.get('thresholds',{}).get('0.5',{}).get('tau');save['m'][ms][str(gate)]=dict(tr,event=ev);rows.append({'R0':d['R0'],'eta':d['eta'],'m':int(ms),'gate':gate,'status':ev['status'],'half':h,'old_status':old['status'],'half_change':None if h is None or ho is None else h-ho,'first_branch_changes':len(tr['jump_indices'])})
 (DEST/f'{fn.stem}.json').write_text(json.dumps(save,indent=2))
with (DEST/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary={'rows':len(rows),'status_changes':sum(r['status']!=r['old_status'] for r in rows),'defined_half_changes_above_1e-8':sum(r['half_change'] is not None and abs(r['half_change'])>1e-8 for r in rows),'maximum_defined_half_change':max(abs(r['half_change']) for r in rows if r['half_change'] is not None),'not_a_proof':'finite sampling and matching gates cannot certify branch topology through an unresolved merger'};(DEST/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
