import json,time
from pathlib import Path
from direct_check import direct
from study import event
ROOT=Path(__file__).resolve().parents[1];rows=[];start=time.time()
for R in [24,40,96]:
  for width,cutoff in [(.5,22.),(.25,26.)]:
    es=[event(lambda t:direct(t,R,.35,m,width=width,cutoff=cutoff,nx=40,model='reduced_RS'),scan=False) for m in [1,3]]
    row=dict(R0=R,width=width,cutoff=cutoff,events=es,advance=es[0]['h']-es[1]['h'],scaled=R*(es[0]['h']-es[1]['h']));rows.append(row);print(row,flush=True)
    (ROOT/'data/direct_events.json').write_text(json.dumps(dict(rows=rows,seconds=time.time()-start),indent=2))
