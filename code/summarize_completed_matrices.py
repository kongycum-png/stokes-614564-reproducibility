from pathlib import Path
import json,csv,collections
OUT=Path(__file__).resolve().parents[1];DEST=OUT/'data/analysis_refined';rows=[];errs=[]
for fn in sorted((OUT/'data/model_comparison_refined').glob('R*_eta*.json')):
 d=json.loads(fn.read_text());R=d['R0'];eta=d['eta'];src=json.loads((OUT/'data/principal_refined'/f'R{R}_eta{eta:g}.json').read_text());errs.append(dict(R0=R,eta=eta,**d['unexpanded_RS_spotcheck']))
 for m in range(1,7):
  rec={'R0':R,'eta':eta,'m':m}
  for name,ev in [('reduced_RS',src['events'])]+[(name,v['events']) for name,v in d['models'].items()]:
   e=ev[str(m)]['scaled_1'];h=e['thresholds'].get('0.5',{}).get('tau');h1=ev['1']['scaled_1']['thresholds'].get('0.5',{}).get('tau');rec[name+'_event_status']=e['status'];rec[name+'_half_tau']=h;rec[name+'_advance']=h1-h if h is not None and h1 is not None else None
  for name in ['fresnel','reduced_RS']:
   for q in ['half_tau','advance']:
    a=rec['exact_kz_'+q];b=rec[name+'_'+q];rec[f'exact_minus_{name}_{q}']=a-b if a is not None and b is not None else None
  rows.append(rec)
with (DEST/'model_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(DEST/'model_spotchecks.json').write_text(json.dumps(errs,indent=2));print('model beams',len(rows),'RS max error',max(v['relative_complex_L2'] for v in errs))
for r in rows:
 if r['m']==4 and (r['R0'],r['eta']) in [(24,.7),(192,.2),(192,.7),(256,.9)]:print(r)
rows=[];status=collections.Counter()
for fn in sorted((OUT/'data/extended').glob('*.json')):
 if fn.name=='RUN_STATUS.json':continue
 d=json.loads(fn.read_text());status[d['job_status']]+=1
 if d['job_status']!='RUN_COMPLETED':continue
 for bid,b in d['beams'].items():
  P=b['P'];m=b['m'];ref=d['beams'].get(f'P{P}_m{1 if P==1 else P}')
  for key,e in b['events'].items():
   h=e['thresholds'].get('0.5',{}).get('tau');e1=ref['events'][key] if ref else None;h1=e1['thresholds'].get('0.5',{}).get('tau') if e1 else None
   rows.append({'tag':d['tag'],'kind':d['kind'],'R0':d['R0'],'eta':d['eta'],'kw':d['kw'],'P':P,'m':m,'boundary':key,'status':e['status'],'tau_half':h,'tau_peak':e['tau_peak'],'advance':h1-h if h is not None and h1 is not None else None,'root_jumps':e.get('root_jumps'),'missing_planes':e.get('missing_plane_count'),'phase_bound':d['phase_bound']})
with (DEST/'extended_events.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(DEST/'extended_summary.json').write_text(json.dumps({'job_status_counts':dict(status),'beam_cases':len(rows)//17,'boundary_rows':len(rows),'proxy_event_status_counts':dict(collections.Counter(r['status'] for r in rows if r['boundary']=='scaled_1')),'proxy_nonpositive_advance_count':sum(r['advance'] is not None and r['m']!=r['P'] and r['advance']<=0 for r in rows if r['boundary']=='scaled_1')},indent=2));print('extended',dict(status),'beams',len(rows)//17)
