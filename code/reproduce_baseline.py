"""Original author pipeline, all writes redirected to an isolated reproduction."""
from pathlib import Path
import sys,json,time,resource,hashlib,csv
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'revision_614564/baseline/recomputed'
sys.path.insert(0,str(ROOT/'scripts'))
import compute_formula_locked_sci_data as a
a.OUT=OUT;a.DATA=OUT/'data';a.CASES=a.DATA/'cases'
sys.argv=[sys.argv[0],'--skip-breakdown']
t0=time.perf_counter();a.main()
import compute_enriched_figure_data as e
e.main()
import plot_formula_locked_sci_figures as p
p.DATA=a.DATA;p.FIGURES=OUT/'figures';p.INTERNAL_FIGURES=OUT/'internal_validation'
p.save_all.__defaults__=(p.FIGURES,)
p.main()
comparisons=[]
for fn in sorted(a.CASES.glob('*.json')):
 new=json.loads(fn.read_text());old=json.loads((ROOT/'output/sci_formula_figures/data/cases'/fn.name).read_text())
 for model,by_m in new['events'].items():
  for m,row in by_m.items():
   for metric,v in row.items():
    comparisons.append({'R0':new['R0'],'eta':new['eta'],'m':m,'model':model,'metric':metric,'new':v,'archived':old['events'][model][m][metric],'difference':v-old['events'][model][m][metric]})
with (OUT/'baseline_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
summary={'beam_cases':36,'comparison_rows':len(comparisons),'seconds':time.perf_counter()-t0,'maxrss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_abs_tau50_difference':max(abs(r['difference']) for r in comparisons if r['metric']=='tau_50'),'max_abs_peak_difference':max(abs(r['difference']) for r in comparisons if r['metric']=='tau_peak'),'scope':'author_program_reproduction; independent_validation_pending'}
(OUT/'run_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
