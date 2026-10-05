from pathlib import Path
import numpy as np,json,time
from run_measurement import P,run_scenario,detector_response,OUT,DEST
rows=[]
for i,(R,eta,N) in enumerate([(24,.7,1e6),(96,.35,1e6),(96,.7,1e6),(96,.7,1e7)]):
 prefix=f'R{R}_eta{eta:g}';fields=dict(np.load(OUT/'data/principal_refined'/f'{prefix}_fields.npz'));native=json.loads((OUT/'data/principal_refined'/f'{prefix}.json').read_text());cfg=dict(P['baseline'],Nincident=N);Lc=float(fields['kw'])*float(fields['w_mm'])/np.sqrt(R);ideal=np.array([native['events'][str(m)]['scaled_1']['thresholds']['0.5']['tau']*Lc for m in [1,2,4]]);tau=-1.5+np.arange(int(np.floor(4*Lc/.1))+1)*.1/Lc;response=detector_response(fields,[1,2,4],tau,.001,.001);name='baseline' if N==1e6 else 'Nincident_1e+07';old=json.loads((DEST/f'{prefix}_{name}.json').read_text());rec=run_scenario(response,cfg,614564800+i,4000,f'{prefix}_N{N:g}_stability4000',ideal)
 for m in ['2','4']:
  p1=old['orders'][m]['advance_sign_recovery_rate'];p4=rec['orders'][m]['advance_sign_recovery_rate'];se=np.sqrt(p1*(1-p1)/1000+p4*(1-p4)/4000);rows.append({'R0':R,'eta':eta,'Nincident':N,'m':int(m),'sign_1000':p1,'sign_4000':p4,'difference':p4-p1,'binomial_combined_SE':float(se),'bias_1000_mm':old['orders'][m]['delta_bias_ideal_mm'],'bias_4000_mm':rec['orders'][m]['delta_bias_ideal_mm'],'SD_1000_mm':old['orders'][m]['delta_SD_mm'],'SD_4000_mm':rec['orders'][m]['delta_SD_mm']})
 print(prefix,N,rows[-2:],flush=True)
(OUT/'baseline/measurement_MC_stability.json').write_text(json.dumps(rows,indent=2))
