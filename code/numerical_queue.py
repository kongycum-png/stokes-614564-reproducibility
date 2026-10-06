"""Run the configured numerical calculations sequentially."""
from pathlib import Path
import time,json,subprocess,sys,datetime
OUT=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+4*3600
status=OUT/'data/principal_refined/RUN_STATUS.json'
print('Waiting for current source-tail refinement, then configured extended matrix',flush=True)
while True:
 if status.exists():
  try:
   d=json.loads(status.read_text())
   if d['beam_cases_completed']==600:break
  except (json.JSONDecodeError,KeyError):pass
 if time.monotonic()>deadline:raise TimeoutError('Tail refinement did not finish within four hours; no next computation started')
 time.sleep(10)
print('Source-tail refinement complete; starting extended matrix',datetime.datetime.now().isoformat(),flush=True)
with (OUT/'logs/extended_matrix.log').open('w') as log:
 subprocess.run([sys.executable,str(OUT/'code/run_extended_matrix.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
print('Extended matrix process complete',datetime.datetime.now().isoformat(),flush=True)
