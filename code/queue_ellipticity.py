from pathlib import Path
import json,time,subprocess,sys
out=Path(__file__).resolve().parents[1];deadline=time.monotonic()+4*3600
while True:
 fn=out/'data/input_mismatch/RUN_STATUS.json'
 if fn.exists():
  try:
   if json.loads(fn.read_text())['parameter_groups_completed']==4:break
  except (KeyError,json.JSONDecodeError):pass
 if time.monotonic()>deadline:raise TimeoutError('Input mismatch queue incomplete')
 time.sleep(10)
with (out/'logs/elliptic_input_modes.log').open('w') as f:subprocess.run([sys.executable,str(out/'code/elliptic_input_modes.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
