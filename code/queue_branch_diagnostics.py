from pathlib import Path
import time,json,subprocess,sys
out=Path(__file__).resolve().parents[1];deadline=time.monotonic()+4*3600
while True:
 fn=out/'data/convergence_final/RUN_STATUS.json'
 if fn.exists():
  try:
   if json.loads(fn.read_text())['completed']==18:break
  except (KeyError,json.JSONDecodeError):pass
 if time.monotonic()>deadline:raise TimeoutError('Final convergence queue did not complete')
 time.sleep(10)
with (out/'logs/hankel_branches.log').open('w') as f:subprocess.run([sys.executable,str(out/'code/hankel_branch_diagnostics.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
