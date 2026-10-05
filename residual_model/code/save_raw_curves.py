from pathlib import Path
import numpy as np,json
from compact_rs import signal_rs
R=Path(__file__).resolve().parents[1];rows=json.loads((R/'data/radius_study.json').read_text())['rows'];rads=np.array([r['R0'] for r in rows]);t=np.linspace(-3,5,161);ms=np.array([1,3])
qs=np.array([[[signal_rs(tau,r,m=int(m),nx=32,nt=64) for tau in t] for m in ms] for r in rads])
np.savez_compressed(R/'data/raw_Q_curves.npz',R0=rads,m=ms,tau=t,Q_normalized=qs,eta=.35,nx=32,nt=64)
