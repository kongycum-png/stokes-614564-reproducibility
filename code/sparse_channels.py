"""Memory-bounded subset of the independently checked Eq.(7) channel solver."""
import numpy as np
from scipy.special import jv
from revision_solver import Solver,Grid
class SparseSolver(Solver):
 def __init__(self,R0,eta,orders,grid=Grid(),**kwargs):
  super().__init__(R0,eta,0,grid,**kwargs)
  self.orders=np.array(sorted(set(abs(int(q)) for q in orders)))
  self.kernel=np.stack([jv(q,self.a[:,None]*self.s[None,:])*(self.weights*self.f*self.s)[None,:] for q in self.orders]);self.qmax=len(self.orders)-1
 def propagate(self,tau,model='reduced_RS',source_multiplier=None):
  d=super().propagate(tau,model,source_multiplier)
  d['U']*=((-1j)**(self.orders-np.arange(len(self.orders))))[None,:,None]
  d['q_orders']=self.orders.copy();return d
