"""Declared Savitzky-Golay peak and preceding rising half-height estimator."""
import numpy as np
from scipy.signal import savgol_filter

def estimate(tau,raw,return_gradient=False):
 raw=np.atleast_2d(raw);step=float(tau[1]-tau[0]);win=max(5,int(round(.25/step)));win+=1-win%2
 y=savgol_filter(raw,win,3,axis=1,mode='interp');n=len(tau);rows=np.arange(len(y));eligible=(tau>=.2)&(tau<=1.8);maxima=np.zeros_like(y,dtype=bool);maxima[:,1:-1]=(y[:,1:-1]>y[:,:-2])&(y[:,1:-1]>y[:,2:]);maxima&=eligible
 index=np.argmax(np.where(maxima,y,-np.inf),axis=1);index=np.clip(index,1,n-2);a=y[rows,index-1];b=y[rows,index];c=y[rows,index+1];den=a-2*b+c
 with np.errstate(divide='ignore',invalid='ignore'):v=(a-c)/(2*den);peak=b-(a-c)**2/(8*den)
 valid=maxima.any(axis=1)&(den<0)&(abs(v)<1)&(peak>0);threshold=.5*peak
 rises=(y[:,:-1]<threshold[:,None])&(y[:,1:]>=threshold[:,None])&(np.arange(n-1)[None,:]<index[:,None]);cross=np.max(np.where(rises,np.arange(n-1)[None,:],-1),axis=1);valid&=cross>=0;cross=np.maximum(cross,0)
 slope=(y[rows,cross+1]-y[rows,cross])/step
 with np.errstate(divide='ignore',invalid='ignore'):f=(threshold-y[rows,cross])/(slope*step)
 h=tau[cross]+f*step;valid&=np.isfinite(h)&(slope>0)&(f>=0)&(f<=1);h[~valid]=np.nan
 result={'half_tau':h,'peak_tau':tau[index]+v*step,'n_local_maxima':maxima.sum(axis=1),'valid':valid,'window_samples':win}
 if return_gradient:
  if len(raw)!=1:raise ValueError('Linearized gradient is evaluated at the declared noise-free detector mean')
  H=savgol_filter(np.eye(n),win,3,axis=0,mode='interp');j=index[0];l=cross[0];vv=v[0];pp=.5*vv*(vv-1)*H[j-1]+(1-vv*vv)*H[j]+.5*vv*(vv+1)*H[j+1];qq=(1-f[0])*H[l]+f[0]*H[l+1];grad=(.5*pp-qq)/slope[0]
  result['gradient_tau_per_Q']=grad if valid[0] else np.full(n,np.nan)
 return result
