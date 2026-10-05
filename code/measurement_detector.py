"""Pixel-integrated I+/I- detector. All radii in mm, intensities normalized by Pin.
Binary pixel-centre apertures permit exact aggregation of independent Poisson pixels.
No Gaussian or percentage noise is applied directly to Q.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import i0e,jnp_zeros
from scipy.linalg.blas import dgemm
from numpy.polynomial.legendre import leggauss

def pixel_geometry(radius,pitch,rgrid,quadrature=3):
 n=int(np.ceil((max(radius)+pitch)/pitch));g=np.arange(-n,n+1)*pitch
 xx,yy=np.meshgrid(g,g,indexing='xy');keep=xx*xx+yy*yy<=(max(radius)+pitch)**2
 xx=xx[keep];yy=yy[keep];order=np.argsort(xx*xx+yy*yy);xx=xx[order];yy=yy[order];center=np.hypot(xx,yy)
 a,w=leggauss(quadrature);dx,dy=np.meshgrid(a*pitch/2,a*pitch/2);ww=np.outer(w,w).ravel()/4*pitch*pitch
 sub=np.hypot(xx[:,None]+dx.ravel(),yy[:,None]+dy.ravel());dr=rgrid[1];lo=np.floor(sub/dr).astype(int);f=sub/dr-lo
 if lo.max()+1>=len(rgrid):raise ValueError('Detector interpolation support too short')
 ninc=np.searchsorted(center,radius,side='right');W=np.zeros((len(radius),len(rgrid)));acc=np.zeros(len(rgrid));prev=0
 for ni in np.unique(ninc):
  sl=slice(prev,ni);l=lo[sl].ravel();frac=f[sl].ravel();weight=np.broadcast_to(ww,lo[sl].shape).ravel()
  acc+=np.bincount(l,weights=weight*(1-frac),minlength=len(rgrid))+np.bincount(l+1,weights=weight*frac,minlength=len(rgrid))
  W[ninc==ni]=acc;prev=ni
 return W,ninc,{'x':xx,'y':yy,'center_radius':center,'sub_left':lo,'sub_fraction':f,'quadrature_area_weights':ww}

def detector_response(fields,ms,tau_scan,pitch,blur,quadrature=3):
 t=fields['tau'];R=float(fields['R0']);kw=float(fields['kw']);w=float(fields['w_mm']);k=kw/w;Lc=k*w*w/np.sqrt(R)
 bounds=np.array([float(jnp_zeros(m,1)[0])*2*w/(2*np.sqrt(R)+tau_scan/np.sqrt(R)) for m in ms])
 dr=min(pitch/8,blur/8 if blur else .0001,.0001);rmax=float(bounds.max()+2*pitch+8*blur);nr=int(np.ceil(rmax/dr))+1;nr+=1-nr%2;rgrid=np.linspace(0,rmax,nr)
 if rmax>np.min(fields['rho_mm'][:,-1]):raise ValueError('Stored observation field does not cover blur/aperture support')
 orders=sorted(set([m-1 for m in ms]+[m+1 for m in ms]));orig_orders=list(fields.get('q_orders',range(fields['U'].shape[1])))
 intensity=[]
 for q in orders:
  vals=np.array([CubicSpline(fields['rho_mm'][i],fields['U'][i,orig_orders.index(q)])(rgrid) for i in range(len(t))]);intensity.append(.5*abs(vals)**2/float(fields['Pin']))
 intensity=np.array(intensity)
 if blur:
  weights=np.ones(nr);weights[1:-1:2]=4;weights[2:-1:2]=2;weights*=rgrid[1]/3
  K=np.exp(-.5*((rgrid[:,None]-rgrid[None,:])/blur)**2)*i0e(rgrid[:,None]*rgrid[None,:]/blur**2)*(rgrid*weights/blur**2)[None,:]
  intensity=np.einsum('ij,qtj->qti',K,intensity,optimize=True)
 cs=[];mu=[];der=[];npixels=[];geo=[];matrices=[]
 for j,m in enumerate(ms):
  W,npx,geometry=pixel_geometry(bounds[j],pitch,rgrid,quadrature);arr=np.array([dgemm(1.,W,intensity[orders.index(q)].T) for q in [m-1,m+1]])
  splines=[CubicSpline(t,ar.T) for ar in arr];idx=np.clip(np.searchsorted(t,tau_scan)-1,0,len(t)-2);delta=tau_scan-t[idx];jj=np.arange(len(tau_scan))
  def evaluate(cs,derivative=False):
   c=cs.c[:,idx,jj]
   return (3*c[0]*delta+2*c[1])*delta+c[2] if derivative else ((c[0]*delta+c[1])*delta+c[2])*delta+c[3]
  mu.append([evaluate(s) for s in splines]);der.append([evaluate(s,True)/Lc for s in splines]);cs.append(splines);npixels.append(npx);geo.append(geometry);matrices.append(W)
 return {'splines':cs,'mu':np.array(mu),'dmu_dz':np.array(der),'npixels':np.array(npixels),'tau':tau_scan,'native_tau':t,'Lc':Lc,'rgrid':rgrid,'intensity':intensity,'orders':orders,'geometry':geo,'W':matrices,'bounds':bounds,'quadrature':quadrature,'pitch':pitch,'blur':blur}

def shifted_means(response,offset_mm):
 # offset shape [replicate, m, plane]; aperture stays at nominal z.
 ts=response['tau'][None,None,:]+offset_mm/response['Lc'];native=response['native_tau'];idx=np.clip(np.searchsorted(native,ts)-1,0,len(native)-2);dx=ts-native[idx];jj=np.arange(len(response['tau']))[None,:];out=[]
 if ts.min()<native[0] or ts.max()>native[-1]:raise ValueError('Axial drift outside cached field interval')
 for m,splines in enumerate(response['splines']):
  channels=[]
  for sp in splines:
   c=sp.c[:,idx[:,m],jj];v=((c[0]*dx[:,m]+c[1])*dx[:,m]+c[2])*dx[:,m]+c[3]
   if v.min()<-1e-12:raise ValueError('Negative interpolated count mean')
   channels.append(np.maximum(v,0))
  out.append(channels)
 return np.transpose(np.array(out),(2,0,1,3))
