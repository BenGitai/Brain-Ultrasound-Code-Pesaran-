"""Small CPU checks of the notebook's physics:
 (A) does the silicone lens focus on-axis at ~FOCAL in a homogeneous brain-water bath?
 (B) does the step-4-style medium (air outside + coupling + brain slab) stay FINITE?"""
import numpy as np
from scipy.ndimage import gaussian_filter
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import tone_burst

dx=0.5e-3; freq=500e3
c_med,rho_med=1560.,1040.; c_lens,rho_lens=1000.,1030.; c_air,rho_air=343.,100.
r_tx=8e-3; FOCAL=18e-3; P0=1e5; SM=0.6

def lens_t(r):
    t=(np.sqrt(r_tx**2+FOCAL**2)-np.sqrt(r**2+FOCAL**2))/(c_med*(1/c_lens-1/c_med))
    return np.where(r<=r_tx,np.maximum(t,0),0.)
def stamp(c,rho,zf,xc,yc):
    Nx,Ny,Nz=c.shape; xx,yy=np.meshgrid(np.arange(Nx),np.arange(Ny),indexing="ij")
    r=np.sqrt(((xx-xc)*dx)**2+((yy-yc)*dx)**2); tv=np.round(lens_t(r)/dx).astype(int)
    zz=np.arange(Nz)[None,None,:]; L=(zz>=zf)&(zz<zf+tv[:,:,None])&(r[:,:,None]<=r_tx)
    c[L],rho[L]=c_lens,rho_lens; return L
def run(c,rho,zf,xc,yc,label):
    Nx,Ny,Nz=c.shape; kg=kWaveGrid([Nx,Ny,Nz],[dx,dx,dx]); kg.makeTime(c_med,cfl=0.3,t_end=1.2*(Nz*dx)/c_med)
    c=gaussian_filter(c,SM).astype(np.float32); rho=gaussian_filter(rho,SM).astype(np.float32)
    med=kWaveMedium(sound_speed=c,density=rho)
    disc=(np.sqrt((np.arange(Nx)[:,None]-xc)**2+(np.arange(Ny)[None,:]-yc)**2)*dx)<=r_tx
    s=kSource(); s.p_mask=np.zeros((Nx,Ny,Nz),np.uint8); s.p_mask[:,:,zf]=disc
    s.p=(tone_burst(1/kg.dt,freq,5).flatten()*P0).reshape(1,-1)
    sen=kSensor(); sen.mask=np.ones((Nx,Ny,Nz),np.uint8); sen.record=["p_max"]
    sd=kspaceFirstOrder(kg,med,s,sen,device="cpu",dtype="float32",pml_size=10)
    p=np.zeros((Nx,Ny,Nz),np.float32); p[sen.mask.astype(bool)]=np.squeeze(np.asarray(sd["p_max"]))
    nbad=int((~np.isfinite(p)).sum()); p=np.nan_to_num(p)
    return p,nbad,kg.Nt

# (A) homogeneous bath
Nx=Ny=52; Nz=60; xc=yc=Nx//2; zf=8
c=np.full((Nx,Ny,Nz),c_med,np.float32); rho=np.full((Nx,Ny,Nz),rho_med,np.float32); stamp(c,rho,zf,xc,yc)
p,nbad,Nt=run(c.copy(),rho.copy(),zf,xc,yc,"A")
fi=np.unravel_index(np.argmax(p),p.shape)
print(f"(A) homog lens: Nt {Nt} nonfinite {nbad} | focus {fi} depth {(fi[2]-zf)*dx*1e3:.1f}mm (design {FOCAL*1e3:.0f}) "
      f"off-axis {np.hypot(fi[0]-xc,fi[1]-yc)*dx*1e3:.1f}mm peak {p.max()/1e3:.1f}kPa "
      f"{'ON-AXIS OK' if np.hypot(fi[0]-xc,fi[1]-yc)<=2 else 'OFF-AXIS!'}")

# (B) air outside + coupling cylinder + brain slab
Nx=Ny=52; Nz=64; xc=yc=Nx//2; zf=8; zbrain=30
zz=np.broadcast_to(np.arange(Nz),(Nx,Ny,Nz))
xx,yy=np.meshgrid(np.arange(Nx),np.arange(Ny),indexing="ij"); r2=np.sqrt(((xx-xc)*dx)**2+((yy-yc)*dx)**2)
brain=(zz>=zbrain)&(r2<=18e-3)[:,:,None]; couple=(r2<=12e-3)[:,:,None]&(zz>=zf)&(zz<zbrain)&~brain
c=np.full((Nx,Ny,Nz),c_air,np.float32); rho=np.full((Nx,Ny,Nz),rho_air,np.float32)
c[couple]=c_med;rho[couple]=rho_med; c[brain]=c_med;rho[brain]=rho_med; stamp(c,rho,zf,xc,yc)
p,nbad,Nt=run(c.copy(),rho.copy(),zf,xc,yc,"B")
pb=np.where(brain,p,0); fi=np.unravel_index(np.argmax(pb),pb.shape)
print(f"(B) air+brain : Nt {Nt} nonfinite {nbad} | focus {fi} depth-in-brain {(fi[2]-zbrain)*dx*1e3:.1f}mm "
      f"peak {pb.max()/1e3:.1f}kPa {'FINITE/OK' if nbad==0 else 'NaN/UNSTABLE'}")
