"""Print the full on-axis pressure profile p(xc,yc,z) for the homogeneous-bath
lens test, to see whether there's a genuine local max near the design focus
(z=44) in addition to / instead of the near-lens peak found at z=10."""
import numpy as np
from scipy.ndimage import gaussian_filter
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import tone_burst

dx=0.5e-3; freq=500e3
c_med,rho_med=1560.,1040.; c_lens,rho_lens=1000.,1030.
r_tx=8e-3; FOCAL=18e-3; P0=1e5; SM=0.6

def lens_t(r):
    t=(np.sqrt(r_tx**2+FOCAL**2)-np.sqrt(r**2+FOCAL**2))/(c_med*(1/c_lens-1/c_med))
    return np.where(r<=r_tx,np.maximum(t,0),0.)
def stamp(c,rho,zf,xc,yc):
    Nx,Ny,Nz=c.shape; xx,yy=np.meshgrid(np.arange(Nx),np.arange(Ny),indexing="ij")
    r=np.sqrt(((xx-xc)*dx)**2+((yy-yc)*dx)**2); tv=np.round(lens_t(r)/dx).astype(int)
    zz=np.arange(Nz)[None,None,:]; L=(zz>=zf)&(zz<zf+tv[:,:,None])&(r[:,:,None]<=r_tx)
    c[L],rho[L]=c_lens,rho_lens; return L, tv

Nx=Ny=52; Nz=80; xc=yc=Nx//2; zf=8          # Nz extended past the old 60 so we can see past the focus too
c=np.full((Nx,Ny,Nz),c_med,np.float32); rho=np.full((Nx,Ny,Nz),rho_med,np.float32)
L, tv = stamp(c,rho,zf,xc,yc)
print(f"lens thickness at center: {tv[xc,yc]} vox = {tv[xc,yc]*dx*1e3:.2f} mm -> lens spans z=[{zf},{zf+tv[xc,yc]})")
print(f"design focus at z = {zf}+{FOCAL/dx:.0f} = {zf+FOCAL/dx:.0f}")

# longer sim so the focus has plenty of time to fully form and we can see decay past it too
t_end = 2.0*(Nz*dx)/c_med
kg=kWaveGrid([Nx,Ny,Nz],[dx,dx,dx]); kg.makeTime(c_med,cfl=0.3,t_end=t_end)
cs=gaussian_filter(c,SM).astype(np.float32); rhos=gaussian_filter(rho,SM).astype(np.float32)
med=kWaveMedium(sound_speed=cs,density=rhos)
disc=(np.sqrt((np.arange(Nx)[:,None]-xc)**2+(np.arange(Ny)[None,:]-yc)**2)*dx)<=r_tx
s=kSource(); s.p_mask=np.zeros((Nx,Ny,Nz),np.uint8); s.p_mask[:,:,zf]=disc
s.p=(tone_burst(1/kg.dt,freq,5).flatten()*P0).reshape(1,-1)
sen=kSensor(); sen.mask=np.ones((Nx,Ny,Nz),np.uint8); sen.record=["p_max"]
print(f"running Nt={kg.Nt} ...")
sd=kspaceFirstOrder(kg,med,s,sen,device="cpu",dtype="float32",pml_size=10)
p=np.zeros((Nx,Ny,Nz),np.float32); p[sen.mask.astype(bool)]=np.squeeze(np.asarray(sd["p_max"]))
nbad=int((~np.isfinite(p)).sum()); p=np.nan_to_num(p)
print(f"nonfinite {nbad}")

prof = p[xc,yc,:]
print("\nz | depth_from_face_mm | p_max(kPa)")
for z in range(Nz):
    marker = ""
    if zf <= z < zf+tv[xc,yc]: marker = "  <- in lens"
    if z == zf: marker += "  <- source plane"
    if abs(z-(zf+round(FOCAL/dx))) <= 1: marker += "  <- DESIGN FOCUS"
    print(f"{z:3d} | {(z-zf)*dx*1e3:6.1f} | {prof[z]/1e3:8.1f}{marker}")

gi = np.argmax(prof)
print(f"\nglobal on-axis max at z={gi} ({(gi-zf)*dx*1e3:.1f}mm from face) = {prof[gi]/1e3:.1f} kPa")
post_lens = prof[zf+tv[xc,yc]:]
if len(post_lens):
    pi = np.argmax(post_lens) + zf+tv[xc,yc]
    print(f"max AFTER lens (z>{zf+tv[xc,yc]}) at z={pi} ({(pi-zf)*dx*1e3:.1f}mm from face) = {prof[pi]/1e3:.1f} kPa")
