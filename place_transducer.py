"""
place_transducer.py  --  click where the STATIC transducer sits on the head.

Run from the `pesaranlab` env:
    python place_transducer.py

Three ortho views of the brain (skull outline in orange). Click in any view to move
the cursor to where the flat 20 mm transducer is mounted over the craniotomy. The tool
snaps to the nearest brain-surface point, draws the **transducer disc (blue) and beam
axis (yellow, into the brain)** at the chosen standoff, normal to the surface.

    click       move the transducer location
    [ / ]       decrease / increase standoff (gel thickness)  by 1 mm
    s           SAVE  -> transducer_location.json (+ 3D confirmation PNG)
    p           3D preview (disc + beam on the brain surface)
    r           reset cursor to brain centroid
    q           quit

Saved: transducer face-centre position, the surface mount point, the beam normal, the
standoff, and the equivalent 'left / back from vertex' angles -- ready to drop into the
simulation notebook.  Orientation is the local surface normal (transducer normal to the
brain, per the rig); tell Claude if you instead want free tilt control.
"""
import json
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
if __name__ == "__main__":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

BRAIN_NII = r"C:\Users\bengi\Documents\Pesaran Lab\brain_mask.nii"
SKULL_NII = r"C:\Users\bengi\Documents\Pesaran Lab\skull_mask.nii"
OUT_JSON  = "transducer_location.json"
OUT_PNG   = "transducer_location_3d.png"
SMOOTH_MM = 3.0
TX_DIAM   = 20e-3
STANDOFF0 = 8e-3        # initial standoff (gel thickness), metres

# ----------------------------------------------------------------------------- load
img   = nib.load(BRAIN_NII)
brain = ndimage.binary_fill_holes(img.get_fdata() > 0.5)
brain = np.rot90(brain, k=1, axes=(0,1))   # fix: was hitting front of brain instead of top
brain = np.flip(brain, axis=1)             # fix: was upside-down after the rotation above
A     = img.affine
dx_mm = float(img.header.get_zooms()[0]);  dx_m = dx_mm*1e-3
NX, NY, NZ = brain.shape
centroid = np.argwhere(brain).mean(0)
AX = nib.aff2axcodes(A)
try:
    skull = nib.load(SKULL_NII).get_fdata() > 0.5
    skull = np.rot90(skull, k=1, axes=(0,1))
    skull = np.flip(skull, axis=1)
    if skull.shape != brain.shape: skull = None
except Exception:
    skull = None

surf_mask = brain & ~ndimage.binary_erosion(brain)
surf_vox  = np.argwhere(surf_mask)
surf_tree = cKDTree(surf_vox)
gb = ndimage.gaussian_filter(brain.astype(np.float32), SMOOTH_MM/dx_mm)
G  = np.gradient(gb)

S = np.array([0., -1.,  0.]); L = np.array([0., 0., -1.]); P = np.array([1., 0., 0.])
R_TX_V = (TX_DIAM/2)/dx_m                       # transducer radius in voxels

# ----------------------------------------------------------------------------- geometry
def world(vox):  return (A @ np.array([*vox, 1.0]))[:3]

def anat_offset(vox):
    d = (np.array(vox, float) - centroid)*dx_mm
    return (f"{abs(d@L):.0f}mm {'Left' if d@L>0 else 'Right'}, "
            f"{abs(d@P):.0f}mm {'Post' if d@P>0 else 'Ant'}, "
            f"{abs(d@S):.0f}mm {'Sup' if d@S>0 else 'Inf'} (from centroid)")

def snap_surface(vox):
    return surf_vox[surf_tree.query(np.array(vox, float))[1]]

def outward_normal(vox):
    i, j, k = np.clip(np.round(vox).astype(int), 0, [NX-1, NY-1, NZ-1])
    n = -np.array([G[0][i,j,k], G[1][i,j,k], G[2][i,j,k]])
    if np.linalg.norm(n) < 1e-9:
        n = np.array(vox, float) - centroid
    return n/np.linalg.norm(n)

def normal_to_angles(n):
    aL = np.degrees(np.arcsin(np.clip(n@L, -1, 1)))
    aP = np.degrees(np.arcsin(np.clip((n@P)/max(np.cos(np.radians(aL)), 1e-6), -1, 1)))
    return aL, aP

def transducer_pose(cursor_vox, standoff_m):
    """surface mount point, outward normal, and transducer face-centre (voxels)."""
    sv   = snap_surface(cursor_vox)
    n    = outward_normal(sv)
    face = sv + n*(standoff_m/dx_m)              # face sits 'standoff' above the surface
    return sv, n, face

# project a 3D voxel point onto a panel's 2D axes
PANEL = {"AX": (2, 0, np.array([0.,1,0])),       # (x=k, y=i, slice-normal=j)
         "COR":(2, 1, np.array([1.,0,0])),       # (x=k, y=j, slice-normal=i)
         "SAG":(0, 1, np.array([0.,0,1]))}        # (x=i, y=j, slice-normal=k)
def to_panel(pt, panel):
    hx, vy, _ = PANEL[panel]
    return pt[hx], pt[vy]

# ----------------------------------------------------------------------------- save + 3D
def save_pose(cursor_vox, standoff_m):
    sv, n, face = transducer_pose(cursor_vox, standoff_m)
    aL, aP = normal_to_angles(n)
    nw = A[:3,:3] @ n; nw /= np.linalg.norm(nw)
    out = {
        "transducer_face_center_voxel": [round(float(v),2) for v in face],
        "transducer_face_center_world_RAS": [round(float(v),2) for v in world(face)],
        "surface_mount_voxel": [int(v) for v in sv],
        "surface_mount_world_RAS": [round(float(v),2) for v in world(sv)],
        "beam_normal_voxel": [round(float(v),4) for v in n],
        "beam_normal_world_RAS": [round(float(v),4) for v in nw],
        "standoff_mm": round(standoff_m*1e3,1), "tx_diam_mm": round(TX_DIAM*1e3,1),
        "equiv_angles_deg": {"left_from_vertex": round(float(aL),1),
                             "back_from_vertex": round(float(aP),1)},
        "anat_offset": anat_offset(sv), "voxel_size_mm": dx_mm, "axcodes": list(AX),
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print("\n" + "="*64)
    print(f"SAVED -> {OUT_JSON}")
    print(f"  transducer face : voxel {out['transducer_face_center_voxel']}  world {out['transducer_face_center_world_RAS']} mm")
    print(f"  surface mount   : voxel {out['surface_mount_voxel']}  world {out['surface_mount_world_RAS']} mm")
    print(f"  beam normal     : {out['beam_normal_voxel']} (voxel)   standoff {out['standoff_mm']} mm")
    print(f"  ~ {aL:+.0f} deg left, {aP:+.0f} deg back from vertex   |  {out['anat_offset']}")
    print("="*64)
    render_3d(sv, n, standoff_m, save=True)
    return out

def render_3d(sv, n, standoff_m, save=False):
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    figp = plt.figure("transducer 3D", figsize=(7,6)); figp.clf()
    ax = figp.add_subplot(111, projection="3d")
    verts, faces, _, _ = measure.marching_cubes(brain.astype(np.float32), 0.5, step_size=3)
    mesh = Poly3DCollection(verts[faces], alpha=0.12, linewidth=0); mesh.set_facecolor("#3cb371")
    ax.add_collection3d(mesh)
    face = sv + n*(standoff_m/dx_m)
    ax.scatter(*sv, color="red", s=50)
    ax.plot(*np.column_stack([face, sv - n*(40e-3/dx_m)]), "y-", lw=2)   # beam into brain
    e1 = np.cross(n, [0,0,1.]);  e1 /= (np.linalg.norm(e1)+1e-9); e2 = np.cross(n, e1)
    th = np.linspace(0, 2*np.pi, 40)
    circ = face[:,None] + R_TX_V*(np.cos(th)*e1[:,None] + np.sin(th)*e2[:,None])
    ax.plot(circ[0], circ[1], circ[2], "b-", lw=2.5)
    ax.set_title("transducer (blue disc) + beam (yellow)"); ax.set_box_aspect((1,1,1))
    if save:
        figp.savefig(OUT_PNG, dpi=130); print(f"  3D confirmation -> {OUT_PNG}")
    plt.draw()

# ----------------------------------------------------------------------------- UI
if __name__ == "__main__":
    state = {"P": np.round(centroid).astype(int), "standoff": STANDOFF0}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axA, axC = axes[0]; axS, axI = axes[1]; axI.axis("off")
    info = axI.text(0.02, 0.98, "", va="top", ha="left", fontsize=11, family="monospace")
    PANELAX = {"AX": axA, "COR": axC, "SAG": axS}

    def draw_overlay(ax, panel, sv, n, face):
        beam_end = sv - n*(40e-3/dx_m)
        sx, sy = to_panel(sv, panel); fx, fy = to_panel(face, panel); bx, by = to_panel(beam_end, panel)
        ax.plot([fx, bx], [fy, by], "y--", lw=1.2)                     # beam into brain
        ax.plot([sx, fx], [sy, fy], color="magenta", lw=1.2)          # standoff
        u = np.cross(n, PANEL[panel][2]); nu = np.linalg.norm(u)       # disc face in this slice
        if nu > 1e-6:
            u = u/nu*R_TX_V; p1 = face+u; p2 = face-u
            ax.plot(*zip(to_panel(p1,panel), to_panel(p2,panel)), "b-", lw=3)
        ax.scatter([sx],[sy], c="cyan", s=25, zorder=5)               # mount point
        ax.scatter([fx],[fy], c="magenta", s=25, zorder=5)            # face centre

    def redraw():
        pi, pj, pk = state["P"]
        sv, n, face = transducer_pose(state["P"], state["standoff"])
        slices = {"AX": (brain[:, pj, :], None if skull is None else skull[:, pj, :], pk, pi),
                  "COR":(brain[pi, :, :], None if skull is None else skull[pi, :, :], pk, pj),
                  "SAG":(brain[:, :, pk].T, None if skull is None else skull[:, :, pk].T, pi, pj)}
        titles = {"AX":f"AXIAL  j={pj}   x:Left/Right  y:Ant/Post",
                  "COR":f"CORONAL  i={pi}   x:Left/Right  y:Sup/Inf",
                  "SAG":f"SAGITTAL  k={pk}   x:Ant/Post  y:Sup/Inf"}
        for panel, ax in PANELAX.items():
            img2, sk2, cvx, cvy = slices[panel]
            ax.cla(); ax.imshow(img2, origin="lower", cmap="bone", aspect="equal")
            if sk2 is not None: ax.contour(sk2, levels=[0.5], colors="orange", linewidths=0.6)
            ax.axvline(cvx, color="0.4", lw=0.5); ax.axhline(cvy, color="0.4", lw=0.5)
            draw_overlay(ax, panel, sv, n, face)
            ax.invert_yaxis(); ax.set_title(titles[panel])
        wf = world(face); aL, aP = normal_to_angles(n)
        info.set_text(f"TRANSDUCER (blue disc) | beam = yellow\n\n"
                      f"face center voxel = ({face[0]:.0f},{face[1]:.0f},{face[2]:.0f})\n"
                      f"face world RAS    = ({wf[0]:+.1f}, {wf[1]:+.1f}, {wf[2]:+.1f}) mm\n"
                      f"standoff (gel)    = {state['standoff']*1e3:.0f} mm   tx = {TX_DIAM*1e3:.0f} mm\n"
                      f"beam ~ {aL:+.0f} deg left, {aP:+.0f} deg back from vertex\n"
                      f"mount: {anat_offset(snap_surface(state['P']))}\n\n"
                      f"[click] move   [ / ] standoff   [s] SAVE   [p] 3D   [r] reset   [q] quit")
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes not in PANELAX.values() or ev.xdata is None: return
        x, y = int(round(ev.xdata)), int(round(ev.ydata))
        if ev.inaxes is axA:   state["P"][0], state["P"][2] = np.clip(y,0,NX-1), np.clip(x,0,NZ-1)
        elif ev.inaxes is axC: state["P"][1], state["P"][2] = np.clip(y,0,NY-1), np.clip(x,0,NZ-1)
        elif ev.inaxes is axS: state["P"][0], state["P"][1] = np.clip(x,0,NX-1), np.clip(y,0,NY-1)
        redraw()

    def on_key(ev):
        if ev.key == "s":   save_pose(state["P"], state["standoff"])
        elif ev.key == "p":
            sv, n, _ = transducer_pose(state["P"], state["standoff"]); render_3d(sv, n, state["standoff"])
        elif ev.key == "]": state["standoff"] = min(state["standoff"]+1e-3, 30e-3); redraw()
        elif ev.key == "[": state["standoff"] = max(state["standoff"]-1e-3, 0.0);   redraw()
        elif ev.key == "r": state["P"][:] = np.round(centroid).astype(int); redraw()
        elif ev.key == "q": plt.close("all")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print(__doc__)
    print(f"brain {brain.shape} @ {dx_mm} mm | axcodes {AX} | centroid {np.round(centroid).astype(int)}")
    redraw(); plt.tight_layout(); plt.show()
