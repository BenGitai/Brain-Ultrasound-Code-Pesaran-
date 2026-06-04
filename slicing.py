"""
interactive_skull_mask.py
-------------------------
Loads a NIfTI CT, extracts a 2D slice, and opens an interactive Matplotlib 
GUI. Allows you to manually dial in the HU threshold and morphological 
parameters in real-time. 

Click "Save Maps to NPY" when you are happy with the mask.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_closing, 
    binary_opening, 
    gaussian_filter,
    binary_fill_holes,
    label
)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

CT_PATH = r"C:\Users\bengi\Documents\Pesaran Lab\EeveeCT_260115_Registered.nii"
SLICE_PLANE = 'coronal'  # 'axial', 'coronal', or 'sagittal'

# Default starting values for the sliders
INIT_HU = 300
INIT_CLOSING = 3
INIT_OPENING = 2
INIT_SMOOTH = 1.5

# ── Constant acoustic properties ─────────────────────────────────────────────
C_WATER, RHO_WATER, ALPHA_WATER = 1482.0, 1000.0, 3.48e-4
C_BONE, RHO_BONE, ALPHA_BONE    = 3100.0, 2200.0, 21.5

# ─────────────────────────────────────────────────────────────────────────────

def load_ct(path: str):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    return data, spacing

def get_slice(volume: np.ndarray, spacing: np.ndarray, plane: str):
    plane = plane.lower()
    if plane == 'axial':
        sl = volume[:, :, volume.shape[2] // 2].T
        return sl, (spacing[0], spacing[1])
    elif plane == 'coronal':
        sl = volume[:, volume.shape[1] // 2, :].T
        return sl, (spacing[0], spacing[2])
    elif plane == 'sagittal':
        sl = volume[volume.shape[0] // 2, :, :].T
        return sl, (spacing[1], spacing[2])
    else:
        raise ValueError("Plane must be 'axial', 'coronal', or 'sagittal'")

def _disk_kernel(radius: int) -> np.ndarray:
    if radius == 0: return np.ones((1,1))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x**2 + y**2) <= radius**2

def make_bone_mask(sl, hu, close_r, open_r, smooth):
    """Generates the mask using the parameters from the sliders."""
    bone_raw = sl >= hu
    
    # Keep largest connected component (removes scanner bed/noise)
    labeled_array, num_features = label(bone_raw)
    if num_features > 0:
        largest_cc = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
        bone_main = labeled_array == largest_cc
    else:
        bone_main = bone_raw

    bone_filled = binary_fill_holes(bone_main)

    if open_r > 0:
        bone_clean = binary_opening(bone_filled, structure=_disk_kernel(int(open_r)))
    else:
        bone_clean = bone_filled

    if close_r > 0:
        bone_solid = binary_closing(bone_clean, structure=_disk_kernel(int(close_r)))
    else:
        bone_solid = bone_clean

    if smooth > 0:
        bone_float = gaussian_filter(bone_solid.astype(np.float32), sigma=smooth)
        return (bone_float >= 0.5).astype(np.float32)
    return bone_solid.astype(np.float32)

def main():
    print(f"Loading {CT_PATH}...")
    volume, spacing_3d = load_ct(CT_PATH)
    target_slice, spacing_2d = get_slice(volume, spacing_3d, SLICE_PLANE)

    # Setup Figure and Axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.4) # Leave room for sliders at the bottom
    
    # Calculate physical extents for proper aspect ratio
    px, py = spacing_2d[0], spacing_2d[1]
    ext = [0, target_slice.shape[1] * px, target_slice.shape[0] * py, 0]

    # Draw initial images
    ax.imshow(target_slice, cmap="gray", vmin=-200, vmax=1500, extent=ext)
    initial_mask = make_bone_mask(target_slice, INIT_HU, INIT_CLOSING, INIT_OPENING, INIT_SMOOTH)
    
    # Overlay the mask in red with some transparency
    mask_display = ax.imshow(initial_mask, cmap="Reds", alpha=initial_mask*0.5, extent=ext)
    ax.set_title(f"Interactive Bone Mask ({SLICE_PLANE.capitalize()} Slice)")
    ax.set_xlabel("Width (mm)")
    ax.set_ylabel("Height (mm)")

    # ─── SLIDERS ─────────────────────────────────────────────────────────────
    axcolor = 'lightgoldenrodyellow'
    
    ax_hu = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_close = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_open = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_smooth = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)

    s_hu = Slider(ax_hu, 'HU Threshold', 0.0, 1500.0, valinit=INIT_HU, valstep=10)
    s_close = Slider(ax_close, 'Fill Gaps (Close)', 0, 10, valinit=INIT_CLOSING, valstep=1)
    s_open = Slider(ax_open, 'Remove Noise (Open)', 0, 10, valinit=INIT_OPENING, valstep=1)
    s_smooth = Slider(ax_smooth, 'Smooth Edges', 0.0, 5.0, valinit=INIT_SMOOTH, valstep=0.5)

    def update(val):
        """Called every time a slider is moved."""
        new_mask = make_bone_mask(target_slice, s_hu.val, s_close.val, s_open.val, s_smooth.val)
        # Update the alpha channel so only the 1s are red, 0s are invisible
        mask_display.set_data(new_mask)
        mask_display.set_alpha(new_mask * 0.4) 
        fig.canvas.draw_idle()

    # Link sliders to the update function
    s_hu.on_changed(update)
    s_close.on_changed(update)
    s_open.on_changed(update)
    s_smooth.on_changed(update)

    # ─── SAVE BUTTON ─────────────────────────────────────────────────────────
    saveax = plt.axes([0.4, 0.025, 0.2, 0.05])
    button = Button(saveax, 'Save Maps to NPY', color='lightblue', hovercolor='0.975')

    def save_data(event):
        """Called when the Save button is clicked."""
        print("\nSaving with current parameters...")
        final_mask = make_bone_mask(target_slice, s_hu.val, s_close.val, s_open.val, s_smooth.val)
        
        c_map     = np.where(final_mask, C_BONE,     C_WATER).astype(np.float32)
        rho_map   = np.where(final_mask, RHO_BONE,   RHO_WATER).astype(np.float32)
        alpha_map = np.where(final_mask, ALPHA_BONE, ALPHA_WATER).astype(np.float32)
        
        out_dir = Path(CT_PATH).parent
        np.save(out_dir / "c_map_2d.npy",     c_map)
        np.save(out_dir / "rho_map_2d.npy",   rho_map)
        np.save(out_dir / "alpha_map_2d.npy", alpha_map)
        np.save(out_dir / "bone_mask_2d.npy", final_mask)
        
        print(f"✅ Maps successfully saved to {out_dir}")
        print(f"Grid shape: {c_map.shape}, Voxel size: {px:.4f} x {py:.4f} mm")

    button.on_clicked(save_data)

    plt.show()

if __name__ == "__main__":
    main()