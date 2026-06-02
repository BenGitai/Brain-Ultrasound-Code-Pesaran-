import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import tone_burst
import matplotlib.pyplot as plt

# --- Same parameters as 02 ---
freq         = 500e3
c0           = 1482.0
rho0         = 1000.0
focal_length = 30e-3
aperture     = 30e-3
transducer_x = 10

wavelength = c0 / freq
dx = wavelength / 6
Nx, Ny = 256, 256

kgrid = kWaveGrid([Nx, Ny], [dx, dx])
kgrid.makeTime(c0)
n_time = kgrid.Nt

medium = kWaveMedium(sound_speed=c0, density=rho0)

# --- Source (identical to 02) ---
center_y     = Ny // 2
aperture_pts = int(aperture / dx)
half_ap      = aperture_pts // 2
source_y_indices = np.arange(center_y - half_ap, center_y + half_ap)
n_sources    = len(source_y_indices)

focal_point_x = transducer_x + int((focal_length + 3.5e-3) / dx)  # +3.5mm correction
focal_point_y = center_y

burst = tone_burst(1/kgrid.dt, freq, 5).flatten()

delays_s = np.zeros(n_sources)
for i, sy in enumerate(source_y_indices):
    dist = np.sqrt(((transducer_x - focal_point_x) * dx)**2 +
                   ((sy - focal_point_y) * dx)**2)
    delays_s[i] = dist / c0
delays_s = delays_s.max() - delays_s
max_delay_steps = int(np.max(delays_s) / kgrid.dt)

n_time_needed = max_delay_steps + len(burst) + 100
if n_time < n_time_needed:
    n_time = n_time_needed
kgrid.t_array = np.arange(0, n_time) * kgrid.dt

source_signals = np.zeros((n_sources, n_time))
for i in range(n_sources):
    d = int(delays_s[i] / kgrid.dt)
    source_signals[i, d:d + len(burst)] = burst * 1e5

source = kSource()
source.p_mask = np.zeros((Nx, Ny))
source.p_mask[transducer_x, source_y_indices] = 1
source.p = source_signals

sensor = kSensor()
sensor.mask = np.ones((Nx, Ny))
sensor.record = ['p_max']

print("Running simulation...")
sensor_data = kspaceFirstOrder(kgrid, medium, source, sensor)
p_max = sensor_data['p_max'].reshape(Nx, Ny)

# --- Validation ---
print("\n=== VALIDATION ===")

# 1. Where is the actual peak?
peak_idx = np.unravel_index(np.argmax(p_max), p_max.shape)
peak_x_mm = peak_idx[0] * dx * 1e3
peak_y_mm = peak_idx[1] * dx * 1e3
expected_x_mm = (transducer_x + int(focal_length / dx)) * dx * 1e3
print(f"\n[1] Focal position")
print(f"    Expected x: {expected_x_mm:.1f} mm")
print(f"    Actual peak x: {peak_x_mm:.1f} mm")
print(f"    Offset: {abs(peak_x_mm - expected_x_mm):.1f} mm")

# 2. Beamwidth (FWHM) at focal plane
focal_profile = p_max[peak_idx[0], :]   # lateral slice at peak x
half_max = np.max(focal_profile) / 2
above_half = np.where(focal_profile >= half_max)[0]
if len(above_half) > 1:
    fwhm_mm = (above_half[-1] - above_half[0]) * dx * 1e3
else:
    fwhm_mm = 0
print(f"\n[2] Beamwidth (FWHM)")
print(f"    Simulated FWHM: {fwhm_mm:.1f} mm")
print(f"    Mueller et al.: ~4-6 mm")
print(f"    Match: {'YES' if 2 < fwhm_mm < 8 else 'CHECK'}")

# 3. Peak pressure
peak_pa  = np.max(p_max)
peak_kpa = peak_pa / 1e3
print(f"\n[3] Peak pressure at focus")
print(f"    Simulated: {peak_kpa:.0f} kPa")
print(f"    Input source: 100 kPa")
print(f"    Gain from focusing: {peak_kpa/100:.1f}x")
print(f"    Mueller et al. (scaled to 1030 kPa): would need {1030/peak_kpa:.1f}x more input")

# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: 2D field
ax = axes[0]
im = ax.imshow(p_max.T, aspect='equal',
               extent=[0, Nx*dx*1e3, 0, Ny*dx*1e3],
               origin='lower', cmap='hot')
ax.axvline(x=expected_x_mm, color='cyan', linestyle='--', lw=0.8, label='target focal x')
ax.axvline(x=peak_x_mm,     color='lime', linestyle='--', lw=0.8, label='actual peak x')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_title('2D pressure field')
ax.legend(fontsize=7)
plt.colorbar(im, ax=ax, label='Pa')

# Plot 2: Axial profile (along beam axis, through center)
ax = axes[1]
axial = p_max[:, center_y]
x_mm  = np.arange(Nx) * dx * 1e3
ax.plot(x_mm, axial)
ax.axvline(x=expected_x_mm, color='cyan', linestyle='--', lw=0.8, label='target focal x')
ax.axvline(x=peak_x_mm,     color='lime', linestyle='--', lw=0.8, label='actual peak')
ax.set_xlabel('x (mm)')
ax.set_ylabel('Peak pressure (Pa)')
ax.set_title('Axial profile (along beam)')
ax.legend(fontsize=7)

# Plot 3: Lateral profile (FWHM)
ax = axes[2]
y_mm = np.arange(Ny) * dx * 1e3
ax.plot(y_mm, focal_profile)
ax.axhline(y=half_max, color='red', linestyle='--', lw=0.8, label=f'half max')
ax.axvline(x=above_half[0]*dx*1e3,  color='orange', linestyle=':', lw=0.8)
ax.axvline(x=above_half[-1]*dx*1e3, color='orange', linestyle=':', lw=0.8, label=f'FWHM={fwhm_mm:.1f}mm')
ax.set_xlabel('y (mm)')
ax.set_ylabel('Peak pressure (Pa)')
ax.set_title('Lateral profile at focus')
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('03_validate.png', dpi=150)
plt.show()
print("\nSaved to 03_validate.png")