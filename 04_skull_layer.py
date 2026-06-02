import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import tone_burst
import matplotlib.pyplot as plt

# --- Parameters ---
freq         = 500e3
c0           = 1482.0
rho0         = 1000.0
focal_length = 30e-3
aperture     = 30e-3
transducer_x = 10

wavelength = c0 / freq
dx = wavelength / 6

Nx, Ny = 256, 256

# --- Skull properties (Mueller et al.) ---
c_skull   = 2900.0
rho_skull = 1900.0

# Build grid using skull speed for stable dt
kgrid = kWaveGrid([Nx, Ny], [dx, dx])
kgrid.makeTime(c_skull, cfl=0.3)
n_time = kgrid.Nt
print(f"dt={kgrid.dt:.2e}s, Nt={n_time}")

# --- Skull geometry ---
skull_thickness_mm = 3.0
skull_start_x = transducer_x + int(20e-3 / dx)
skull_end_x   = skull_start_x + int(skull_thickness_mm * 1e-3 / dx)
print(f"Skull: x={skull_start_x*dx*1e3:.1f}mm to {skull_end_x*dx*1e3:.1f}mm")

# --- Heterogeneous medium (NO attenuation map — keep it simple first) ---
sound_speed_map = np.ones((Nx, Ny), dtype=float) * c0
density_map     = np.ones((Nx, Ny), dtype=float) * rho0
sound_speed_map[skull_start_x:skull_end_x, :] = c_skull
density_map[skull_start_x:skull_end_x, :]     = rho_skull

medium = kWaveMedium(
    sound_speed=sound_speed_map,
    density=density_map,
)

# --- Source ---
center_y     = Ny // 2
aperture_pts = int(aperture / dx)
half_ap      = aperture_pts // 2
source_y_indices = np.arange(center_y - half_ap, center_y + half_ap)
n_sources    = len(source_y_indices)

focal_point_x = transducer_x + int((focal_length + 3.5e-3) / dx)
focal_point_y = center_y

burst = tone_burst(1/kgrid.dt, freq, 5).flatten()
print(f"Burst length: {len(burst)} samples")

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
    end = min(d + len(burst), n_time)
    source_signals[i, d:end] = burst[:end-d] * 1e5

source = kSource()
source.p_mask = np.zeros((Nx, Ny))
source.p_mask[transducer_x, source_y_indices] = 1
source.p = source_signals

# --- Sensor ---
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny))
sensor.record = ['p_max']

# --- Run with skull ---
print("Running with skull...")
sensor_data = kspaceFirstOrder(kgrid, medium, source, sensor)
p_max_skull = sensor_data['p_max'].reshape(Nx, Ny)

# Check for NaN
if np.isnan(p_max_skull).any():
    print("WARNING: NaN detected in skull simulation")
else:
    print(f"Skull sim OK — max pressure: {np.nanmax(p_max_skull)/1e3:.1f} kPa")

# --- Run water baseline ---
medium_water = kWaveMedium(sound_speed=c0, density=rho0)
print("Running water baseline...")
sensor_data_water = kspaceFirstOrder(kgrid, medium_water, source, sensor)
p_max_water = sensor_data_water['p_max'].reshape(Nx, Ny)
print(f"Water sim OK — max pressure: {np.max(p_max_water)/1e3:.1f} kPa")

# --- Results ---
# Only measure in post-skull region to avoid reflection artifacts
post_skull_water = p_max_water[skull_end_x:, :]
post_skull_skull = p_max_skull[skull_end_x:, :]

peak_water = np.max(post_skull_water)
peak_skull = np.max(post_skull_skull)
transmission = peak_skull / peak_water * 100

print(f"\n=== SKULL VALIDATION ===")
print(f"Peak pressure (water, post-skull):      {peak_water/1e3:.1f} kPa")
print(f"Peak pressure (with skull, post-skull): {peak_skull/1e3:.1f} kPa")
print(f"Transmission:               {transmission:.1f}%")
print(f"Pressure loss:              {100-transmission:.1f}%")
print(f"Mueller et al. expected:    ~65% transmission (35% loss)")

# --- Reflection coefficient check ---
Z_water = rho0 * c0
Z_skull = rho_skull * c_skull
R = ((Z_skull - Z_water) / (Z_skull + Z_water))**2
T_theoretical = 1 - R
print(f"\nTheoretical transmission (impedance only): {T_theoretical*100:.1f}%")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vmax = np.max(p_max_water)

ax = axes[0]
im = ax.imshow(p_max_water.T, aspect='equal',
               extent=[0, Nx*dx*1e3, 0, Ny*dx*1e3],
               origin='lower', cmap='hot', vmax=vmax)
ax.set_title('Water baseline')
ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
plt.colorbar(im, ax=ax, label='Pa')

ax = axes[1]
im = ax.imshow(p_max_skull.T, aspect='equal',
               extent=[0, Nx*dx*1e3, 0, Ny*dx*1e3],
               origin='lower', cmap='hot', vmax=vmax)
ax.axvline(x=skull_start_x*dx*1e3, color='cyan', lw=1)
ax.axvline(x=skull_end_x*dx*1e3,   color='cyan', lw=1, label='skull')
ax.set_title(f'With 3mm skull\n{transmission:.0f}% transmission')
ax.set_xlabel('x (mm)')
ax.legend(fontsize=7)
plt.colorbar(im, ax=ax, label='Pa')

ax = axes[2]
x_mm = np.arange(Nx) * dx * 1e3
ax.plot(x_mm, p_max_water[:, center_y], label='water', color='steelblue')
ax.plot(x_mm, p_max_skull[:, center_y], label='with skull', color='coral')
ax.axvspan(skull_start_x*dx*1e3, skull_end_x*dx*1e3,
           alpha=0.2, color='gray', label='skull')
ax.set_xlabel('x (mm)')
ax.set_ylabel('Peak pressure (Pa)')
ax.set_title('Axial profile comparison')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('04_skull_layer.png', dpi=150)
plt.show()
print("Saved to 04_skull_layer.png")