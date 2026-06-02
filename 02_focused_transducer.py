import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import tone_burst
import matplotlib.pyplot as plt

# --- Parameters ---
freq  = 500e3
c0    = 1482.0
rho0  = 1000.0

# --- Grid ---
wavelength = c0 / freq
dx = wavelength / 6
Nx, Ny = 256, 256
kgrid = kWaveGrid([Nx, Ny], [dx, dx])
kgrid.makeTime(c0)
n_time = kgrid.Nt

# --- Medium ---
medium = kWaveMedium(sound_speed=c0, density=rho0)

# --- Transducer geometry ---
focal_length = 30e-3
aperture     = 30e-3
transducer_x = 10
center_y     = Ny // 2
aperture_pts = int(aperture / dx)
half_ap      = aperture_pts // 2
source_y_indices = np.arange(center_y - half_ap, center_y + half_ap)
n_sources = len(source_y_indices)

print(f"Grid size: {Nx} x {Ny}, dt={kgrid.dt:.2e}s, Nt={n_time}")
print(f"Number of source elements: {n_sources}")

# --- Build base tone burst ---
burst = tone_burst(1/kgrid.dt, freq, 5).flatten()
print(f"Burst length: {len(burst)} samples")

# Make sure n_time is long enough for burst + max delay
focal_point_x = transducer_x + int((focal_length + 3.5e-3) / dx)  # +3.5mm correction
focal_point_y = center_y

# Compute delays
delays_s = np.zeros(n_sources)
for i, sy in enumerate(source_y_indices):
    dist = np.sqrt(((transducer_x - focal_point_x) * dx)**2 +
                   ((sy - focal_point_y) * dx)**2)
    delays_s[i] = dist / c0

# Normalize so max delay fires first (converging focus)
delays_s = delays_s.max() - delays_s
max_delay_steps = int(np.max(delays_s) / kgrid.dt)

# Ensure time array is long enough
n_time_needed = max_delay_steps + len(burst) + 100
if n_time < n_time_needed:
    n_time = n_time_needed
    print(f"Extended n_time to {n_time}")

# --- Build source signals ---
source_signals = np.zeros((n_sources, n_time))
for i in range(n_sources):
    d = int(delays_s[i] / kgrid.dt)
    source_signals[i, d:d + len(burst)] = burst * 1e5

# Rebuild kgrid time with correct n_time
kgrid.t_array = np.arange(0, n_time) * kgrid.dt

source = kSource()
source.p_mask = np.zeros((Nx, Ny))
source.p_mask[transducer_x, source_y_indices] = 1
source.p = source_signals

# --- Sensor ---
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny))
sensor.record = ['p_max']

# --- Run ---
print("Running simulation...")
sensor_data = kspaceFirstOrder(kgrid, medium, source, sensor)

# --- Plot ---
p_max = sensor_data['p_max'].reshape(Nx, Ny)

plt.figure(figsize=(8, 7))
plt.imshow(p_max.T, aspect='equal',
           extent=[0, Nx*dx*1e3, 0, Ny*dx*1e3],
           origin='lower', cmap='hot')
plt.colorbar(label='Peak pressure (Pa)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Focused transducer — 2D pressure field (500 kHz, f=30mm)')
plt.axvline(x=(transducer_x + int(focal_length/dx)) * dx * 1e3,
            color='cyan', linestyle='--', linewidth=0.8, label='focal point')
plt.legend()
plt.tight_layout()
plt.savefig('02_focused_transducer.png', dpi=150)
plt.show()
print(f"Max pressure: {np.max(p_max):.1f} Pa")
print("Done! Saved to 02_focused_transducer.png")