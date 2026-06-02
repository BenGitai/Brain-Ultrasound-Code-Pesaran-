import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.signals import tone_burst
import matplotlib.pyplot as plt

# --- Parameters (Mueller et al. 500 kHz setup) ---
freq = 500e3          # 500 kHz
c0 = 1482.0           # speed of sound in water (m/s)
rho0 = 1000.0         # density of water (kg/m^3)

# --- Grid setup ---
wavelength = c0 / freq          # ~3 mm
dx = wavelength / 6             # 6 points per wavelength
Nx = 128
Ny = 128

kgrid = kWaveGrid([Nx, Ny], [dx, dx])
kgrid.makeTime(c0)

# --- Medium (pure water) ---
medium = kWaveMedium(sound_speed=c0, density=rho0)

# --- Source ---
source = kSource()
source.p_mask = np.zeros((Nx, Ny))
source.p_mask[Nx // 2, 10] = 1
source.p = tone_burst(1/kgrid.dt, freq, 5)

# --- Sensor ---
sensor = kSensor()
sensor.mask = np.zeros((Nx, Ny))
sensor.mask[Nx // 2, :] = 1

# --- Run ---
sim_options = SimulationOptions(
    save_to_disk=True,
    save_to_disk_exit=False,
    data_cast='single'
)
execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

sensor_data = kspaceFirstOrder(kgrid, medium, source, sensor)

# --- Plot ---
# Replace the plot section with this:
plt.figure(figsize=(10, 4))
p_max = np.max(np.abs(sensor_data['p']), axis=0)
plt.plot(p_max)
plt.xlabel('Grid position (y)')
plt.ylabel('Peak pressure (Pa)')
plt.title('Water baseline — peak pressure along sensor line')
plt.tight_layout()
plt.savefig('01_water_baseline.png')
plt.show()
print(f"Max pressure: {np.max(p_max):.2f} Pa")