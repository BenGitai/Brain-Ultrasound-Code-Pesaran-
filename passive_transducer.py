import numpy as np
import jax.numpy as jnp
import jax
# Temporarily disable JIT compilation to see if the code is actually running
jax.config.update('jax_disable_jit', True)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import argparse

# k-wave imports
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.signals import tone_burst

# j-wave imports (UPDATED FOR CORRECT API)
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.acoustics import simulate_wave_propagation
from jaxdf.discretization import OnGrid

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
N_ANGLES = 12                     
MAX_ITER = 50                     
PATIENCE = 5                      
LEARNING_RATE = 0.01

FREQ = 500e3                      
OMEGA = 2 * np.pi * FREQ

C_WATER, RHO_WATER = 1482.0, 1000.0
C_GEL, RHO_GEL = 1000.0, 1000.0   
C_SKULL = 2900.0
C_BRAIN = 1560.0

# ==========================================
# 2. Environment & Data Loading
# ==========================================
def load_environment():
    print("Loading monkey skull CT data and initializing grid...")
    Nx, Ny = 256, 256
    dx = 1e-3  
    
    # --- DUMMY DATA FOR SCRIPT SKELETON ---
    c_map = np.full((Nx, Ny), C_BRAIN)
    rho_map = np.full((Nx, Ny), 1000.0)
    bone_mask = np.zeros((Nx, Ny))
    # --------------------------------------

    target_x, target_y = Nx // 2, Ny // 2
    
    kgrid = kWaveGrid([Nx, Ny], [dx, dx])
    medium = kWaveMedium(sound_speed=c_map, density=rho_map)
    
    return Nx, Ny, dx, c_map, rho_map, bone_mask, target_x, target_y, kgrid, medium

# ==========================================
# 3. Helper Functions
# ==========================================
def compute_holographic_gel(p_recorded, k_gel, k_water, tx_mask, dt, target_freq):
    N_time = p_recorded.shape[1]
    P_fft = np.fft.rfft(p_recorded, axis=1)
    freqs = np.fft.rfftfreq(N_time, d=dt)
    target_idx = np.argmin(np.abs(freqs - target_freq))
    phase = np.angle(P_fft[:, target_idx])
    
    delta_k = k_gel - k_water
    if delta_k == 0: delta_k = 1e-6 
        
    thickness_vals = (phase - np.min(phase)) / delta_k 
    thickness_vals = np.maximum(thickness_vals, 0.0) 
    
    thickness_map = np.zeros_like(tx_mask)
    xs, ys = np.where(tx_mask > 0.5)
    
    for i in range(len(xs)):
        if i < len(thickness_vals):
            thickness_map[xs[i], ys[i]] = thickness_vals[i]
            
    return thickness_map

def build_soft_gel_mask(transducer_mask, thickness):
    return transducer_mask * thickness

def scan_angle(ang, Nx, Ny):
    tx_pixels = [(Nx//4, Ny//4)] 
    return {'tx_pixels': tx_pixels}

# ==========================================
# 4. FWI Objective Function (JAX) - UPDATED API
# ==========================================
def create_objective_fn(Nx, Ny, dx, target_x, target_y, bone_mask, tof):
    """
    Creates a JIT-compiled loss function tailored to the Time-of-Flight
    of the current transducer angle.
    """
    domain = Domain(N=(Nx, Ny), dx=(dx, dx))
    
    # 1. Define a reference medium with the maximum possible sound speed 
    # to pre-calculate a statically safe dt and step count for JAX compilation.
    max_c = float(max(C_SKULL, C_BRAIN, C_GEL, C_WATER))
    ref_c_map = np.full((Nx, Ny), max_c)
    ref_rho_map = np.full((Nx, Ny), 1000.0)
    
    ref_medium = Medium(
        domain=domain, 
        sound_speed=OnGrid(ref_c_map, domain), 
        density=OnGrid(ref_rho_map, domain)
    )
    
    # 2. Pre-compute the static TimeAxis using JWave's native constructor
    time_axis = TimeAxis.from_medium(ref_medium, cfl=0.3, t_end=float(tof))
    
    @jax.jit
    def objective_fn(gel_thickness, transducer_mask, base_c_map, base_rho_map):
        gel_mask = build_soft_gel_mask(transducer_mask, gel_thickness) 
        skull_penalty = jnp.sum(gel_mask * bone_mask) * 1e5
        
        c_map_sim = jnp.where(gel_mask > 0.5, C_GEL, base_c_map)
        rho_map_sim = jnp.where(gel_mask > 0.5, RHO_GEL, base_rho_map)
        
        c_field = OnGrid(c_map_sim, domain)
        rho_field = OnGrid(rho_map_sim, domain)
        
        # 3. Build dynamic medium for this specific optimization step
        medium = Medium(domain=domain, sound_speed=c_field, density=rho_field)
        
        # 4. Use the mask as an initial pressure impulse (p0)
        p0_field = OnGrid(transducer_mask, domain) 
        
        # 5. Run J-Wave simulation
        out = simulate_wave_propagation(medium, time_axis, p0=p0_field)
        
        # 6. Extract final pressure frame safely
        if isinstance(out, tuple):
            p_out = out[0].on_grid
        elif hasattr(out, 'p'):
            p_out = out.p.on_grid
        else:
            p_out = out.on_grid
            
        # Target pressure at exactly the moment the impulse arrives (last time frame)
        p_target = p_out[-1, target_x, target_y]
        
        # Maximize focus pressure at target
        loss = -jnp.max(jnp.abs(p_target)) + skull_penalty
        
        return loss, jnp.max(jnp.abs(p_target))
        
    return objective_fn

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Holographic Transducer FWI Optimization")
    parser.add_argument('--gpu', action='store_true', help='Force JAX/k-wave to use GPU')
    args = parser.parse_args()

    Nx, Ny, dx, c_map, rho_map, bone_mask, cx, cy, kgrid, medium = load_environment()
    
    kgrid.makeTime(medium.sound_speed)
    dt = kgrid.dt
    t_burst = tone_burst(1/dt, FREQ, 5).flatten()
    
    angles = np.linspace(0, 360, N_ANGLES, endpoint=False)
    results = []

    best_overall_transducer = None
    best_overall_pressure = -np.inf
    best_overall_gel = None

    print("\nStarting the transducer sweep...")
    print(f"{'Angle':>6} | {'Iter':>5} | {'Target P (Pa)':>15} | {'Status'}")
    print("-" * 50)

    for ang in angles:
        print(f"\n--- Setting up Transducer at {ang} degrees ---")
        config = scan_angle(ang, Nx, Ny)
        if not config:
            print("Skipping: Invalid geometry for this angle.")
            continue
            
        tx_pixels = config['tx_pixels']
        tx_mask = np.zeros((Nx, Ny))
        for xi, yi in tx_pixels:
            tx_mask[xi, yi] = 1.0

        # --- Phase 1: Backward Simulation ---
        print("  -> Running k-wave backward simulation (Target -> Transducer)...")
        src_bwd = kSource()
        src_bwd.p_mask = np.zeros((Nx, Ny))
        src_bwd.p_mask[cx, cy] = 1
        src_bwd.p = t_burst
        
        sen_bwd = kSensor()
        sen_bwd.mask = tx_mask
        sen_bwd.record = ['p']
        
        device_flag = 'gpu' if args.gpu else 'cpu'
        data_bwd = kspaceFirstOrder(kgrid, medium, src_bwd, sen_bwd, device=device_flag, dtype='float32', quiet=True)
        p_recorded = data_bwd['p']
        
        print("  -> Backward simulation complete. Initializing gel thickness...")
        k_gel = OMEGA / C_GEL
        k_water = OMEGA / C_WATER
        initial_thickness = compute_holographic_gel(p_recorded, k_gel, k_water, tx_mask, dt, FREQ)
        
        gel_thickness = jnp.array(initial_thickness)
        base_c_map = jnp.array(c_map)
        base_rho_map = jnp.array(rho_map)
        
        # --- CALCULATE TIME OF FLIGHT ---
        xs, ys = np.where(tx_mask > 0.5)
        mean_tx_x, mean_tx_y = np.mean(xs), np.mean(ys)
        dist = np.sqrt((mean_tx_x - cx)**2 + (mean_tx_y - cy)**2) * dx
        tof = dist / C_WATER  
        
        # Create a fresh gradient function locked to this exact time of flight
        objective_fn = create_objective_fn(Nx, Ny, dx, cx, cy, bone_mask, tof)
        grad_fn = jax.jit(jax.value_and_grad(objective_fn, has_aux=True))

        # --- Phase 2: FWI Optimization Loop ---
        print("  -> Starting J-Wave optimization loop (If it hangs here, it's compiling JAX)...")
        best_p_tgt = -np.inf
        best_gel = gel_thickness
        no_improve_count = 0
        
        for i in range(MAX_ITER):
            (loss, p_tgt), grads = grad_fn(gel_thickness, tx_mask, base_c_map, base_rho_map)
            
            gel_thickness = gel_thickness - LEARNING_RATE * grads
            gel_thickness = jnp.maximum(gel_thickness, 0.0) 
            
            if p_tgt > best_p_tgt:
                best_p_tgt = p_tgt
                best_gel = gel_thickness
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            status_str = 'Opt...' if no_improve_count == 0 else 'No improve'
            print(f"      {ang:>6.1f} | {i:>5} | {float(p_tgt):>15.4e} | {status_str}")
            
            if no_improve_count >= PATIENCE:
                print(f"  -> Stopped early for transducer at {ang:.1f} deg (Patience reached).")
                break

        results.append({
            "angle": ang,
            "tx_mask": tx_mask,
            "best_p": float(best_p_tgt),
            "best_gel": np.array(best_gel)
        })
        
        if best_p_tgt > best_overall_pressure:
            best_overall_pressure = best_p_tgt
            best_overall_transducer = tx_mask
            best_overall_gel = best_gel

    # ==========================================
    # 6. Final Results & Visualization
    # ==========================================
    if best_overall_transducer is not None:
        print(f"\nOptimization Complete. Best Pressure: {float(best_overall_pressure):.4e} Pa")

        final_gel_mask = build_soft_gel_mask(best_overall_transducer, best_overall_gel)
        final_c_map = np.where(final_gel_mask > 0.5, C_GEL, c_map)

        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [0, Nx*dx*1e3, 0, Ny*dx*1e3]

        im = ax.imshow(final_c_map.T, origin='lower', extent=extent, cmap='viridis')
        ax.contour(bone_mask.T, levels=[0.5], extent=extent, colors='cyan', linewidths=0.8, alpha=0.6)
        ax.plot(cx*dx*1e3, cy*dx*1e3, 'r*', ms=10, label='Target')
        
        ax.set_title('Optimized Holographic Gel Profile & Skull Map')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend()
        plt.colorbar(im, ax=ax, label='Sound Speed (m/s)')

        plt.tight_layout()
        plt.savefig('fwi_optimization_result.png', dpi=300)
        print("Saved plot to 'fwi_optimization_result.png'")
        plt.show()
    else:
        print("\nNo valid transducers evaluated.")

if __name__ == "__main__":
    main()