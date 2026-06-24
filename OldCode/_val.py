"""Exec the notebook cells with kspaceFirstOrder stubbed, to validate every non-sim code
path (lens stamp, crop+transpose, medium build, focus, plotting) without the GPU."""
import nbformat, numpy as np, matplotlib
matplotlib.use("Agg")
nb = nbformat.read("lens_focus.ipynb", as_version=4)
ns = {}
def stub(kgrid, medium, source, sensor, **kw):
    n = int(np.asarray(sensor.mask).sum())
    return {"p_max": np.random.default_rng(0).random(n).astype(np.float32)}
for i, c in enumerate([c for c in nb.cells if c.cell_type == "code"]):
    if "from kwave.kspaceFirstOrder import kspaceFirstOrder" in c.source:
        pass
    try:
        exec(compile(c.source, f"<cell {i}>", "exec"), ns)
        ns["kspaceFirstOrder"] = stub   # override after imports cell
        print(f"[cell {i}] OK")
    except Exception as e:
        import traceback; print(f"[cell {i}] ERROR {type(e).__name__}: {e}"); traceback.print_exc(); break
print("done")
