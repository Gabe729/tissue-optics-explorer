"""CLI wrapper: run the CPU Monte Carlo for one finger model and save the field.

The MC itself lives in mc_core.simulate (shared with the interactive explorer).
Writes outputs/finger_<wl>nm.{npy,json} (relative fluence) for the Blender /
matplotlib renderers.

    python mc_finger.py [wavelength_nm] [nphoton]      # e.g. python mc_finger.py 850 1e6
"""
import sys
import time
import tov_common as T
import mc_core

WL = float(sys.argv[1]) if len(sys.argv) > 1 else 850.0
N = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_000_000

model = T.get_model(WL)
t0 = time.time()
res = mc_core.simulate(model, nphoton=N)
dt = time.time() - t0
print(f"[cpu-mc] {model['label']}  N={N}  steps<={res['steps']}  t={dt:.1f}s  "
      f"absorbed={res['absorbed']:.3f} escaped={res['escaped']:.3f} "
      f"sum={res['absorbed'] + res['escaped']:.3f} (launched {res['launched']:.3f})")

T.save_field(f"finger_{WL:.0f}nm", res["F"], dict(
    engine="CPU Monte Carlo (NumPy; HG scatter, MCML-style roulette)",
    quantity="fluence (track-length estimator, relative units)",
    model=model["label"], wavelength=WL, nphoton=N,
    props={k: model["props"][k] for k in model["classes"]},
    status="Illustrative: simplified transport for visualisation; not validated dosimetry",
))
print(f"saved outputs/finger_{WL:.0f}nm.npy")
