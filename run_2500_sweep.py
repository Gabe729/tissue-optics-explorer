"""One-shot driver: rebuild finger + wrist sweeps over 500-2500 nm, then merge.

Runs both sites' fluence + detector sweeps (resumable cache) and merges into
outputs/tissue_optics_explorer.html. Single merge at the end (not per-site) so the
deliverable is never left half-updated.

    python run_2500_sweep.py 500000 12 > outputs/build_2500.log 2>&1

Arguments: photons per fluence combo (default 1e6), worker processes (default 1).
Each worker needs about 0.5 GB of memory. The full grid is ~8,500 fluence combos and
~270 detector wavelengths: about 18 h on one core, about 2 h on 12 cores of a recent laptop.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker; parallelism comes from processes

import sys
import time
import explorer_build as E


def main():
    nph = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1_000_000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t0 = time.time()
    for site in ("finger", "wrist"):
        print(f"\n######## BUILD {site} @ {nph:.0e} photons, {workers} workers ########", flush=True)
        E.build_site("sweep", nph, site, workers=workers)
        print(f"   [{site}] done; cumulative {(time.time() - t0) / 60:.1f} min", flush=True)
    E.merge()
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
