"""Build the interactive Tissue Optics Explorer (multi-site, self-contained .html).

Per site (finger | wrist) it runs:
  (1) a fluence sweep over wavelength x melanin x scattering x dermis (resumable
      cache), calibrates to absolute radiance (W/mm^2), log-normalises, and
      SVD-compresses the log-fluence fields (mean + K basis volumes + per-combo
      coefficients) so the viewer interpolates continuously from ~MB of data;
  (2) a white-MC detector sweep (mc_white) over a dense wavelength grid, recording
      photon paths once per wavelength, then Beer-Lambert-reweighting into a
      compact R[wl,SDS,mel,SO2,blood] tensor (absorption axes for free).
Output per site: outputs/data_<site>.json  ({DATA, DET}).
The `merge` step injects all available sites into the template -> one HTML with a
finger/wrist selector.

    python explorer_build.py test 100000 finger      # tiny grid, a few minutes
    python explorer_build.py validate 250000 wrist   # small grid for fast iteration
    python explorer_build.py sweep 500000 finger 12  # full grid; last argument = worker processes
    python explorer_build.py sweep 500000 wrist
    python explorer_build.py merge                   # combine -> outputs/tissue_optics_explorer.html

Physics is simplified for visualisation, not validated dosimetry.
"""
import sys
import json
import base64
import time
import shutil
import numpy as np
from pathlib import Path
import tov_common as T
import mc_core
import mc_white
import optics_nir as O

BIN = 5                                  # downsample (0.5 mm voxels)
KCOMP = 96                               # SVD components kept
SEED = 7
E0 = 1.0e-3                              # display scale: peak fluence rate mapped to 1 mW/mm^2 (illustrative)
DECADES = 5.0                           # dynamic range -> noise floor = E0/10^DECADES
NOISE = E0 / 10 ** DECADES
HOVER_CLASSES = ["epidermis", "plexus", "dermis", "fat", "muscle", "bone"]
SITE_ORDER = ["finger", "wrist"]
DATA_JSON = lambda s: T.OUT / f"data_{s}.json"


def lin(a, b, n):
    return [round(float(v), 4) for v in np.linspace(a, b, n)]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def downsample(F, b):
    F = F[:F.shape[0] // b * b, :F.shape[1] // b * b, :F.shape[2] // b * b]
    return F.reshape(F.shape[0] // b, b, F.shape[1] // b, b, F.shape[2] // b, b).mean((1, 3, 5))


def grids(mode, site):
    """(WLS, MELS, MUS, DERS) per mode + site. DERS from the site template."""
    ders = T.SITES[site]["dermis_grid"]
    if mode == "test":
        return lin(500, 2500, 7), [0.02, 0.30], [1.0], ders[:1]
    if mode == "validate":
        return lin(500, 2500, 14), [0.02, 0.18, 0.38], [0.8, 1.4], ders[:2]
    # full sweep: finger keeps the rich grid; wrist trimmed (3x3) to bound run time
    if site == "wrist":
        return lin(500, 2500, 68), [0.01, 0.04, 0.10, 0.20, 0.35], [0.7, 1.0, 1.4], ders[:3]
    return lin(500, 2500, 68), [0.01, 0.04, 0.10, 0.20, 0.35], [0.6, 0.9, 1.2, 1.6], ders


# ---- detector (white-MC reflectance) config ----
def det_grids(mode):
    if mode == "test":
        return lin(500, 2500, 28)
    if mode == "validate":
        return lin(500, 2500, 60)
    return lin(500, 2500, 136)             # ~15 nm over 500-2500
SDS_EDGES = [round(float(v), 3) for v in np.arange(0.4, 10.0001, 0.4)]
SDS_MID = [round((SDS_EDGES[i] + SDS_EDGES[i + 1]) / 2, 3) for i in range(len(SDS_EDGES) - 1)]
SO2S = [0.50, 0.65, 0.80, 0.90, 0.98]
BFS = [0.50, 0.75, 1.00, 1.50, 2.00]
DET_FLOOR = 1.0e-6
DET_ADET = 1.0


# ---- per-job workers (module level so a process pool can pickle them) ----
def _fluence_job(args):
    """One fluence combo -> cached downsampled field + metadata. Returns (Fd, md)."""
    site, wl, mel, mus, der, nphoton, fp, mp = args
    T.set_site(site)                     # workers start fresh; geometry globals are per site
    model = T.get_model(wl, melanin=mel, mus_scale=mus, dermis_um=der)
    res = mc_core.simulate(model, nphoton=nphoton, seed=SEED)
    Fd = downsample(res["F"], BIN).astype(np.float32)
    p = model["props"]
    md = dict(epi_mua=p["epidermis"]["mua"], derm_mua=p["dermis"]["mua"],
              derm_musp=p["dermis"]["musp"], fmax=float(Fd.max()),
              pct={c: float(res["abs_pct_by_class"][i]) for i, c in enumerate(T.CLASSES)})
    np.save(fp, Fd)
    Path(mp).write_text(json.dumps(md))
    return Fd, md


def _det_job(args):
    """One detector wavelength: white MC once, then Beer-Lambert reweighting. Returns R[SDS,mel,SO2,blood]."""
    site, wl, mels, det_nphoton, cp = args
    T.set_site(site)                     # workers start fresh; geometry globals are per site
    nB, nM, nO, nF = len(SDS_MID), len(mels), len(SO2S), len(BFS)
    areas = np.array([np.pi * (SDS_EDGES[i + 1] ** 2 - SDS_EDGES[i] ** 2) for i in range(nB)])
    model = T.get_model(wl, melanin=mels[0], mus_scale=1.0)
    r = mc_white.simulate_white(model, nphoton=det_nphoton, seed=SEED)
    classes = r["classes"]
    binidx = np.digitize(r["rho"], SDS_EDGES) - 1
    valid = (binidx >= 0) & (binidx < nB)
    binidx = binidx[valid]; L = r["Lpath"][valid].astype(np.float64)
    mua_grid = np.empty((nM, nO, nF, len(classes)))
    for im, mel in enumerate(mels):
        for io, so2 in enumerate(SO2S):
            for ib, bf in enumerate(BFS):
                mua_grid[im, io, ib] = [O.tissue_optics(c, wl, melanin=mel, so2=so2,
                                                        blood_scale=bf)["mua"] for c in classes]
    w0N = r["w0"] / r["n_launched"]
    R = np.zeros((nB, nM, nO, nF), np.float32)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for im in range(nM):
            for io in range(nO):
                for ib in range(nF):
                    w = w0N * np.exp(-(L @ mua_grid[im, io, ib]))
                    Rbin = np.bincount(binidx, weights=w, minlength=nB)[:nB]
                    R[:, im, io, ib] = (Rbin / areas).astype(np.float32)
    np.save(cp, R)
    return R


def _run_parallel(fn, jobs, workers, label):
    """Fill the cache with a process pool; results are read back from the cache afterwards."""
    if not jobs:
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time(); done = 0
    print(f"  {label}: {len(jobs)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for f in as_completed(futs):
            f.result(); done += 1
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"    {done}/{len(jobs)}  {el:.0f}s elapsed, ~{el/done*(len(jobs)-done):.0f}s left", flush=True)


def build_site(mode, nphoton, site, workers=1):
    T.set_site(site)
    WLS, MELS, MUS, DERS = grids(mode, site)
    GRID = [len(WLS), len(MELS), len(MUS), len(DERS)]
    TOTAL = int(np.prod(GRID))
    CACHE = T.OUT / f"_cache_{site}_{mode}"; CACHE.mkdir(exist_ok=True)
    print(f"\n=== [{site}] {mode} : grid {GRID} = {TOTAL} combos, {nphoton:.0e} photons, "
          f"domain {T.NX}x{T.NY}x{T.NZ} ===")

    # ---- fluence sweep (resumable) ----
    if workers > 1:
        jobs = []
        for iw, wl in enumerate(WLS):
            for im, mel in enumerate(MELS):
                for iu, mus in enumerate(MUS):
                    for idd, der in enumerate(DERS):
                        key = f"{iw}_{im}_{iu}_{idd}"
                        fp = CACHE / f"{key}.npy"; mp = CACHE / f"{key}.json"
                        if not (fp.exists() and mp.exists()):
                            jobs.append((site, wl, mel, mus, der, nphoton, str(fp), str(mp)))
        _run_parallel(_fluence_job, jobs, workers, f"[{site}] fluence")
    t0 = time.time(); fields = {}; metas = {}; n = 0
    keys = []
    for iw, wl in enumerate(WLS):
        for im, mel in enumerate(MELS):
            for iu, mus in enumerate(MUS):
                for idd, der in enumerate(DERS):
                    key = f"{iw}_{im}_{iu}_{idd}"; keys.append(key)
                    fp = CACHE / f"{key}.npy"; mp = CACHE / f"{key}.json"; n += 1
                    if fp.exists() and mp.exists():
                        fields[key] = np.load(fp); metas[key] = json.loads(mp.read_text()); continue
                    Fd, md = _fluence_job((site, wl, mel, mus, der, nphoton, str(fp), str(mp)))
                    fields[key] = Fd; metas[key] = md
                    if n % 25 == 0 or n == TOTAL:
                        el = time.time() - t0
                        print(f"  [{n}/{TOTAL}] {el:.0f}s elapsed, ~{el/n*(TOTAL-n):.0f}s left")
    print(f"  fluence grid ready in {time.time()-t0:.0f}s")

    # ---- absolute radiance calibration + log-normalise ----
    nz = fields[keys[0]].shape
    gmax_rel = max(float(F.max()) for F in fields.values())
    cal = E0 / gmax_rel
    lo, hi = np.log10(NOISE), np.log10(E0)
    ix, iy, iz = np.indices(nz)
    xc = (ix + 0.5) * T.DX * BIN; zc = (iz + 0.5) * T.DX * BIN
    inside = (np.hypot(xc - T.CX, zc - T.CZ) <= T.RF).ravel()

    def to_lg(F):
        phi = np.clip(F.ravel() * cal, NOISE, E0)
        lg = (np.log10(phi) - lo) / (hi - lo)
        lg[~inside] = 0.0
        return lg.astype(np.float32)

    X = np.stack([to_lg(fields[k]) for k in keys])
    NV = X.shape[1]

    # ---- SVD compression (Gram-matrix PCA) ----
    mean = X.mean(0); Xc = X - mean
    G = Xc @ Xc.T
    ev, evec = np.linalg.eigh(G)
    order = np.argsort(ev)[::-1]
    K = min(KCOMP, int((ev > 1e-9).sum()))
    sel = order[:K]
    s = np.sqrt(np.maximum(ev[sel], 1e-30))
    U = evec[:, sel]
    coeff = U * s
    basis = (Xc.T @ (U / s)).T
    rmse = float(np.sqrt(((mean + coeff @ basis - X) ** 2).mean()))
    print(f"  SVD: K={K}, reconstruction RMSE={rmse:.4f}")

    mean_u8 = np.clip(mean * 255, 0, 255).astype(np.uint8)
    scales = np.abs(basis).max(1); scales[scales == 0] = 1.0
    basis_i8 = np.clip(np.round(basis / scales[:, None] * 127), -127, 127).astype(np.int8)
    coeff_f = coeff.astype(np.float32)
    meta_arr = lambda f: [round(f(metas[k]), 5) for k in keys]
    pct_grid = [[round(metas[k]["pct"][c] * 100, 2) for c in HOVER_CLASSES] for k in keys]

    DATA = dict(
        axes=dict(wavelength=WLS, melanin=MELS, mus_scale=MUS, dermis_um=DERS),
        grid_shape=GRID, dims=list(nz), dx=T.DX * BIN,
        geom=dict(RF=T.RF, CX=T.CX, CZ=T.CZ, r_bone=T.R_BONE, bone_core=T.BONE_CORE,
                  source_y=T.NY * T.DX / 2.0, colors=T.COLORS, hover_classes=HOVER_CLASSES,
                  layers=T._SITE_CFG["layers"], site=site, site_label=T._SITE_CFG["label"]),
        basis=dict(mean=b64(mean_u8), comp=b64(basis_i8),
                   scales=[round(float(v), 8) for v in scales], K=int(K)),
        coeff=b64(coeff_f),
        meta=dict(epi_mua=meta_arr(lambda m: m["epi_mua"]),
                  derm_mua=meta_arr(lambda m: m["derm_mua"]),
                  derm_musp=meta_arr(lambda m: m["derm_musp"]),
                  maxphi=[round(metas[k]["fmax"] * cal, 9) for k in keys],
                  pct=pct_grid),
        radiance=dict(E0=E0, noise=NOISE, decades=DECADES, units="W/mm²"),
        nphoton=nphoton, rmse=round(rmse, 4),
    )

    # ---- detector white-MC sweep + reweight tensor ----
    WLS_DET = det_grids(mode)
    DET_NPHOTON = max(nphoton, 1_000_000)
    DET_MELS = MELS
    nB, nM, nO, nF = len(SDS_MID), len(DET_MELS), len(SO2S), len(BFS)
    DCACHE = T.OUT / f"_cache_{site}_{mode}_det"; DCACHE.mkdir(exist_ok=True)
    print(f"  [detector] {len(WLS_DET)} wl x {nB} SDS x {nM}x{nO}x{nF}, {DET_NPHOTON:.0e} photons")
    areas = np.array([np.pi * (SDS_EDGES[i + 1] ** 2 - SDS_EDGES[i] ** 2) for i in range(nB)])
    cfg = f"{DET_NPHOTON}_{nB}_{nM}{nO}{nF}_{int(SDS_EDGES[-1]*10)}"
    Rten = np.zeros((len(WLS_DET), nB, nM, nO, nF), np.float32)
    if workers > 1:
        jobs = [(site, wl, DET_MELS, DET_NPHOTON, str(DCACHE / f"{cfg}_{iw}.npy"))
                for iw, wl in enumerate(WLS_DET) if not (DCACHE / f"{cfg}_{iw}.npy").exists()]
        _run_parallel(_det_job, jobs, workers, f"[{site}] detector")
    td = time.time()
    for iw, wl in enumerate(WLS_DET):
        cp = DCACHE / f"{cfg}_{iw}.npy"
        if cp.exists():
            Rten[iw] = np.load(cp); continue
        Rten[iw] = _det_job((site, wl, DET_MELS, DET_NPHOTON, str(cp)))
        if (iw + 1) % 20 == 0 or iw + 1 == len(WLS_DET):
            el = time.time() - td
            print(f"    [{iw+1}/{len(WLS_DET)}] {el:.0f}s, ~{el/(iw+1)*(len(WLS_DET)-iw-1):.0f}s left")
    print(f"  detector sweep done in {time.time()-td:.0f}s")

    RL = np.log10(np.maximum(Rten, 1e-12)).astype(np.float16)
    DET = dict(wavelengths=[round(float(w), 1) for w in WLS_DET], sds=SDS_MID,
               melanin=DET_MELS, so2=SO2S, bf=BFS, shape=list(RL.shape), logR=b64(RL),
               Adet=DET_ADET, floor=DET_FLOOR, nphoton=DET_NPHOTON)

    DATA_JSON(site).write_text(json.dumps(dict(DATA=DATA, DET=DET)))
    print(f"  wrote {DATA_JSON(site).name}  ({DATA_JSON(site).stat().st_size/1e6:.1f} MB)")
    if mode == "test":
        shutil.rmtree(CACHE, ignore_errors=True)
    return dict(DATA=DATA, DET=DET)


def merge():
    sites = {}
    for s in SITE_ORDER:
        if DATA_JSON(s).exists():
            sites[s] = json.loads(DATA_JSON(s).read_text())
            print(f"  + {s}  ({DATA_JSON(s).stat().st_size/1e6:.1f} MB)")
    if not sites:
        print("no data_<site>.json found; build a site first"); return
    html = Path(__file__).with_name("explorer_template.html").read_text()
    html = html.replace("/*__SITES__*/", json.dumps(dict(order=[s for s in SITE_ORDER if s in sites],
                                                         sites=sites)))
    out = T.OUT / "tissue_optics_explorer.html"
    out.write_text(html)
    print(f"saved {out}  ({out.stat().st_size/1e6:.1f} MB, sites: {list(sites)})")


if __name__ == "__main__":
    MODE = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if MODE == "merge":
        merge()
    else:
        NPHOTON = int(float(sys.argv[2])) if len(sys.argv) > 2 else 350_000
        SITE = sys.argv[3] if len(sys.argv) > 3 else "finger"
        WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        build_site(MODE, NPHOTON, SITE, workers=WORKERS)
        merge()
