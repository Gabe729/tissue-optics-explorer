"""Importable CPU Monte Carlo core for the Tissue Optics Explorer.

Pure NumPy voxel Monte Carlo (Henyey-Greenstein scatter, MCML-style roulette,
track-length fluence estimator) shared by the CLI (`mc_finger.py`) and the interactive
explorer (`explorer_build.py`). Reads a parameterised model from tov_common, so
wavelength / skin tone / scattering / layer thicknesses all flow through.

Runs on any CPU (roughly 10-20 s per million photons on a laptop). Physics
is simplified for visualisation, not validated dosimetry: no Fresnel reflection or
total internal reflection at exit, and no step splitting at internal boundaries.
"""
from __future__ import annotations
import numpy as np
import tov_common as T


def simulate(model: dict, nphoton: int = 300_000, seed: int = 7,
             maxstep: int = 4000) -> dict:
    """Monte Carlo fluence (track-length estimator) for `model`.

    Returns dict(F, absorbed, escaped, launched): F is the (NX,NY,NZ) relative
    fluence field; the scalars are per-launched-photon energy fractions.
    """
    rng = np.random.default_rng(seed)
    ld = model.get("layer_depth")
    mua_l = np.array([model["props"][c]["mua"] for c in model["classes"]])
    mus_l = np.array([model["props"][c]["mus"] for c in model["classes"]])
    g_l = np.array([model["props"][c]["g"] for c in model["classes"]])
    n_surf = model["props"]["epidermis"]["n"]
    dx, nx, ny, nz = T.DX, T.NX, T.NY, T.NZ
    cx, cz, Rf = T.CX, T.CZ, T.RF
    N = int(nphoton)

    # emitter: 1 mm x 4 mm strip (0.5 mm half-width in x, +/-2 mm along the axis y)
    # on the dorsal (max-z) surface, beam -z inward. A strip gives a clean
    # cross-section; mc_white uses a point-like disk for SDS-resolved reflectance.
    spot = 0.5
    ang = rng.uniform(0, 2 * np.pi, N)
    rr = spot * np.sqrt(rng.uniform(0, 1, N))
    x = cx + rr * np.cos(ang)
    y = ny * dx / 2 + rng.uniform(-2.0, 2.0, N)
    # place each photon just inside the curved skin surface (the cylinder is not
    # flat across the spot; a flat launch plane at the apex starts most photons in air)
    z = cz + np.sqrt(np.maximum(Rf ** 2 - (x - cx) ** 2, 0.0)) - 1e-4
    ux = np.zeros(N); uy = np.zeros(N); uz = -np.ones(N)

    Rsp = ((1.0 - n_surf) / (1.0 + n_surf)) ** 2
    w = np.full(N, 1.0 - Rsp)
    alive = np.ones(N, bool)
    F = np.zeros((nx, ny, nz))
    absorbed = 0.0
    abs_by_class = np.zeros(len(model["classes"]))    # deposited energy per tissue class
    esc = 0.0
    wth, m = 1e-4, 10

    for step in range(maxstep):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        lab0 = T.label_2d(x[idx], z[idx], ld)
        air = lab0 == 0
        if air.any():
            esc += w[idx][air].sum()
            alive[idx[air]] = False
        keep = ~air
        ins = idx[keep]
        if ins.size == 0:
            continue
        li = lab0[keep].astype(int) - 1
        mt = mua_l[li] + mus_l[li]
        s = -np.log(rng.uniform(0, 1, ins.size)) / mt
        x[ins] += s * ux[ins]; y[ins] += s * uy[ins]; z[ins] += s * uz[ins]
        inb = ~((x[ins] < 0) | (x[ins] >= nx * dx) | (y[ins] < 0) |
                (y[ins] >= ny * dx) | (z[ins] < 0) | (z[ins] >= nz * dx))
        if (~inb).any():
            esc += w[ins][~inb].sum()
            alive[ins[~inb]] = False
        ins = ins[inb]; s = s[inb]
        if ins.size == 0:
            continue
        lab1 = T.label_2d(x[ins], z[ins], ld)
        solid = lab1 >= 1
        ins = ins[solid]; s = s[solid]
        if ins.size == 0:
            continue
        li = lab1[solid].astype(int) - 1
        ma, ms, gg = mua_l[li], mus_l[li], g_l[li]
        mt = ma + ms
        xi = (x[ins] / dx).astype(int); yi = (y[ins] / dx).astype(int); zi = (z[ins] / dx).astype(int)
        np.add.at(F, (xi, yi, zi), w[ins] * s)          # track-length fluence
        dw = w[ins] * (ma / mt)
        absorbed += dw.sum()
        np.add.at(abs_by_class, li, dw)                 # deposition per tissue class
        w[ins] -= dw
        rg = rng.uniform(0, 1, ins.size)
        cost = np.where(np.abs(gg) > 1e-3,
                        (1 + gg ** 2 - ((1 - gg ** 2) / (1 - gg + 2 * gg * rg)) ** 2) / (2 * gg),
                        1 - 2 * rg)
        cost = np.clip(cost, -1, 1)
        sint = np.sqrt(1 - cost ** 2)
        phi = rng.uniform(0, 2 * np.pi, ins.size)
        cph, sph = np.cos(phi), np.sin(phi)
        uxx, uyy, uzz = ux[ins], uy[ins], uz[ins]
        near = np.abs(uzz) > 0.99999
        d = np.sqrt(np.maximum(1 - uzz ** 2, 1e-12))
        nux = np.where(near, sint * cph, sint * (uxx * uzz * cph - uyy * sph) / d + uxx * cost)
        nuy = np.where(near, sint * sph, sint * (uyy * uzz * cph + uxx * sph) / d + uyy * cost)
        nuz = np.where(near, np.sign(uzz) * cost, -d * sint * cph + uzz * cost)
        nr = np.sqrt(nux ** 2 + nuy ** 2 + nuz ** 2)
        ux[ins] = nux / nr; uy[ins] = nuy / nr; uz[ins] = nuz / nr
        low = ins[w[ins] < wth]
        if low.size:
            srv = rng.uniform(0, 1, low.size) < 1 / m
            w[low] = np.where(srv, w[low] * m, 0.0)
            alive[low[~srv]] = False

    return dict(F=F, absorbed=absorbed / N, escaped=esc / N, launched=1 - Rsp,
                steps=step, abs_by_class=abs_by_class / N,
                abs_pct_by_class=abs_by_class / max(absorbed, 1e-30))
