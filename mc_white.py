"""White Monte Carlo for source-detector reflectance (Tissue Optics Explorer).

A *white* (absorption-free) photon transport: photons propagate using scattering
only (mu_a = 0), and we record, for every photon remitted back through the top
skin surface, its surface distance from the source (the source-detector
separation, SDS) and the path length it spent in each tissue class. Then the
detected signal for ANY absorption configuration is recovered analytically by
Beer-Lambert reweighting:

    R(rho ; mu_a) = (1/N) * sum_{remitted in rho-bin} exp( -sum_class mu_a,class * L_class )

This is the standard white-MC scaling trick: trajectories depend only on
scattering, so a single run per (wavelength, scattering, geometry) lets us sweep
all *absorption* axes (melanin, SO2, blood volume) for free in post-processing.
Exact for melanin + SO2 (pure absorption); for blood volume it holds the small
scattering contribution of blood fixed (good approximation at skin blood
fractions). mu_s depends on wavelength, so one white run is needed per wavelength.

Geometry and classes match mc_core / tov_common exactly. Simplified transport
for intuition, not validated dosimetry.
"""
from __future__ import annotations
import numpy as np
import tov_common as T


def simulate_white(model: dict, nphoton: int = 400_000, seed: int = 7,
                   maxstep: int = 12000, Lmax: float = 220.0) -> dict:
    """Scattering-only MC; return photons remitted through the top surface.

    Returns dict:
      rho       (M,)        surface distance source->exit (mm) for M remitted photons
      Lpath     (M, nclass) path length per tissue class (mm)
      classes   list        tissue-class order matching Lpath columns
      w0        float        (1 - specular reflectance): fraction entering tissue
      n_launched, n_remit, n_capped, steps
    """
    rng = np.random.default_rng(seed)
    ld = model.get("layer_depth")
    classes = model["classes"]
    nclass = len(classes)
    mus_l = np.array([model["props"][c]["mus"] for c in classes])
    g_l = np.array([model["props"][c]["g"] for c in classes])
    n_surf = model["props"]["epidermis"]["n"]
    dx, nx, ny, nz = T.DX, T.NX, T.NY, T.NZ
    cx, cz, Rf = T.CX, T.CZ, T.RF
    N = int(nphoton)

    # emitter: small POINT-like spot (0.5 mm disk) on the dorsal (max-z) surface,
    # beam -z inward. Unlike mc_core's 4 mm line source (chosen for a clean
    # cross-section), a point source is required for SDS-resolved reflectance.
    spot = 0.5
    ang = rng.uniform(0, 2 * np.pi, N)
    rr = spot * np.sqrt(rng.uniform(0, 1, N))
    y_src = ny * dx / 2.0
    x = cx + rr * np.cos(ang)
    y = y_src + rr * np.sin(ang)                 # 2-D spot on the surface (not a line)
    # place each photon just inside the curved skin surface (the cylinder is not
    # flat across the spot; a flat launch plane at the apex starts most photons in air)
    z = cz + np.sqrt(np.maximum(Rf ** 2 - (x - cx) ** 2, 0.0)) - 1e-4
    ux = np.zeros(N); uy = np.zeros(N); uz = -np.ones(N)

    Rsp = ((1.0 - n_surf) / (1.0 + n_surf)) ** 2     # specular loss at entry
    alive = np.ones(N, bool)
    Lpath = np.zeros((N, nclass), np.float32)
    tot = np.zeros(N)                                 # total path length (mm)
    exit_rho = np.full(N, np.nan)
    remit = np.zeros(N, bool)
    n_capped = 0

    for step in range(maxstep):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        lab0 = T.label_2d(x[idx], z[idx], ld)
        air = lab0 == 0
        if air.any():
            ai = idx[air]
            dzp = z[ai] - cz; dxp = x[ai] - cx
            upper = dzp > 0                            # exited through the top arc
            theta = np.arctan2(dxp, dzp)              # 0 at apex, +x toward +SDS
            rho = np.hypot(Rf * theta, y[ai] - y_src)  # surface distance from source
            exit_rho[ai] = np.where(upper, rho, np.nan)
            remit[ai] = upper
            alive[ai] = False
        keep = ~air
        ins = idx[keep]
        if ins.size == 0:
            continue
        li = lab0[keep].astype(int) - 1
        mus = mus_l[li]                                # mu_t = mu_s (mu_a = 0)
        s = -np.log(rng.uniform(0, 1, ins.size)) / mus
        x[ins] += s * ux[ins]; y[ins] += s * uy[ins]; z[ins] += s * uz[ins]
        tot[ins] += s
        np.add.at(Lpath, (ins, li), s.astype(np.float32))   # path attributed to traversed medium
        # domain-edge escape (lateral / deep) -> not remitted
        oob = ((x[ins] < 0) | (x[ins] >= nx * dx) | (y[ins] < 0) |
               (y[ins] >= ny * dx) | (z[ins] < 0) | (z[ins] >= nz * dx))
        if oob.any():
            alive[ins[oob]] = False
        ins = ins[~oob]
        if ins.size == 0:
            continue
        capped = tot[ins] > Lmax
        if capped.any():
            n_capped += int(capped.sum())
            alive[ins[capped]] = False
        ins = ins[~capped]
        if ins.size == 0:
            continue
        # scatter (Henyey-Greenstein) at the new position; air-bound photons skip
        lab1 = T.label_2d(x[ins], z[ins], ld)
        solid = lab1 >= 1
        ins = ins[solid]
        if ins.size == 0:
            continue
        gg = g_l[lab1[solid].astype(int) - 1]
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

    mask = remit & np.isfinite(exit_rho)
    return dict(rho=exit_rho[mask].astype(np.float32),
                Lpath=Lpath[mask], classes=classes, w0=float(1.0 - Rsp),
                n_launched=N, n_remit=int(mask.sum()), n_capped=n_capped,
                steps=step)


if __name__ == "__main__":
    import time
    for wl in (850.0, 1450.0):
        m = T.get_model(wl, melanin=0.04, mus_scale=1.0)
        t0 = time.time()
        r = simulate_white(m, nphoton=200_000, seed=7)
        dt = time.time() - t0
        rho = r["rho"]
        print(f"\n--- {wl:.0f} nm --- {dt:.1f}s  remitted={r['n_remit']} "
              f"({100*r['n_remit']/r['n_launched']:.1f}%)  capped={r['n_capped']}  steps={r['steps']}")
        # nominal-absorption reflectance vs SDS (sanity: should fall off with rho)
        import optics_nir as O
        mua = np.array([O.tissue_optics(c, wl, melanin=0.04)["mua"] for c in r["classes"]])
        w = r["w0"] * np.exp(-(r["Lpath"].astype(np.float64) @ mua)) / r["n_launched"]
        edges = np.array([0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (rho >= a) & (rho < b)
            area = np.pi * (b ** 2 - a ** 2)
            Rd = w[sel].sum() / area
            print(f"  SDS {a:.1f}-{b:.1f} mm: R_area={Rd:.3e} /mm²   (n={sel.sum()})")
