"""Literature-based near-infrared optical properties for the Tissue Optics Explorer.

Computes per-tissue (mua, mus, g, n) at an arbitrary wavelength by mixing
chromophore spectra with tissue composition fractions.

Sources (the two spectral tables ship in ../data, see data/README.md):
  - Chromophore absorption vs wavelength: Roblyer et al. 2025 SWIR review,
    stitched extinction dataset (oxy/deoxy-Hb in mm^-1/uM, water + lipid in
    mm^-1), 430-2000 nm, extended to 2500 nm (water from Segelstein 1981, a
    seamless join; lipid + Hb held flat past 2000 nm -- see _load_chromophores).
  - Tissue composition (water / lipid / blood / melanosome volume fractions):
    seeded from Jacques 2013, "Optical properties of biological tissues: a review".
  - Reduced scattering mus'(lambda) = a' (lambda/500nm)^-b: Jacques 2013
    power-law coefficients (skin / dermis / epidermis / fat / muscle / bone).
  - Melanosome interior absorption: Jacques 2013 power law 519 (l/500)^-3 cm^-1.

Absorption model (Beer-Lambert mixing, all mua in 1/mm):
    mua(l) = Wf*mua_water(l) + Lf*mua_lipid(l)
           + Bf*[S*eps_oxy(l) + (1-S)*eps_deoxy(l)]*C_HB
           + Mf*mua_melanosome(l) + baseline
where C_HB = 2330 uM is the haemoglobin concentration of whole blood
(150 g/L / 64458 g/mol). Sanity-checked in __main__ against Bosschaart 2014 whole
blood and measured human-skin ranges (Troy 2001 / Bashkatov 2005).

This is a *literature-order* forward model for visualisation and intuition. It is
not validated for quantitative dosimetry.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
_ROBLYER = DATA / "roblyer_2025_stitched_extinction_dataset1.csv"
# Water tail (2000-2500 nm) for the SWIR slider. Roblyer's own water column is
# Segelstein 1981 resampled (median difference ~0% over 600-2000 nm; identical at
# the 2000 nm seam), so appending Segelstein extends water continuously to 2500 nm.
_SEGELSTEIN = DATA / "segelstein_1981_water_absorption.csv"
_EXT_MAX_NM = 2500.0

C_HB_WHOLE_BLOOD_UM = 2330.0     # uM haemoglobin in whole blood (150 g/L / 64458)


# --------------------------------------------------------------- chromophores
def _load_chromophores():
    """Return wl[nm] (430-2500) and per-uM (Hb) / per-mm (water,lipid) spectra.

    Roblyer 430-2000 nm verbatim, then 2001-2500 nm appended at 1 nm spacing:
    water from Segelstein 1981 (seamless, same source as Roblyer's water column),
    lipid and oxy/deoxy-Hb held flat at their 2000 nm value. Beyond ~2000 nm the
    bundled tables do not cover lipid or Hb; both are minor in the SWIR (water
    absorption ~3-10 mm^-1 swamps lipid ~0.2 mm^-1 and the near-zero Hb tail), so
    holding them flat is a documented approximation, not measured SWIR structure.
    """
    d = np.genfromtxt(_ROBLYER, delimiter=",", names=True)
    wl = d["wavelength_nm"].astype(float)
    oxy = d["oxy_extinction_mm_inv_per_uM"].astype(float)
    deoxy = d["deoxy_extinction_mm_inv_per_uM"].astype(float)
    water = d["water_mu_a_mm_inv"].astype(float)
    lipid = d["lipid_mu_a_mm_inv"].astype(float)

    wl_ext = np.arange(wl[-1] + 1.0, _EXT_MAX_NM + 0.5)             # 2001..2500 nm
    sg = np.genfromtxt(_SEGELSTEIN, delimiter=",", names=True)
    o = np.argsort(sg["wavelength_nm"])
    water_ext = np.interp(wl_ext, sg["wavelength_nm"][o], sg["mu_a_mm_inv"][o])
    flat = np.ones_like(wl_ext)
    return np.concatenate([wl, wl_ext]), {
        "oxy":   np.concatenate([oxy,   oxy[-1]   * flat]),    # mm^-1 / uM (flat >2000)
        "deoxy": np.concatenate([deoxy, deoxy[-1] * flat]),    # mm^-1 / uM (flat >2000)
        "water": np.concatenate([water, water_ext]),           # mm^-1 (Segelstein >2000)
        "lipid": np.concatenate([lipid, lipid[-1] * flat]),    # mm^-1 (flat >2000)
    }


_WL, _CHROMO = _load_chromophores()


def _interp(name: str, lam: float) -> float:
    return float(np.interp(lam, _WL, _CHROMO[name]))


def mua_water(lam):  return _interp("water", lam)
def mua_lipid(lam):  return _interp("lipid", lam)
def eps_oxy(lam):    return _interp("oxy", lam)      # mm^-1 / uM
def eps_deoxy(lam):  return _interp("deoxy", lam)    # mm^-1 / uM


def mua_blood(lam: float, s: float = 0.97) -> float:
    """Whole-blood absorption (1/mm) at oxygen saturation s."""
    return C_HB_WHOLE_BLOOD_UM * (s * eps_oxy(lam) + (1 - s) * eps_deoxy(lam))


def mua_melanosome(lam: float) -> float:
    """Melanosome interior absorption (1/mm), Jacques 2013 power law."""
    return 51.9 * (lam / 500.0) ** -3.0          # 519 cm^-1 -> 51.9 mm^-1


# --------------------------------------------------------- tissue definitions
# Composition volume fractions + scattering power law per tissue class.
#   Wf,Lf,Bf,Mf = water, lipid, blood, melanosome volume fractions
#   S           = blood oxygen saturation
#   a_mm, b     = reduced scattering mus'(l)=a_mm*(l/500)^-b   [a_mm in 1/mm]
#   g, n        = anisotropy, refractive index
# Composition seeded from Jacques 2013 (skin composition); scattering a', b from
# the Jacques 2013 power-law fits (skin/dermis/epidermis/fat/muscle/bone).
# Plexus/vessel are blood-dominated variants.
TISSUES = {
    #                  Wf     Lf     Bf      Mf     S     a_mm    b      g     n
    "epidermis": dict(Wf=0.20, Lf=0.00, Bf=0.000, Mf=0.020, S=0.97, a_mm=6.6, b=1.42, g=0.90, n=1.40),
    "dermis":    dict(Wf=0.65, Lf=0.00, Bf=0.004, Mf=0.000, S=0.70, a_mm=4.4, b=1.42, g=0.90, n=1.38),
    "plexus":    dict(Wf=0.60, Lf=0.00, Bf=0.050, Mf=0.000, S=0.70, a_mm=4.4, b=1.42, g=0.90, n=1.38),
    "fat":       dict(Wf=0.20, Lf=0.60, Bf=0.005, Mf=0.000, S=0.80, a_mm=1.5, b=0.68, g=0.90, n=1.44),
    "muscle":    dict(Wf=0.75, Lf=0.02, Bf=0.030, Mf=0.000, S=0.65, a_mm=1.3, b=0.93, g=0.90, n=1.40),
    "bone":      dict(Wf=0.12, Lf=0.10, Bf=0.020, Mf=0.000, S=0.70, a_mm=2.2, b=0.66, g=0.90, n=1.55),
    "blood":     dict(Wf=0.83, Lf=0.00, Bf=1.000, Mf=0.000, S=0.85, a_mm=22.0, b=0.66, g=0.98, n=1.40),
}

BASELINE_MUA = 0.002    # 1/mm, residual background absorption


def tissue_optics(name: str, lam: float, melanin=None, mus_scale: float = 1.0,
                  so2=None, blood_scale: float = 1.0) -> dict:
    """(mua, mus, musp, g, n) in 1/mm for `name` at wavelength `lam` (nm).

    melanin     -- override the epidermal melanosome volume fraction (skin tone);
                   None keeps the tissue default. Typical: 0.02 light, 0.10 medium,
                   0.30 dark (Jacques 2013 ranges).
    mus_scale   -- multiply reduced scattering mus' (tissue scattering knob).
    so2         -- override blood oxygen saturation (0..1) for *all* tissues;
                   None keeps each tissue's default S. Pure-absorption knob.
    blood_scale -- multiply every tissue's blood volume fraction Bf (blood-volume
                   knob). Treated as absorption-only: mus' is NOT changed, so the
                   white-MC reweight (fixed scattering, varied mua) stays valid for
                   the small blood fractions in skin. Validity note in module docs.
    """
    t = TISSUES[name]
    mf = t["Mf"] if melanin is None or name != "epidermis" else float(melanin)
    s = t["S"] if so2 is None else float(so2)
    bf = t["Bf"] * float(blood_scale)
    mua = (t["Wf"] * mua_water(lam)
           + t["Lf"] * mua_lipid(lam)
           + bf * mua_blood(lam, s)
           + mf * mua_melanosome(lam)
           + BASELINE_MUA)
    musp = t["a_mm"] * (lam / 500.0) ** (-t["b"]) * float(mus_scale)
    mus = musp / (1.0 - t["g"])
    return dict(mua=float(mua), mus=float(mus), musp=float(musp), g=t["g"], n=t["n"])


if __name__ == "__main__":
    print(f"Chromophore spectra: {_WL.min():.0f}-{_WL.max():.0f} nm, {_WL.size} pts\n")
    # sanity: SWIR water structure -- should peak ~1450 & ~1940 nm, dip ~2200, rise to 2500
    print("water mua(/mm):  " + "  ".join(
        f"{lam}:{mua_water(lam):.2f}" for lam in (500, 970, 1450, 1940, 2000, 2200, 2300, 2500)))
    # sanity: whole-blood mua vs Bosschaart 2014 (oxygenated, ~0.1-0.5 /mm in NIR)
    for lam in (660, 800, 850, 940, 1300, 1550, 2000, 2300):
        print(f"  whole blood mua({lam}nm, S=0.97) = {mua_blood(lam, 0.97):.3f} /mm")
    print()
    for lam in (660, 850, 1300, 1550, 2200, 2500):
        print(f"--- {lam} nm ---")
        for name in TISSUES:
            o = tissue_optics(name, lam)
            print(f"  {name:9s} mua={o['mua']:.4f}  mus'={o['musp']:.2f}  "
                  f"mus={o['mus']:.1f}  g={o['g']}  n={o['n']}")
        print()
