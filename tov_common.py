"""Shared tissue model + I/O for the Tissue Optics Explorer (multi-site).

Single source of truth so the portable CPU Monte Carlo (mc_core), the white MC
(mc_white) and the Blender renderer simulate the *same* tissue. All lengths are
millimetres; optical coefficients are 1/mm.

Geometry is a layered cylinder (axis = y). Two anatomical **sites**, selected with
`set_site(name)` before building:

  - **finger**: small cylinder, RF = 8 mm, centred in the domain; bone is a
    central core (r < R_BONE). Layers epidermis / plexus / dermis / fat / muscle.
  - **wrist** (forearm/wrist): large-radius cylinder, RF = 30 mm, centre placed
    *below* a shallow domain so the top surface is gently curved (slab-like, which
    is faithful for a large site). Bone is the deepest *layer*, not a core. Layers
    epidermis / plexus / dermis / fat / muscle / bone (thinner epidermis, thicker
    subcutis than the finger).

Blood is folded into the dermal/muscle layers (the plexus sublayer carries the
elevated blood-volume fraction); standalone vessels were removed (VESSELS = []).
Optical properties come from optics_nir.tissue_optics() (literature chromophore
mixing), wavelength-tunable. Simplified transport for intuition, not dosimetry.

`set_site` mutates module globals (DX, NX, NY, NZ, CX, CZ, RF, R_BONE,
LAYER_DEFAULT) so mc_core / mc_white / explorer_build read the active site at call
time. Defaults to 'finger' on import (backward compatible).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import optics_nir as O

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# ---- ordered tissue classes (label index = position + 1; 0 = air) ----
CLASSES = ["epidermis", "dermis", "plexus", "fat", "muscle", "bone", "blood"]
LABEL = {name: i + 1 for i, name in enumerate(CLASSES)}

# anatomical render colours (sRGB 0..1) -- single source of truth for Blender + viewer
COLORS = {
    "air":       (0.00, 0.00, 0.00),
    "epidermis": (0.87, 0.66, 0.49),   # tan
    "dermis":    (0.83, 0.40, 0.39),   # pink-red (vascular)
    "plexus":    (0.74, 0.16, 0.20),   # deeper red (blood-rich)
    "fat":       (0.96, 0.86, 0.52),   # adipose yellow
    "muscle":    (0.66, 0.24, 0.24),   # muscle red-brown
    "bone":      (0.94, 0.91, 0.83),   # ivory
    "blood":     (0.55, 0.04, 0.08),   # dark venous red
}

# ---- site catalogue ----------------------------------------------------------
# layers: ordered (name, thickness_um); thickness None => deepest layer (fills in).
# epidermis_um / dermis_um are the slider-tunable thicknesses; others fixed.
SITES = {
    "finger": dict(
        label="Finger", dx=0.1, nx=180, ny=120, nz=180, rf=8.0, r_bone=3.5,
        cx=None, cz=None,            # None => domain centre (finger fills domain)
        bone_core=True,              # bone is a central core (r < r_bone)
        layers=[("epidermis", 150.0), ("plexus", 150.0), ("dermis", 500.0),
                ("fat", 1500.0), ("muscle", None)],
        dermis_grid=[400.0, 650.0, 950.0, 1350.0],
    ),
    "wrist": dict(
        label="Forearm / wrist", dx=0.1, nx=300, ny=120, nz=160, rf=30.0, r_bone=0.0,
        cx=None, cz=-15.0,           # centre 15 mm below the domain -> apex at z=15 mm
        bone_core=False,             # bone is the deepest *layer*
        layers=[("epidermis", 80.0), ("plexus", 120.0), ("dermis", 1300.0),
                ("fat", 4000.0), ("muscle", 7000.0), ("bone", None)],
        dermis_grid=[900.0, 1300.0, 1800.0, 2400.0],
    ),
}

# module globals populated by set_site() ---------------------------------------
SITE = "finger"
DX = NX = NY = NZ = CX = CZ = RF = R_BONE = None
BONE_CORE = True
LAYER_DEFAULT: list = []            # default (name, outer_depth_mm) for the site
_SITE_CFG: dict = {}


def set_site(name: str = "finger") -> dict:
    """Configure module geometry globals for `name` ('finger' | 'wrist')."""
    global SITE, DX, NX, NY, NZ, CX, CZ, RF, R_BONE, BONE_CORE, LAYER_DEFAULT, _SITE_CFG
    cfg = SITES[name]
    SITE = name
    _SITE_CFG = cfg
    DX = cfg["dx"]; NX = cfg["nx"]; NY = cfg["ny"]; NZ = cfg["nz"]
    RF = cfg["rf"]; R_BONE = cfg["r_bone"]; BONE_CORE = cfg["bone_core"]
    CX = NX * DX / 2.0 if cfg["cx"] is None else cfg["cx"]
    CZ = NZ * DX / 2.0 if cfg["cz"] is None else cfg["cz"]
    LAYER_DEFAULT = layer_depths()
    return cfg


def layer_depths(epidermis_um: float | None = None, dermis_um: float | None = None) -> list:
    """Outer depth (mm below surface) per layer, from the active site's template.

    epidermis_um / dermis_um override those layers (slider knobs); the deepest
    layer (thickness None) extends inward (to the bone core / domain).
    """
    thick = {nm: t for nm, t in _SITE_CFG["layers"]}
    if epidermis_um is not None and "epidermis" in thick:
        thick["epidermis"] = float(epidermis_um)
    if dermis_um is not None and "dermis" in thick:
        thick["dermis"] = float(dermis_um)
    out, acc = [], 0.0
    for nm, _ in _SITE_CFG["layers"]:
        t = thick[nm]
        if t is None:
            out.append((nm, 1e9)); break
        acc += t / 1000.0
        out.append((nm, acc))
    return out


def geom() -> dict:
    """Geometry parameters, JSON-serialisable, for engines + renderer + viewer."""
    return dict(dx=DX, shape=[NX, NY, NZ], Rf=RF, cx=CX, cz=CZ,
                r_bone=R_BONE, bone_core=BONE_CORE, layer_depth=LAYER_DEFAULT,
                vessels=[], classes=CLASSES, label=LABEL, colors=COLORS,
                site=SITE, site_label=_SITE_CFG.get("label", SITE),
                layers=_SITE_CFG["layers"])


def get_model(wavelength_nm: float = 850.0, melanin: float = 0.02,
              mus_scale: float = 1.0, epidermis_um: float | None = None,
              dermis_um: float | None = None, so2=None, blood_scale: float = 1.0) -> dict:
    """Tissue model: per-class optical properties + (parameterised) layer geometry."""
    lam = float(wavelength_nm)
    props = {name: O.tissue_optics(name, lam, melanin=melanin, mus_scale=mus_scale,
                                   so2=so2, blood_scale=blood_scale)
             for name in CLASSES}
    ld = layer_depths(epidermis_um, dermis_um)
    g = geom(); g["layer_depth"] = ld
    return dict(label=f"{lam:.0f} nm ({SITE}, literature chromophore mix)",
                wavelength=lam, melanin=melanin, mus_scale=mus_scale, site=SITE,
                epidermis_um=epidermis_um, dermis_um=dermis_um,
                classes=CLASSES, props=props, layer_depth=ld, geom=g)


def prop_table(model: dict) -> np.ndarray:
    """MCX-style property table [mua, mus, g, n]: row 0 = air, then one row per class."""
    rows = [[0.0, 0.0, 1.0, 1.0]]
    for name in model["classes"]:
        p = model["props"][name]
        rows.append([p["mua"], p["mus"], p["g"], p["n"]])
    return np.asarray(rows, dtype=np.float32)


def label_2d(x_mm, z_mm, layer_depth: list | None = None) -> np.ndarray:
    """Tissue-label map in the (x, z) plane (cylinder along y)."""
    layer_depth = LAYER_DEFAULT if layer_depth is None else layer_depth
    x, z = np.broadcast_arrays(np.asarray(x_mm, float), np.asarray(z_mm, float))
    r = np.sqrt((x - CX) ** 2 + (z - CZ) ** 2)
    depth = RF - r
    lab = np.zeros(x.shape, dtype=np.uint8)
    inside = r <= RF
    prev = 0.0
    for name, outer in layer_depth:
        lab[inside & (depth >= prev) & (depth < outer)] = LABEL[name]
        prev = outer
    if BONE_CORE and R_BONE > 0:
        lab[r <= R_BONE] = LABEL["bone"]
    return lab


def build_label_volume(model: dict | None = None) -> np.ndarray:
    """(NX, NY, NZ) uint8 voxel label volume."""
    ld = None if model is None else model.get("layer_depth")
    xi, _, zi = np.indices((NX, NY, NZ))
    x = (xi + 0.5) * DX
    z = (zi + 0.5) * DX
    return np.ascontiguousarray(label_2d(x, z, ld).astype(np.uint8))


def source_voxel() -> list:
    """Emitter on the dorsal (top, max-z) surface, beam -z inward."""
    return [CX / DX, NY / 2.0, (CZ + RF) / DX - 1.0]


def save_field(name: str, field, meta: dict) -> Path:
    field = np.ascontiguousarray(np.asarray(field, dtype=np.float32))
    np.save(OUT / f"{name}.npy", field)
    out = dict(meta)
    out.update(geom())
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=2))
    return OUT / f"{name}.npy"


def load_field(name: str):
    field = np.load(OUT / f"{name}.npy")
    meta = json.loads((OUT / f"{name}.json").read_text())
    return field, meta


set_site("finger")     # default; backward compatible


if __name__ == "__main__":
    for s in ("finger", "wrist"):
        set_site(s)
        m = get_model(850)
        vol = build_label_volume(m)
        counts = {name: int((vol == LABEL[name]).sum()) for name in CLASSES}
        present = {k: v for k, v in counts.items() if v}
        print(f"[{s}] domain {NX}x{NY}x{NZ} dx={DX} RF={RF} CX={CX} CZ={CZ} "
              f"r_bone={R_BONE} bone_core={BONE_CORE}")
        print(f"     layers(mm): {LAYER_DEFAULT}")
        print(f"     voxel counts: {present}\n")
    set_site("finger")
