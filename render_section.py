"""Labelled scientific cross-section companion to the Blender beauty render.

The Blender frames carry no baked-in text by design; this produces the matched
*annotated* figure: the anatomical layer map and the Monte Carlo log-fluence at
the source plane, with a tissue legend, depth scale and an explicit
"illustrative" label. Pure matplotlib. Needs outputs/finger_<wl>nm.{npy,json}
from mc_finger.py.

    python render_section.py finger_850nm
    python render_section.py compare 850 1550        # side-by-side depth contrast
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import tov_common as T

OUT = T.OUT


def slice_xz(name):
    A, meta = T.load_field(name)
    ny = A.shape[1] // 2
    return A[:, ny, :], meta


def label_xz():
    xi, zi = np.indices((T.NX, T.NZ))
    x = (xi + 0.5) * T.DX; z = (zi + 0.5) * T.DX
    return T.label_2d(x, z)


def anat_cmap():
    names = ["air"] + T.CLASSES
    return ListedColormap([T.COLORS[n] for n in names]), names


def draw_anatomy(ax):
    lab = label_xz().T                      # (z, x)
    cmap, names = anat_cmap()
    ax.imshow(lab, origin="lower", cmap=cmap, vmin=0, vmax=len(names) - 1,
              extent=[0, T.NX * T.DX, 0, T.NZ * T.DX], interpolation="nearest")
    ax.set_title("Anatomy (layer geometry)", fontsize=10)
    handles = [Patch(facecolor=T.COLORS[n], edgecolor="k", lw=0.3,
                     label=n.capitalize()) for n in T.CLASSES]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, framealpha=0.85,
              ncol=1, bbox_to_anchor=(1.01, 1.0))


def draw_fluence(ax, A, meta, title):
    flo = A.max() / 1e4
    lg = np.log10(np.maximum(A.T, flo))
    im = ax.imshow(lg, origin="lower", cmap="inferno",
                   extent=[0, T.NX * T.DX, 0, T.NZ * T.DX], interpolation="bilinear")
    # tissue outline + bone for reference
    lab = label_xz().T
    ax.contour(np.linspace(0, T.NX * T.DX, T.NX), np.linspace(0, T.NZ * T.DX, T.NZ),
               (lab > 0).astype(float), levels=[0.5], colors="w", linewidths=0.6, alpha=0.5)
    ax.contour(np.linspace(0, T.NX * T.DX, T.NX), np.linspace(0, T.NZ * T.DX, T.NZ),
               (lab == T.LABEL["bone"]).astype(float), levels=[0.5], colors="w",
               linewidths=0.6, alpha=0.5)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log$_{10}$ fluence (rel.)", fontsize=8)


def scalebar(ax):
    x0 = T.NX * T.DX - 6.0
    ax.plot([x0, x0 + 5], [1.0, 1.0], "w-", lw=2.5)
    ax.text(x0 + 2.5, 1.4, "5 mm", color="w", ha="center", fontsize=8)


def one(name):
    A, meta = slice_xz(name)
    wl = meta.get("wavelength", "?")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    draw_anatomy(axes[0]); scalebar(axes[0])
    draw_fluence(axes[1], A, meta, f"NIR fluence @ {wl:.0f} nm  (emitter on top)")
    scalebar(axes[1])
    for ax in axes:
        ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("z, depth axis (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Finger cross-section, {meta['model']}  --  illustrative "
                 f"(simplified transport for visualisation; not validated dosimetry)",
                 fontsize=10, y=0.99)
    p = OUT / f"{name}_section.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(p, dpi=140); plt.close(fig)
    print("saved", p)


def compare(wls):
    fig, axes = plt.subplots(1, len(wls), figsize=(5.2 * len(wls), 5.2))
    if len(wls) == 1:
        axes = [axes]
    for ax, wl in zip(axes, wls):
        A, meta = slice_xz(f"finger_{wl}nm")
        draw_fluence(ax, A, meta, f"{wl} nm")
        scalebar(ax)
        ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("z, depth (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Depth of NIR penetration vs wavelength (finger; emitter on top)  --  illustrative",
                 fontsize=11, y=0.99)
    p = OUT / "finger_depth_compare.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(p, dpi=140); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare([int(w) for w in sys.argv[2:]] or [850, 1550])
    else:
        one(sys.argv[1] if len(sys.argv) > 1 else "finger_850nm")
