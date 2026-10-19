"""Cycles render of an anatomically-layered finger with NIR light propagating
through it -- the Tissue Optics Explorer hero render.

Shows BOTH the tissue and the light:
  * a translucent skin envelope (the finger shape, see-through),
  * solid bone (phalanx) and blood vessels inside,
  * an opaque anatomical cross-section "chart" face (epidermis / dermis / plexus
    / fat / muscle / bone rings, in anatomical colours) at the near end,
  * the Monte-Carlo fluence as a glowing emissive voxel cloud (inferno on
    log-fluence) diffusing from a surface emitter -- the light "banana".

Geometry is rebuilt analytically from the JSON meta written by the engine, so it
exactly matches the voxel model the photons were transported through. Robust on
Blender 4.x with no OpenVDB dependency (emissive voxel-cloud route).

    BL=/Applications/Blender.app/Contents/MacOS/Blender
    $BL --background --python blender_tissue.py -- finger_850nm [samples] [view]   # PNG
    $BL            --python blender_tissue.py -- finger_850nm                       # interactive
view: 0 oblique (default) | 1 down-the-axis | 2 top | 3 other-oblique | 4 ALL
"""
import os
import sys
import json
import math
import numpy as np
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
NAME = argv[0] if len(argv) > 0 else "finger_850nm"
SAMPLES = int(argv[1]) if len(argv) > 1 else 96
VIEW = int(argv[2]) if len(argv) > 2 else 0

# ---- tunables ----
ENVELOPE_ALPHA = 0.16     # skin envelope transparency
ENVELOPE_EMIT = 0.30      # faint flesh self-glow so the silhouette reads
BONE_ALPHA = 0.95
VESSEL_ALPHA = 0.97
CHART_EMIT = 0.30         # cross-section chart self-illumination
CHART_ALPHA = 0.72        # chart opacity (reads as anatomy behind the dimmer glow)
GLOW_STRENGTH = 46.0      # fluence emission multiplier
GLOW_TRANSP = 0.32        # per-voxel transparency (higher = anatomy shows through dim glow)
GLOW_THRESH = 0.06        # log-fluence cutoff for a lit voxel
GLOW_BIN = 1             # downsample factor for the fluence cloud (speed/smoothness)
GLOW_DECADES = 3.0        # log dynamic range shown (more = deeper diffuse tail visible)
GLOW_FILL = 1.0           # voxel cube fill fraction (1.0 = touching, continuous)
GLOW_GAMMA = 1.6          # contrast on density (>1 dims the diffuse halo, sharpens core)

# ------------------------------------------------------------------ load
A = np.load(os.path.join(OUT, f"{NAME}.npy")).astype(np.float32)
meta = json.load(open(os.path.join(OUT, f"{NAME}.json")))
DX = float(meta["dx"]); RF = float(meta["Rf"]); CX = float(meta["cx"]); CZ = float(meta["cz"])
RBONE = float(meta["r_bone"]); LAYER_DEPTH = meta["layer_depth"]; VESSELS = meta["vessels"]
COLORS = meta["colors"]; WL = meta.get("wavelength", "?"); STATUS = meta.get("status", "illustrative")
NX, NY, NZ = A.shape
YLEN = NY * DX
# Cutaway at the source plane: the finger is cut where the light enters, so the
# anatomical cross-section IS the plane of the light banana. Tissue is kept for
# y >= YFACE; the near (cut) face carries the chart and the brightest fluence.
YFACE = NY * DX / 2.0

# layer radii (outer, inner) from depth-below-surface
RAD = []
prev = 0.0
for name, outer_depth in LAYER_DEPTH:
    r_out = RF - prev
    r_in = max(RF - min(outer_depth, RF), RBONE)
    RAD.append((name, r_out, r_in))
    prev = outer_depth


def srgb_lin(c):
    return tuple((v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in c[:3])


def col(name, a=1.0):
    r, g, b = srgb_lin(COLORS[name])
    return (r, g, b, a)


# ------------------------------------------------------------------ mesh helpers
def _add(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces); me.update()
    ob = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(ob)
    return ob


def tube(name, radius, y0, y1, segs=128):
    """Lateral cylinder surface (open tube) about the y axis."""
    th = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    xc = CX + radius * np.cos(th); zc = CZ + radius * np.sin(th)
    v0 = [(xc[i], y0, zc[i]) for i in range(segs)]
    v1 = [(xc[i], y1, zc[i]) for i in range(segs)]
    verts = v0 + v1
    faces = [[i, (i + 1) % segs, segs + (i + 1) % segs, segs + i] for i in range(segs)]
    return _add(name, verts, faces)


def solid_cyl(name, radius, y0, y1, cx=None, cz=None, segs=96):
    """Capped solid cylinder about y (for bone / vessels)."""
    cx = CX if cx is None else cx; cz = CZ if cz is None else cz
    th = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    xc = cx + radius * np.cos(th); zc = cz + radius * np.sin(th)
    v0 = [(xc[i], y0, zc[i]) for i in range(segs)]
    v1 = [(xc[i], y1, zc[i]) for i in range(segs)]
    c0 = len(v0) + len(v1); c1 = c0 + 1
    verts = v0 + v1 + [(cx, y0, cz), (cx, y1, cz)]
    faces = [[i, (i + 1) % segs, segs + (i + 1) % segs, segs + i] for i in range(segs)]
    faces += [[c0, (i + 1) % segs, i] for i in range(segs)]                 # cap y0
    faces += [[c1, segs + i, segs + (i + 1) % segs] for i in range(segs)]   # cap y1
    return _add(name, verts, faces)


def annulus(name, r_in, r_out, y, segs=128):
    """Flat ring in the x-z plane at axial position y (for the chart face)."""
    th = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    inner = [(CX + r_in * np.cos(t), y, CZ + r_in * np.sin(t)) for t in th]
    outer = [(CX + r_out * np.cos(t), y, CZ + r_out * np.sin(t)) for t in th]
    verts = inner + outer
    faces = [[i, (i + 1) % segs, segs + (i + 1) % segs, segs + i] for i in range(segs)]
    return _add(name, verts, faces)


def disc(name, radius, y, cx=None, cz=None, segs=96):
    cx = CX if cx is None else cx; cz = CZ if cz is None else cz
    th = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    ring = [(cx + radius * np.cos(t), y, cz + radius * np.sin(t)) for t in th]
    verts = ring + [(cx, y, cz)]; c = segs
    faces = [[c, i, (i + 1) % segs] for i in range(segs)]
    return _add(name, verts, faces)


# ------------------------------------------------------------------ materials
def mat_translucent(name, rgba, emit=0.0):
    """Flesh envelope: faint translucent film with a Fresnel-weighted self-glow
    so the silhouette reads on the dark background (like light through skin)."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Alpha"].default_value = rgba[3]
    bsdf.inputs["Roughness"].default_value = 0.4
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.25
    if emit and "Emission Color" in bsdf.inputs:
        fres = nt.nodes.new("ShaderNodeFresnel"); fres.inputs["IOR"].default_value = 1.4
        estr = nt.nodes.new("ShaderNodeMath"); estr.operation = "MULTIPLY"
        estr.inputs[1].default_value = emit
        nt.links.new(fres.outputs["Fac"], estr.inputs[0])
        nt.links.new(estr.outputs["Value"], bsdf.inputs["Emission Strength"])
        bsdf.inputs["Emission Color"].default_value = (rgba[0], rgba[1] * 0.5, rgba[2] * 0.4, 1)
    return m


def mat_solid(name, rgba, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Alpha"].default_value = rgba[3]
    bsdf.inputs["Roughness"].default_value = 0.5
    if emit and "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = emit
    return m


def assign(ob, mat):
    ob.data.materials.append(mat)


# ------------------------------------------------------------------ build scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene

# 1) translucent skin envelope (the finger shape)
env = tube("skin_envelope", RF, YFACE, YLEN)
assign(env, mat_translucent("m_env", col("epidermis", ENVELOPE_ALPHA), emit=ENVELOPE_EMIT))

# 2) bone (phalanx) + vessels, solid landmarks inside
bone = solid_cyl("bone", RBONE, YFACE, YLEN)
assign(bone, mat_solid("m_bone", col("bone", BONE_ALPHA), emit=0.04))
for k, (vx, vz, vr) in enumerate(VESSELS):
    ves = solid_cyl(f"vessel_{k}", vr, YFACE, YLEN, cx=vx, cz=vz)
    assign(ves, mat_solid(f"m_vessel_{k}", col("blood", VESSEL_ALPHA)))

# 3) anatomical cross-section chart at the cut face (translucent rings, recessed
#    slightly so the brightest fluence at the source plane reads in front of it)
YCHART = YFACE + 0.25
for name, r_out, r_in in RAD:
    ring = annulus(f"chart_{name}", r_in, r_out, YCHART)
    assign(ring, mat_solid(f"mc_{name}", col(name, CHART_ALPHA), emit=CHART_EMIT))
bdisc = disc("chart_bone", RBONE, YCHART)
assign(bdisc, mat_solid("mc_bone", col("bone", min(1.0, CHART_ALPHA + 0.35)), emit=CHART_EMIT))
for k, (vx, vz, vr) in enumerate(VESSELS):
    vd = disc(f"chart_vessel_{k}", vr, YCHART, cx=vx, cz=vz)
    assign(vd, mat_solid(f"mc_vessel_{k}", col("blood", min(1.0, CHART_ALPHA + 0.35)), emit=CHART_EMIT))

# 4) fluence glow -- emissive voxel cloud (binned, log-compressed, inferno)
b = GLOW_BIN
Ab = A[:NX // b * b, :NY // b * b, :NZ // b * b]
Ab = Ab.reshape(Ab.shape[0] // b, b, Ab.shape[1] // b, b, Ab.shape[2] // b, b).mean((1, 3, 5))
dxb = DX * b


def blur3(a, passes=2):
    for _ in range(passes):
        for ax in range(3):
            a = (a + np.roll(a, 1, ax) + np.roll(a, -1, ax)) / 3.0
    return a


Ab = blur3(Ab, 2)                              # smooth out MC voxel noise / blockiness
vmax = float(Ab.max()); floor = vmax / 10 ** GLOW_DECADES
lg = (np.log10(np.maximum(Ab, floor)) - math.log10(floor)) / (math.log10(vmax) - math.log10(floor))
lg = np.clip(lg, 0, 1).astype(np.float32) ** GLOW_GAMMA
ix, iy, iz = np.indices(Ab.shape)
xc = (ix + 0.5) * dxb; yc = (iy + 0.5) * dxb; zc = (iz + 0.5) * dxb
r = np.sqrt((xc - CX) ** 2 + (zc - CZ) ** 2)
inside = (r <= RF - 0.05) & (r >= RBONE + 0.05) & (yc >= YFACE)   # cutaway half
dens = np.where(inside, lg, 0.0)
mask = dens > GLOW_THRESH
P = np.stack([xc[mask], yc[mask], zc[mask]], 1).astype(np.float64)
D = dens[mask].astype(np.float32)
print(f"[blender] fluence cloud: {len(P)} lit voxels (bin {b}, {dxb:.2f} mm)")

h = dxb * GLOW_FILL
base = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                 [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64) * h
verts = (P[:, None, :] + base[None, :, :]).reshape(-1, 3)
quad = np.array([[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                 [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]], np.int64)
faces = (quad[None, :, :] + (np.arange(len(P)) * 8)[:, None, None]).reshape(-1, 4)
gmesh = bpy.data.meshes.new("fluence")
gmesh.from_pydata(verts.tolist(), [], faces.tolist()); gmesh.update()
rgba = np.repeat(D, 8).astype(np.float32)
cattr = np.zeros((len(verts), 4), np.float32); cattr[:, 0] = cattr[:, 1] = cattr[:, 2] = cattr[:, 3] = rgba
ca = gmesh.color_attributes.new("dcol", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", cattr.ravel())
gobj = bpy.data.objects.new("fluence", gmesh); bpy.context.collection.objects.link(gobj)

gm = bpy.data.materials.new("m_glow"); gm.use_nodes = True
nt = gm.node_tree; nt.nodes.clear()
attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_type = "GEOMETRY"; attr.attribute_name = "dcol"
sep = nt.nodes.new("ShaderNodeSeparateColor"); nt.links.new(attr.outputs["Color"], sep.inputs["Color"])
fac = sep.outputs["Red"]
# incandescent / fire ramp (dark-red -> orange -> yellow -> white): reads as
# light, not the purple low-end of inferno.
rmp = nt.nodes.new("ShaderNodeValToRGB"); cr = rmp.color_ramp
cr.elements[0].position = 0.0; cr.elements[0].color = (0.18, 0.005, 0.004, 1)
cr.elements[1].position = 1.0; cr.elements[1].color = (1.0, 0.99, 0.92, 1)
for pos, c in [(0.35, (0.78, 0.10, 0.02, 1)), (0.62, (0.98, 0.42, 0.06, 1)), (0.84, (1.0, 0.78, 0.28, 1))]:
    e = cr.elements.new(pos); e.color = c
nt.links.new(fac, rmp.inputs["Fac"])
em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(rmp.outputs["Color"], em.inputs["Color"])
es = nt.nodes.new("ShaderNodeMath"); es.operation = "MULTIPLY"; es.inputs[1].default_value = GLOW_STRENGTH
nt.links.new(fac, es.inputs[0]); nt.links.new(es.outputs["Value"], em.inputs["Strength"])
tr = nt.nodes.new("ShaderNodeBsdfTransparent"); mix = nt.nodes.new("ShaderNodeMixShader")
mix.inputs["Fac"].default_value = GLOW_TRANSP
nt.links.new(tr.outputs["BSDF"], mix.inputs[1]); nt.links.new(em.outputs["Emission"], mix.inputs[2])
mout = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(mix.outputs["Shader"], mout.inputs["Surface"])
gobj.data.materials.append(gm)

# 5) surface emitter marker + entry beam at the source point (top, max-z surface)
src = (CX, YFACE, CZ + RF)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.45, location=src)
emit = bpy.context.active_object
me = bpy.data.materials.new("m_emit"); me.use_nodes = True
eb = me.node_tree.nodes["Principled BSDF"]
eb.inputs["Emission Color"].default_value = (0.7, 0.85, 1.0, 1); eb.inputs["Emission Strength"].default_value = 30
emit.data.materials.append(me)
bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=1.4, location=(CX, src[1], CZ + RF - 0.7))
beam = bpy.context.active_object; beam.data.materials.append(me)

# ------------------------------------------------------------------ world / light / camera
world = bpy.data.worlds.new("w"); scn.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.006, 0.008, 0.015, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

key = bpy.data.objects.new("key", bpy.data.lights.new("key", "AREA"))
key.data.energy = 220; key.data.size = 14; key.location = (CX - 6, -6, CZ - 12)
scn.collection.objects.link(key)
rim = bpy.data.objects.new("rim", bpy.data.lights.new("rim", "AREA"))
rim.data.energy = 90; rim.data.size = 10; rim.location = (CX + 8, YLEN + 6, CZ + 6)
scn.collection.objects.link(rim)

CENTER = Vector((CX, YFACE + (YLEN - YFACE) * 0.35, CZ))
SPAN = max(2 * RF, YLEN - YFACE)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("c")); scn.collection.objects.link(cam)
aim = bpy.data.objects.new("aim", None); aim.location = CENTER; scn.collection.objects.link(aim)
cam.constraints.new("TRACK_TO").target = aim
scn.camera = cam
cam.data.lens = 46

# views: camera offset in (x,y,z) span-units, looking at CENTER (chart at -y end);
# +z is up and the emitter is on top, so view slightly from above (+z).
VIEWS = [(-0.80, -1.45, 0.75),    # oblique from the chart end, above
         (0.04, -1.85, 0.30),     # near head-on to the cross-section
         (0.0, 0.30, 1.95),       # top-down
         (1.05, -1.20, 0.55)]     # other oblique


def place(view):
    ox, oy, oz = VIEWS[view]
    cam.location = (CENTER.x + ox * SPAN, CENTER.y + oy * SPAN, CENTER.z + oz * SPAN)


# ------------------------------------------------------------------ render settings
scn.render.engine = "CYCLES"
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"; prefs.refresh_devices()
    for d in prefs.devices:
        d.use = True
    scn.cycles.device = "GPU"; print("[blender] Cycles device: GPU/Metal")
except Exception as e:
    print(f"[blender] GPU setup failed ({e}); CPU")
scn.cycles.samples = SAMPLES
scn.cycles.use_denoising = True
scn.cycles.transparent_max_bounces = 64
scn.cycles.max_bounces = 16
scn.render.resolution_x = 1500; scn.render.resolution_y = 1100
scn.render.film_transparent = False
scn.view_settings.view_transform = "AgX" if "AgX" in [v.name for v in
    (bpy.context.scene.view_settings.bl_rna.properties['view_transform'].enum_items)] else "Filmic"
scn.view_settings.look = "AgX - Medium High Contrast" if scn.view_settings.view_transform == "AgX" else "None"

if bpy.app.background:
    views = range(len(VIEWS)) if VIEW == 4 else [VIEW]
    for v in views:
        place(v)
        fn = f"{NAME}_tissue.png" if VIEW != 4 else f"{NAME}_tissue_v{v + 1}.png"
        scn.render.filepath = os.path.join(OUT, fn)
        bpy.ops.render.render(write_still=True)
        print("[blender] rendered ->", fn)
else:
    place(VIEW)
    bpy.context.preferences.inputs.use_mouse_emulate_3_button = True
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "RENDERED"
                        space.clip_end = max(space.clip_end, SPAN * 80)
                        if space.region_3d:
                            space.region_3d.view_location = CENTER
                            space.region_3d.view_distance = SPAN * 3.2
    print("[blender] interactive viewport ready - orbit to explore")
