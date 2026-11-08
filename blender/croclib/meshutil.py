"""Low-level mesh construction helpers.

Everything here is plain geometry maths that happens to hand its result to
Blender at the end. Cross-sections are superellipses, which is the one knob
that matters for this character: a circle reads as a hose, a hard rectangle
reads as a crate, and the ~0.65 exponent in between reads as a muscled limb or
a crocodile's boxy snout.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import bpy
import bmesh
from mathutils import Matrix, Vector

Vec3 = Sequence[float]


# --------------------------------------------------------------------------
# cross-sections
# --------------------------------------------------------------------------

def superellipse(n: int, rx: float, ry: float, power: float = 1.0) -> list[tuple[float, float]]:
    """`n` points around |x/rx|^p + |y/ry|^p = 1, counter-clockwise.

    power 1.0 is an ellipse, lower is boxier (0.5 is a squircle-ish), higher is
    a pinched diamond. Signed powers keep the four quadrants symmetric.
    """
    pts = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        ca, sa = math.cos(a), math.sin(a)
        x = math.copysign(abs(ca) ** power, ca) * rx
        y = math.copysign(abs(sa) ** power, sa) * ry
        pts.append((x, y))
    return pts


def ring(
    centre: Vec3,
    rx: float,
    ry: float,
    n: int,
    power: float = 1.0,
    *,
    normal: Vec3 = (0.0, 0.0, 1.0),
    up: Vec3 | None = None,
    shear_y: float = 0.0,
) -> list[Vector]:
    """A closed loop of `n` points on the plane through `centre` facing `normal`.

    `shear_y` leans the ring's local +y out of plane, which is how a belly
    sags or a jaw slopes without needing a second control point.
    """
    nrm = Vector(normal).normalized()
    if up is None:
        up = (0.0, 0.0, 1.0) if abs(nrm.z) < 0.9 else (0.0, 1.0, 0.0)
    u = Vector(up)
    side = u.cross(nrm)
    if side.length < 1e-6:
        side = Vector((1.0, 0.0, 0.0)).cross(nrm)
    side.normalize()
    upv = nrm.cross(side).normalized()

    c = Vector(centre)
    out = []
    for x, y in superellipse(n, rx, ry, power):
        p = c + side * x + upv * y + nrm * (y * shear_y)
        out.append(p)
    return out


# --------------------------------------------------------------------------
# lofting
# --------------------------------------------------------------------------

def loft(rings: Sequence[Sequence[Vector]], cap_start: bool = True, cap_end: bool = True):
    """Bridge equal-length rings in order into (verts, faces).

    Rings must all have the same point count and consistent winding, which is
    what every generator in this module produces.
    """
    rings = [list(r) for r in rings]
    n = len(rings[0])
    for r in rings:
        if len(r) != n:
            raise ValueError("loft(): rings differ in length")

    verts: list[Vector] = []
    for r in rings:
        verts.extend(r)

    faces: list[tuple[int, ...]] = []
    for s in range(len(rings) - 1):
        a = s * n
        b = (s + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append((a + i, a + j, b + j, b + i))

    if cap_start:
        centre = sum(rings[0], Vector()) / n
        ci = len(verts)
        verts.append(centre)
        for i in range(n):
            j = (i + 1) % n
            faces.append((ci, j, i))
    if cap_end:
        last = (len(rings) - 1) * n
        centre = sum(rings[-1], Vector()) / n
        ci = len(verts)
        verts.append(centre)
        for i in range(n):
            j = (i + 1) % n
            faces.append((ci, last + i, last + j))
    return verts, faces


def frames_along(points: Sequence[Vector]) -> list[Vector]:
    """Unit tangents at each point of a polyline, averaged at the interior."""
    pts = [Vector(p) for p in points]
    tans = []
    for i, p in enumerate(pts):
        if i == 0:
            t = pts[1] - p
        elif i == len(pts) - 1:
            t = p - pts[-2]
        else:
            t = (pts[i + 1] - pts[i - 1])
        if t.length < 1e-9:
            t = Vector((0.0, 0.0, 1.0))
        tans.append(t.normalized())
    return tans


def tube(
    points: Sequence[Vec3],
    radii: Sequence[tuple[float, float]],
    n: int = 16,
    power: float = 1.0,
    *,
    cap_start: bool = True,
    cap_end: bool = True,
    powers: Sequence[float] | None = None,
    roll_up: Vec3 = (0.0, 0.0, 1.0),
):
    """Sweep a superellipse along a polyline using parallel transport.

    Parallel transport rather than a fixed up-vector: a tail that curves
    through the vertical would spin its own cross-section if every ring were
    built against world up, and the twist shows as a corkscrew in the scales.
    """
    pts = [Vector(p) for p in points]
    tans = frames_along(pts)

    # Carry one reference direction along the curve, re-orthogonalised each step.
    ref = Vector(roll_up)
    if abs(ref.dot(tans[0])) > 0.95:
        ref = Vector((0.0, 1.0, 0.0))
    ref = (ref - tans[0] * ref.dot(tans[0])).normalized()

    rings = []
    for i, (p, t) in enumerate(zip(pts, tans)):
        if i:
            ref = (ref - t * ref.dot(t))
            if ref.length < 1e-6:
                ref = Vector((0.0, 0.0, 1.0)).cross(t).normalized()
            ref.normalize()
        rx, ry = radii[i]
        pw = power if powers is None else powers[i]
        rings.append(ring(p, rx, ry, n, pw, normal=t, up=ref))
    return loft(rings, cap_start, cap_end)


# --------------------------------------------------------------------------
# Blender object plumbing
# --------------------------------------------------------------------------

def new_object(name: str, verts, faces, collection: bpy.types.Collection | None = None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def shade_smooth(obj: bpy.types.Object, angle_deg: float = 40.0) -> None:
    """Smooth shading with creases kept past `angle_deg`.

    Blender moved "smooth by angle" out of the modifier list and into a node
    asset in 4.1, so there is no modifier type to add any more. The operator
    exists but needs the object selected and active; Edge Split is the fallback
    that works on every build and bakes the same split normals into the export.
    """
    for poly in obj.data.polygons:
        poly.use_smooth = True
    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle_deg))
    except (AttributeError, RuntimeError):
        mod = obj.modifiers.new("SharpEdges", "EDGE_SPLIT")
        mod.split_angle = math.radians(angle_deg)
        mod.use_edge_sharp = False


def add_subsurf(obj: bpy.types.Object, levels: int = 1, render: int | None = None):
    mod = obj.modifiers.new("Subsurf", "SUBSURF")
    mod.levels = levels
    mod.render_levels = levels if render is None else render
    mod.use_limit_surface = True
    return mod


def mirror_x(obj: bpy.types.Object, merge: float = 0.0005):
    mod = obj.modifiers.new("Mirror", "MIRROR")
    mod.use_axis = (True, False, False)
    mod.use_clip = True
    mod.merge_threshold = merge
    return mod


def apply_modifiers(obj: bpy.types.Object) -> None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated)
    obj.modifiers.clear()
    old = obj.data
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)


def join(objects: Sequence[bpy.types.Object], name: str) -> bpy.types.Object:
    """Merge objects into the first one, preserving each one's materials."""
    objects = [o for o in objects if o is not None]
    if not objects:
        raise ValueError("join(): nothing to join")
    target = objects[0]
    target.name = target.data.name = name
    if len(objects) == 1:
        return target

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.join()
    return target


def set_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def weld(obj: bpy.types.Object, distance: float = 0.001) -> None:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def transform(obj: bpy.types.Object, matrix: Matrix) -> None:
    obj.data.transform(matrix)
    obj.data.update()


# --------------------------------------------------------------------------
# small solids used all over the costume and the claws
# --------------------------------------------------------------------------

def cone(base: Vec3, tip: Vec3, radius: float, n: int = 10, *, radius2: float = 0.0,
         up: Vec3 = (0.0, 0.0, 1.0)):
    """A tapered spike from `base` to `tip`; `radius2` > 0 makes it a frustum."""
    b, t = Vector(base), Vector(tip)
    axis = (t - b)
    if axis.length < 1e-9:
        raise ValueError("cone(): zero length")
    axis = axis.normalized()
    ref = Vector(up)
    if abs(ref.dot(axis)) > 0.95:
        ref = Vector((1.0, 0.0, 0.0))
    ref = (ref - axis * ref.dot(axis)).normalized()
    r0 = ring(b, radius, radius, n, 1.0, normal=axis, up=ref)
    if radius2 > 0.0:
        r1 = ring(t, radius2, radius2, n, 1.0, normal=axis, up=ref)
        return loft([r0, r1], True, True)
    verts = list(r0)
    centre = sum(r0, Vector()) / n
    verts.append(t)
    verts.append(centre)
    ti, ci = n, n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, ti))
        faces.append((ci, j, i))
    return verts, faces


def box(centre: Vec3, size: Vec3):
    cx, cy, cz = centre
    sx, sy, sz = (s * 0.5 for s in size)
    verts = [
        Vector((cx - sx, cy - sy, cz - sz)), Vector((cx + sx, cy - sy, cz - sz)),
        Vector((cx + sx, cy + sy, cz - sz)), Vector((cx - sx, cy + sy, cz - sz)),
        Vector((cx - sx, cy - sy, cz + sz)), Vector((cx + sx, cy - sy, cz + sz)),
        Vector((cx + sx, cy + sy, cz + sz)), Vector((cx - sx, cy + sy, cz + sz)),
    ]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return verts, faces


def merge(*parts):
    """Concatenate (verts, faces) pairs into one, re-basing face indices."""
    verts: list[Vector] = []
    faces: list[tuple[int, ...]] = []
    for pv, pf in parts:
        off = len(verts)
        verts.extend(pv)
        faces.extend(tuple(i + off for i in f) for f in pf)
    return verts, faces


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def resample(points: Sequence[Vec3], count: int) -> list[Vector]:
    """Evenly spaced points along a polyline, by arc length."""
    pts = [Vector(p) for p in points]
    seg = [0.0]
    for a, b in zip(pts, pts[1:]):
        seg.append(seg[-1] + (b - a).length)
    total = seg[-1]
    out = []
    for i in range(count):
        d = total * i / (count - 1)
        k = max(0, min(len(seg) - 2, next((j for j in range(len(seg) - 1) if seg[j + 1] >= d), len(seg) - 2)))
        span = seg[k + 1] - seg[k]
        f = 0.0 if span < 1e-9 else (d - seg[k]) / span
        out.append(pts[k].lerp(pts[k + 1], f))
    return out


def catmull_rom(points: Sequence[Vec3], samples_per_span: int = 8) -> list[Vector]:
    """Smooth a control polyline into a curve passing through every control point."""
    pts = [Vector(p) for p in points]
    if len(pts) < 3:
        return pts
    ext = [pts[0] + (pts[0] - pts[1])] + pts + [pts[-1] + (pts[-1] - pts[-2])]
    out: list[Vector] = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for s in range(samples_per_span):
            t = s / samples_per_span
            t2, t3 = t * t, t * t * t
            out.append(
                0.5 * ((2 * p1) + (-p0 + p2) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            )
    out.append(pts[-1])
    return out


def recalc_normals(obj: bpy.types.Object) -> None:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
