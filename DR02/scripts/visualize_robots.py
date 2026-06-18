"""Generate unified visualization images for all robots in the repository.

Parses each URDF, builds the kinematic chain, loads STL meshes with trimesh,
applies link/joint/visual transforms, then renders an isometric 3/4 view via
matplotlib's mplot3d Poly3DCollection. Outputs square PNGs to images/.

Run:
    conda run -n isaoclab python DR02/scripts/visualize_robots.py
"""

from __future__ import annotations

import os
import sys
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

@dataclass
class Visual:
    mesh_path: str
    scale: np.ndarray  # (3,)
    origin: np.ndarray  # (4,4)


@dataclass
class Link:
    name: str
    visuals: List[Visual] = field(default_factory=list)


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # (4,4)


def _parse_xyz(text: Optional[str]) -> np.ndarray:
    if text is None:
        return np.zeros(3)
    parts = text.replace(",", " ").split()
    return np.array([float(p) for p in parts], dtype=float)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin_matrix(elem: Optional[ET.Element]) -> np.ndarray:
    T = np.eye(4)
    if elem is None:
        return T
    xyz = _parse_xyz(elem.get("xyz"))
    rpy = _parse_xyz(elem.get("rpy"))
    if xyz.size == 0:
        xyz = np.zeros(3)
    if rpy.size == 0:
        rpy = np.zeros(3)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def parse_urdf(urdf_path: str) -> Tuple[Dict[str, Link], Dict[str, Joint], str]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    links: Dict[str, Link] = {}
    for lk in root.findall("link"):
        name = lk.get("name")
        link = Link(name=name)
        for vis in lk.findall("visual"):
            geom = vis.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            if mesh_el is None:
                continue
            filename = mesh_el.get("filename", "")
            # strip a possible package:// prefix
            if filename.startswith("package://"):
                filename = filename.split("/", 3)[-1]
            mesh_path = os.path.normpath(os.path.join(urdf_dir, filename))
            scale_text = mesh_el.get("scale")
            scale = _parse_xyz(scale_text) if scale_text else np.ones(3)
            if scale.size == 1:
                scale = np.array([scale[0]] * 3)
            origin = _origin_matrix(vis.find("origin"))
            link.visuals.append(Visual(mesh_path=mesh_path, scale=scale, origin=origin))
        links[name] = link

    joints: Dict[str, Joint] = {}
    for jt in root.findall("joint"):
        name = jt.get("name")
        parent = jt.find("parent").get("link")
        child = jt.find("child").get("link")
        origin = _origin_matrix(jt.find("origin"))
        joints[name] = Joint(name=name, parent=parent, child=child, origin=origin)

    return links, joints, root.get("name", os.path.basename(urdf_path))


def compute_link_transforms(
    links: Dict[str, Link], joints: Dict[str, Joint]
) -> Dict[str, np.ndarray]:
    """Compute world transform for each link by walking the joint tree."""
    children_set = {j.child for j in joints.values()}
    roots = [n for n in links if n not in children_set]
    if not roots:
        roots = [next(iter(links))]

    # adjacency: parent -> [(joint, child)]
    adj: Dict[str, List[Joint]] = {n: [] for n in links}
    for j in joints.values():
        adj.setdefault(j.parent, []).append(j)

    transforms: Dict[str, np.ndarray] = {}
    for r in roots:
        transforms[r] = np.eye(4)
        stack = [r]
        while stack:
            cur = stack.pop()
            for j in adj.get(cur, []):
                transforms[j.child] = transforms[cur] @ j.origin
                stack.append(j.child)
    return transforms


# ---------------------------------------------------------------------------
# Mesh assembly
# ---------------------------------------------------------------------------

def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (h @ T.T)[:, :3]


def assemble_robot(
    urdf_path: str,
    world_transform: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return list of triangle arrays (N,3,3) in world frame, plus AABB.

    If ``world_transform`` is provided it is applied to every vertex after the
    link/visual transforms (e.g. to rotate the entire robot for rendering).
    """
    links, joints, _ = parse_urdf(urdf_path)
    link_T = compute_link_transforms(links, joints)

    if world_transform is None:
        world_transform = np.eye(4)

    triangle_arrays: List[np.ndarray] = []
    all_min = np.array([np.inf, np.inf, np.inf])
    all_max = -all_min

    for name, link in links.items():
        T_link = link_T.get(name)
        if T_link is None:
            continue
        for vis in link.visuals:
            if not os.path.isfile(vis.mesh_path):
                print(f"  [warn] missing mesh: {vis.mesh_path}", file=sys.stderr)
                continue
            try:
                m = trimesh.load(vis.mesh_path, force="mesh")
            except Exception as e:
                print(f"  [warn] load failed {vis.mesh_path}: {e}", file=sys.stderr)
                continue
            if m.is_empty or m.faces is None or len(m.faces) == 0:
                continue
            verts = np.asarray(m.vertices, dtype=float).copy()
            # apply scale (URDF scale on the mesh), then visual origin, then link xform
            verts *= vis.scale.reshape(1, 3)
            T_total = world_transform @ T_link @ vis.origin
            verts_w = _transform_points(T_total, verts)
            faces = np.asarray(m.faces)
            tris = verts_w[faces]  # (n_faces, 3, 3)

            # If any negative scale entry, mesh winding gets flipped; reverse
            # face order so normals stay outward (only matters if we did
            # backface culling; we don't, but keep tidy).
            if np.prod(vis.scale) < 0:
                tris = tris[:, ::-1, :]

            triangle_arrays.append(tris)
            all_min = np.minimum(all_min, verts_w.min(axis=0))
            all_max = np.maximum(all_max, verts_w.max(axis=0))

    if not triangle_arrays:
        raise RuntimeError(f"No meshes loaded from {urdf_path}")

    aabb = np.stack([all_min, all_max], axis=0)
    return triangle_arrays, aabb


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

VIEW_ELEV = 22.0
VIEW_AZIM = 135.0  # 3/4 isometric, looking from +x +y towards origin
LIGHT_DIR = np.array([0.5, -0.7, 1.0])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
BASE_COLOR = np.array([0.82, 0.84, 0.88])  # silver / light steel
EDGE_COLOR = (0.18, 0.20, 0.24, 0.10)
BG_COLOR = (0.965, 0.965, 0.97)


def _shade_faces(tris: np.ndarray) -> np.ndarray:
    """Return per-face RGBA based on simple Lambertian + ambient."""
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1)
    norms[norms == 0] = 1.0
    n = n / norms[:, None]
    intensity = np.clip(np.abs(n @ LIGHT_DIR), 0.0, 1.0)
    ambient = 0.55
    diffuse = 0.45
    shade = ambient + diffuse * intensity
    rgb = np.clip(BASE_COLOR[None, :] * shade[:, None], 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)
    return rgba


def render_robot(
    triangle_arrays: List[np.ndarray],
    aabb: np.ndarray,
    out_path: str,
    pixels: int = 900,
    dpi: int = 150,
    pad_factor: float = 0.55,
):
    tris = np.concatenate(triangle_arrays, axis=0)
    face_colors = _shade_faces(tris)

    fig_size = pixels / dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi, facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COLOR)

    # Edges add visible line weight; keep thin so geometry reads cleanly.
    coll = Poly3DCollection(
        tris,
        facecolors=face_colors,
        edgecolors=EDGE_COLOR,
        linewidths=0.05,
    )
    ax.add_collection3d(coll)

    # Set equal-aspect cube around the AABB with consistent padding
    center = (aabb[0] + aabb[1]) / 2.0
    extent = (aabb[1] - aabb[0]).max()
    pad = extent * pad_factor  # half-side of the view cube
    ax.set_xlim(center[0] - pad, center[0] + pad)
    ax.set_ylim(center[1] - pad, center[1] + pad)
    ax.set_zlim(center[2] - pad, center[2] + pad)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_axis_off()
    # Remove the default 3D panes
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_visible(False)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=dpi, facecolor=BG_COLOR, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REPO = "/home/ubuntu/deep_robotics/deep_robotics_model"
IMAGES_DIR = os.path.join(REPO, "images")


def _rotz(angle_deg: float) -> np.ndarray:
    """4x4 rotation about the world Z axis (counter-clockwise looking down -Z)."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return T


# Per-robot config: (label, urdf_path, out_name, world_transform, pad_factor)
# - world_transform: rotation/translation applied to every vertex before rendering
# - pad_factor: half-side of the view cube as a fraction of the AABB max extent
#   (smaller -> robot fills more of the frame). Default for non-DR02 stays 0.55.
DR02_WORLD_XFORM = _rotz(90.0)  # 90deg CCW about Z so DR02 faces forward
DR02_PAD = 0.45                 # slightly looser framing so the robot has breathing room

ROBOTS = [
    ("Lite3",     f"{REPO}/Lite3/urdf/Lite3.urdf",                "lite3.png",     None,               0.55),
    ("X30",       f"{REPO}/X30/urdf/X30.urdf",                    "x30.png",       None,               0.55),
    ("M20",       f"{REPO}/M20/urdf/M20.urdf",                    "m20.png",       None,               0.55),
    ("M20_Piper", f"{REPO}/M20_Piper/urdf/M20_Piper.urdf",        "m20_piper.png", None,               0.55),
    ("DR02_pro",  f"{REPO}/DR02/urdf/pro/DR02-pro.urdf",          "dr02_pro.png",  DR02_WORLD_XFORM,   DR02_PAD),
    ("DR02_std",  f"{REPO}/DR02/urdf/standard/DR02-STD.urdf",     "dr02_std.png",  DR02_WORLD_XFORM,   DR02_PAD),
]


def main() -> int:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    # Optional: restrict regeneration to a subset via CLI args (label match).
    # Example: `python visualize_robots.py DR02_pro DR02_std`
    selected = set(sys.argv[1:])
    failures = 0
    for label, urdf_path, out_name, world_xform, pad_factor in ROBOTS:
        if selected and label not in selected:
            continue
        out_path = os.path.join(IMAGES_DIR, out_name)
        print(f"[{label}] {urdf_path}")
        if not os.path.isfile(urdf_path):
            print(f"  [error] URDF not found", file=sys.stderr)
            failures += 1
            continue
        try:
            tris, aabb = assemble_robot(urdf_path, world_transform=world_xform)
            print(f"  meshes={len(tris)} aabb_size={aabb[1]-aabb[0]}")
            render_robot(tris, aabb, out_path, pad_factor=pad_factor)
            print(f"  -> {out_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [error] {e}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
