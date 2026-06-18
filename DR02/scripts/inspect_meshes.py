#!/usr/bin/env python3
"""
inspect_meshes.py
==================

STL mesh validation utility for DR02 robot models, using ``trimesh``.

For every STL file in the DR02 mjcf/<variant>/meshes/ and
urdf/<variant>/meshes/ folders the script:
    * Loads the mesh with ``trimesh``.
    * Checks the mesh is watertight.
    * Looks for degenerate faces (zero-area triangles).
    * Reports vertex / face counts.
    * Reports the axis-aligned bounding box and flags meshes whose
      diagonal looks unreasonable for a humanoid robot link
      (default plausible range: 0.005 m .. 5.0 m).
    * Aggregates statistics per variant (mjcf-pro, mjcf-standard,
      urdf-pro, urdf-standard).

Usage
-----
    conda run -n isaaclab python inspect_meshes.py
    conda run -n isaaclab python inspect_meshes.py --dir /path/to/meshes
    conda run -n isaaclab python inspect_meshes.py --all

Exit codes
----------
    0 -> all meshes passed
    1 -> at least one mesh produced an error
    2 -> trimesh not importable
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TRIMESH_AVAILABLE = True
TRIMESH_IMPORT_ERROR: Optional[str] = None
try:
    import numpy as np  # type: ignore
    import trimesh  # type: ignore
except Exception as exc:  # pragma: no cover - depends on env
    TRIMESH_AVAILABLE = False
    TRIMESH_IMPORT_ERROR = str(exc)

DR02_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)

DEFAULT_MESH_DIRS: List[Tuple[str, str]] = [
    ("mjcf-pro", os.path.join(DR02_ROOT, "mjcf", "pro", "meshes")),
    ("mjcf-standard", os.path.join(DR02_ROOT, "mjcf", "standard", "meshes")),
    ("urdf-pro", os.path.join(DR02_ROOT, "urdf", "pro", "meshes")),
    ("urdf-standard", os.path.join(DR02_ROOT, "urdf", "standard", "meshes")),
]

# Plausible humanoid robot link size range (in metres).
MIN_DIAGONAL_M = 0.005
MAX_DIAGONAL_M = 5.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class MeshReport:
    file_path: str
    vertices: int = 0
    faces: int = 0
    watertight: bool = False
    degenerate_faces: int = 0
    bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    diagonal: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


@dataclass
class GroupReport:
    label: str
    directory: str
    meshes: List[MeshReport] = field(default_factory=list)

    @property
    def total_vertices(self) -> int:
        return sum(m.vertices for m in self.meshes)

    @property
    def total_faces(self) -> int:
        return sum(m.faces for m in self.meshes)

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.meshes)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
def inspect_mesh(file_path: str) -> MeshReport:
    report = MeshReport(file_path=file_path)
    if not os.path.isfile(file_path):
        report.errors.append("File does not exist.")
        return report

    try:
        mesh = trimesh.load(file_path, force="mesh")
    except Exception as exc:
        report.errors.append(f"Failed to load mesh: {exc}")
        return report

    if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        report.errors.append("Mesh loaded but has no geometry.")
        return report

    report.vertices = int(len(mesh.vertices))
    report.faces = int(len(mesh.faces)) if hasattr(mesh, "faces") else 0
    report.watertight = bool(getattr(mesh, "is_watertight", False))

    # Degenerate faces (zero-area triangles)
    try:
        nondeg_mask = mesh.nondegenerate_faces()
        report.degenerate_faces = int(report.faces - int(np.count_nonzero(nondeg_mask)))
    except Exception:
        # Fallback: compute area-based detection
        try:
            areas = mesh.area_faces
            report.degenerate_faces = int(np.count_nonzero(areas <= 1e-12))
        except Exception:
            report.degenerate_faces = -1  # unknown

    try:
        bmin, bmax = mesh.bounds
        report.bbox_min = tuple(float(v) for v in bmin)
        report.bbox_max = tuple(float(v) for v in bmax)
        diag = float(np.linalg.norm(np.asarray(bmax) - np.asarray(bmin)))
        report.diagonal = diag
    except Exception as exc:
        report.warnings.append(f"Could not compute bounding box: {exc}")

    # Warnings -----------------------------------------------------------
    if not report.watertight:
        report.warnings.append("Mesh is not watertight.")
    if report.degenerate_faces and report.degenerate_faces > 0:
        report.warnings.append(
            f"{report.degenerate_faces} degenerate face(s) detected."
        )
    if report.diagonal and (
        report.diagonal < MIN_DIAGONAL_M or report.diagonal > MAX_DIAGONAL_M
    ):
        report.warnings.append(
            f"Bounding box diagonal {report.diagonal:.4f} m outside plausible "
            f"range [{MIN_DIAGONAL_M}, {MAX_DIAGONAL_M}]."
        )

    return report


def inspect_directory(label: str, directory: str) -> GroupReport:
    group = GroupReport(label=label, directory=directory)
    if not os.path.isdir(directory):
        return group
    for entry in sorted(os.listdir(directory)):
        if entry.lower().endswith(".stl"):
            group.meshes.append(inspect_mesh(os.path.join(directory, entry)))
    return group


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_group(group: GroupReport, verbose: bool = True) -> None:
    bar = "=" * 78
    print(bar)
    print(f"Mesh group: {group.label}")
    print(f"directory: {group.directory}")
    print(bar)
    if not os.path.isdir(group.directory):
        print("STATUS: SKIP (directory not found)\n")
        return
    if not group.meshes:
        print("STATUS: SKIP (no STL files found)\n")
        return

    has_err = any(m.errors for m in group.meshes)
    has_warn = any(m.warnings for m in group.meshes)
    if has_err:
        print("STATUS: FAIL")
    elif has_warn:
        print("STATUS: PASS (with warnings)")
    else:
        print("STATUS: PASS")
    print(f"  files:           {len(group.meshes)}")
    print(f"  total vertices:  {group.total_vertices}")
    print(f"  total faces:     {group.total_faces}")

    if verbose:
        print(
            f"\n  {'mesh':<32}{'verts':>9}{'faces':>9}"
            f"{'watertight':>12}{'degen':>8}{'diag(m)':>10}"
        )
        print("  " + "-" * 80)
        for m in group.meshes:
            name = os.path.basename(m.file_path)
            wt = "yes" if m.watertight else "no"
            deg = "?" if m.degenerate_faces < 0 else str(m.degenerate_faces)
            print(
                f"  {name:<32}{m.vertices:>9}{m.faces:>9}"
                f"{wt:>12}{deg:>8}{m.diagonal:>10.4f}"
            )

    # Issues per mesh
    for m in group.meshes:
        if m.errors:
            print(f"\n  X {os.path.basename(m.file_path)}")
            for e in m.errors:
                print(f"      {e}")
        elif m.warnings:
            print(f"\n  - {os.path.basename(m.file_path)}")
            for w in m.warnings:
                print(f"      {w}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", "-d", action="append",
                        help="Directory containing STL files. Repeatable.")
    parser.add_argument("--all", action="store_true",
                        help="Inspect all default DR02 mesh directories.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print summary only, skip per-mesh table.")
    args = parser.parse_args()

    if not TRIMESH_AVAILABLE:
        print(
            "ERROR: trimesh / numpy not importable.\n"
            f"       reason: {TRIMESH_IMPORT_ERROR}\n"
            "       Run with: conda run -n isaaclab python inspect_meshes.py"
        )
        return 2

    groups: List[GroupReport] = []
    if args.dir and not args.all:
        for d in args.dir:
            groups.append(inspect_directory(os.path.basename(d.rstrip(os.sep)), d))
    else:
        for label, d in DEFAULT_MESH_DIRS:
            groups.append(inspect_directory(label, d))
        if args.dir:
            for d in args.dir:
                groups.append(inspect_directory(
                    os.path.basename(d.rstrip(os.sep)), d
                ))

    overall_ok = True
    for g in groups:
        print_group(g, verbose=not args.quiet)
        overall_ok = overall_ok and g.passed

    # Cross-group summary
    bar = "=" * 78
    print(bar)
    print("Mesh inspection summary")
    print(bar)
    for g in groups:
        files = len(g.meshes)
        if files == 0:
            status = "SKIP"
        elif any(m.errors for m in g.meshes):
            status = "FAIL"
        elif any(m.warnings for m in g.meshes):
            status = "WARN"
        else:
            status = "PASS"
        print(
            f"  {g.label:<18} files={files:<3} "
            f"verts={g.total_vertices:<8} faces={g.total_faces:<8} "
            f"-> {status}"
        )
    print()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
