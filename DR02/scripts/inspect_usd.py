#!/usr/bin/env python3
"""
inspect_usd.py
===============

USD validation utility for DR02 robot models.

Validates USD stage files using the ``pxr`` USD bindings shipped with
Isaac Sim / OmniIsaacGym. The script:
    * Opens the USD stage and confirms it loads without errors.
    * Lists every prim and its type (with optional hierarchical tree).
    * Locates articulation roots (UsdPhysics.ArticulationRootAPI) and
      enumerates child joints with their type / lower / upper limits.
    * Counts physics-related prims: rigid bodies, collision APIs,
      mass APIs, joints.
    * Resolves and verifies sublayer / reference / payload asset paths.
    * Reports warnings and errors at the end.

The script handles both the main USD file and the configuration sub-files
(``*_base.usd``, ``*_physics.usd``, ``*_robot.usd``, ``*_sensor.usd``).

Usage
-----
    conda run -n isaaclab python inspect_usd.py
    conda run -n isaaclab python inspect_usd.py --file path/to/robot.usd
    conda run -n isaaclab python inspect_usd.py --all

Exit codes
----------
    0 -> all files passed
    1 -> at least one file produced an error
    2 -> the pxr/USD bindings are not importable
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# pxr import (with helpful error)
# ---------------------------------------------------------------------------
PXR_AVAILABLE = True
PXR_IMPORT_ERROR: Optional[str] = None
try:
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf  # type: ignore
except Exception as exc:  # pragma: no cover - depends on env
    PXR_AVAILABLE = False
    PXR_IMPORT_ERROR = str(exc)

DR02_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)

DEFAULT_USD_MAIN_FILES = [
    os.path.join(DR02_ROOT, "usd", "pro", "DR02-pro.usd"),
    os.path.join(DR02_ROOT, "usd", "standard", "DR02-STD.usd"),
]

DEFAULT_USD_SUBFILES = [
    os.path.join(DR02_ROOT, "usd", "pro", "configuration", "DR02-pro_base.usd"),
    os.path.join(DR02_ROOT, "usd", "pro", "configuration", "DR02-pro_physics.usd"),
    os.path.join(DR02_ROOT, "usd", "pro", "configuration", "DR02-pro_robot.usd"),
    os.path.join(DR02_ROOT, "usd", "pro", "configuration", "DR02-pro_sensor.usd"),
    os.path.join(DR02_ROOT, "usd", "standard", "configuration", "DR02-STD_base.usd"),
    os.path.join(DR02_ROOT, "usd", "standard", "configuration", "DR02-STD_physics.usd"),
    os.path.join(DR02_ROOT, "usd", "standard", "configuration", "DR02-STD_robot.usd"),
    os.path.join(DR02_ROOT, "usd", "standard", "configuration", "DR02-STD_sensor.usd"),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class USDReport:
    file_path: str
    default_prim: str = ""
    up_axis: str = ""
    meters_per_unit: float = 1.0
    prim_count: int = 0
    type_counts: Dict[str, int] = field(default_factory=dict)
    articulation_roots: List[str] = field(default_factory=list)
    joints: List[Dict[str, str]] = field(default_factory=list)
    rigid_body_count: int = 0
    collision_count: int = 0
    mass_api_count: int = 0
    sublayers: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    payloads: List[str] = field(default_factory=list)
    missing_assets: List[str] = field(default_factory=list)
    prim_tree_lines: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
def _safe_attr(prim, name: str) -> Optional[object]:
    attr = prim.GetAttribute(name)
    if attr and attr.IsValid() and attr.HasValue():
        try:
            return attr.Get()
        except Exception:
            return None
    return None


def inspect_usd(file_path: str) -> USDReport:
    report = USDReport(file_path=file_path)

    if not PXR_AVAILABLE:
        report.errors.append(
            f"pxr/USD not importable: {PXR_IMPORT_ERROR}. "
            "Run with: conda run -n isaaclab python inspect_usd.py"
        )
        return report

    if not os.path.isfile(file_path):
        report.errors.append(f"File does not exist: {file_path}")
        return report

    try:
        stage = Usd.Stage.Open(file_path)
    except Exception as exc:
        report.errors.append(f"Failed to open USD stage: {exc}")
        return report

    if stage is None:
        report.errors.append("Usd.Stage.Open returned None.")
        return report

    # ---- stage metadata --------------------------------------------------
    try:
        report.up_axis = UsdGeom.GetStageUpAxis(stage) or ""
        report.meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    except Exception as exc:
        report.warnings.append(f"Could not read stage metadata: {exc}")

    default_prim = stage.GetDefaultPrim()
    report.default_prim = default_prim.GetPath().pathString if default_prim else ""

    # ---- sublayers / references / payloads -------------------------------
    root_layer = stage.GetRootLayer()
    if root_layer is not None:
        for sublayer in root_layer.subLayerPaths:
            report.sublayers.append(sublayer)
            resolved = sublayer
            if not os.path.isabs(sublayer):
                resolved = os.path.normpath(
                    os.path.join(os.path.dirname(file_path), sublayer)
                )
            if not os.path.isfile(resolved):
                report.missing_assets.append(f"sublayer: {sublayer} -> {resolved}")

    # ---- prim traversal --------------------------------------------------
    type_counts: Dict[str, int] = {}
    tree_lines: List[str] = []

    for prim in stage.Traverse():
        report.prim_count += 1
        ptype = prim.GetTypeName() or "(typeless)"
        type_counts[ptype] = type_counts.get(ptype, 0) + 1

        depth = prim.GetPath().pathString.count("/") - 1
        indent = "  " * max(0, depth)
        tree_lines.append(
            f"{indent}- {prim.GetName()} <{ptype}>"
        )

        # Articulation roots
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            report.articulation_roots.append(prim.GetPath().pathString)

        # Rigid bodies / collision / mass
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            report.rigid_body_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            report.collision_count += 1
        if prim.HasAPI(UsdPhysics.MassAPI):
            report.mass_api_count += 1

        # Joints
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            jtype = ptype
            lower = upper = ""
            # Type-specific limits
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rj = UsdPhysics.RevoluteJoint(prim)
                lo = rj.GetLowerLimitAttr().Get()
                up = rj.GetUpperLimitAttr().Get()
                lower = "" if lo is None else f"{lo:.4f}"
                upper = "" if up is None else f"{up:.4f}"
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                pj = UsdPhysics.PrismaticJoint(prim)
                lo = pj.GetLowerLimitAttr().Get()
                up = pj.GetUpperLimitAttr().Get()
                lower = "" if lo is None else f"{lo:.4f}"
                upper = "" if up is None else f"{up:.4f}"
            try:
                bodies0 = joint.GetBody0Rel().GetTargets()
                bodies1 = joint.GetBody1Rel().GetTargets()
                b0 = bodies0[0].pathString if bodies0 else ""
                b1 = bodies1[0].pathString if bodies1 else ""
            except Exception:
                b0 = b1 = ""
            report.joints.append({
                "path": prim.GetPath().pathString,
                "type": jtype,
                "body0": b0,
                "body1": b1,
                "lower": lower,
                "upper": upper,
            })

        # References / payloads
        try:
            ref_query = Usd.PrimCompositionQuery.GetDirectReferences(prim)
            for arc in ref_query.GetCompositionArcs():
                tgt_layer = arc.GetTargetLayer()
                if tgt_layer is not None:
                    p = tgt_layer.realPath or tgt_layer.identifier
                    report.references.append(f"{prim.GetPath()} -> {p}")
        except Exception:
            pass

    report.type_counts = type_counts
    report.prim_tree_lines = tree_lines

    if report.prim_count == 0:
        report.warnings.append("Stage contains zero prims.")
    if not report.default_prim:
        report.warnings.append("No defaultPrim set on stage.")

    if report.missing_assets:
        report.errors.append(
            f"{len(report.missing_assets)} missing referenced asset(s)."
        )

    return report


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_report(report: USDReport, verbose: bool = True,
                 max_tree_lines: int = 80) -> None:
    bar = "=" * 78
    print(bar)
    print(f"USD: {report.file_path}")
    print(bar)
    if report.errors:
        print("STATUS: FAIL")
    elif report.warnings:
        print("STATUS: PASS (with warnings)")
    else:
        print("STATUS: PASS")

    print(f"  default prim:      {report.default_prim or '(none)'}")
    print(f"  up axis:           {report.up_axis or '(none)'}")
    print(f"  meters per unit:   {report.meters_per_unit}")
    print(f"  total prims:       {report.prim_count}")
    print(f"  rigid bodies:      {report.rigid_body_count}")
    print(f"  collision shapes:  {report.collision_count}")
    print(f"  mass APIs:         {report.mass_api_count}")
    print(f"  joints:            {len(report.joints)}")
    print(f"  articulation roots:{len(report.articulation_roots)}")
    print(f"  sublayers:         {len(report.sublayers)}")
    print(f"  references:        {len(report.references)}")
    print(f"  warnings:          {len(report.warnings)}")
    print(f"  errors:            {len(report.errors)}")

    if verbose:
        if report.type_counts:
            print("\n-- Prim type counts --")
            for tname in sorted(report.type_counts,
                                key=lambda k: -report.type_counts[k]):
                print(f"  {tname:<32} {report.type_counts[tname]}")

        if report.articulation_roots:
            print("\n-- Articulation roots --")
            for path in report.articulation_roots:
                print(f"  {path}")

        if report.joints:
            print("\n-- Joints --")
            header = f"{'path':<55}{'type':<22}{'lower':<10}{'upper':<10}"
            print(header)
            print("-" * len(header))
            for j in report.joints:
                print(
                    f"{j['path']:<55}{j['type']:<22}"
                    f"{j['lower']:<10}{j['upper']:<10}"
                )

        if report.sublayers:
            print("\n-- Sublayers --")
            for s in report.sublayers:
                print(f"  {s}")

        if report.prim_tree_lines:
            print("\n-- Prim tree --")
            shown = report.prim_tree_lines[:max_tree_lines]
            for line in shown:
                print(line)
            if len(report.prim_tree_lines) > max_tree_lines:
                remain = len(report.prim_tree_lines) - max_tree_lines
                print(f"  ... ({remain} more prims)")

    if report.missing_assets:
        print("\n-- Missing referenced assets --")
        for m in report.missing_assets:
            print(f"  ! {m}")
    if report.warnings:
        print("\n-- Warnings --")
        for w in report.warnings:
            print(f"  - {w}")
    if report.errors:
        print("\n-- Errors --")
        for e in report.errors:
            print(f"  X {e}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", "-f", action="append",
                        help="USD file path. Repeat to inspect multiple.")
    parser.add_argument("--all", action="store_true",
                        help="Inspect main + configuration USD files.")
    parser.add_argument("--main-only", action="store_true",
                        help="Inspect only the two top-level main USD files.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print summary only.")
    args = parser.parse_args()

    if not PXR_AVAILABLE:
        print(
            "ERROR: pxr/USD bindings not importable in current interpreter.\n"
            f"       reason: {PXR_IMPORT_ERROR}\n"
            "       Run with: conda run -n isaaclab python inspect_usd.py"
        )
        return 2

    files = args.file or []
    if args.main_only:
        files = list(DEFAULT_USD_MAIN_FILES) + files
    elif args.all or not files:
        files = list(DEFAULT_USD_MAIN_FILES) + list(DEFAULT_USD_SUBFILES) + files

    overall_ok = True
    for f in files:
        report = inspect_usd(f)
        print_report(report, verbose=not args.quiet)
        overall_ok = overall_ok and report.passed

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
