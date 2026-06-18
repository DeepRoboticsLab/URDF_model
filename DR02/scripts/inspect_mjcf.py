#!/usr/bin/env python3
"""
inspect_mjcf.py
================

MJCF (MuJoCo XML) validation utility for DR02 robot models.

Validates MJCF XML files by:
    * Parsing the XML and checking well-formedness.
    * Listing every <body> and <joint> with name, type, range, axis.
    * Verifying every <mesh file="..."> in <asset> exists on disk
      (relative to the <compiler meshdir="..."/> attribute).
    * Detecting missing or duplicate body / joint names.
    * Validating actuator definitions reference existing joints.
    * Printing the kinematic tree of <body> elements.
    * Reporting warnings and errors.

Only the standard library (xml.etree.ElementTree) is used because mujoco /
pybullet are not available in the target environment.

Usage
-----
    conda run -n isaaclab python inspect_mjcf.py
    conda run -n isaaclab python inspect_mjcf.py --file path/to/robot.xml
    conda run -n isaaclab python inspect_mjcf.py --all

Exit codes
----------
    0  -> all files passed
    1  -> at least one file produced an error
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DR02_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
DEFAULT_MJCF_FILES = [
    os.path.join(DR02_ROOT, "mjcf", "pro", "DR02_pro.xml"),
    os.path.join(DR02_ROOT, "mjcf", "standard", "DR02_S.xml"),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class MJCFReport:
    file_path: str
    model_name: str = ""
    meshdir: str = ""
    bodies: List[str] = field(default_factory=list)
    joints: List[Dict[str, str]] = field(default_factory=list)
    actuators: List[Dict[str, str]] = field(default_factory=list)
    duplicate_bodies: List[str] = field(default_factory=list)
    duplicate_joints: List[str] = field(default_factory=list)
    missing_meshes: List[str] = field(default_factory=list)
    found_meshes: List[str] = field(default_factory=list)
    actuator_issues: List[str] = field(default_factory=list)
    body_tree: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    body_root: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
def _walk_bodies(parent_el: ET.Element,
                 parent_name: Optional[str],
                 report: MJCFReport,
                 body_count: Dict[str, int],
                 joint_count: Dict[str, int]) -> None:
    """Recursively walk <body> elements and collect data."""
    for body in parent_el.findall("body"):
        bname = body.attrib.get("name", "")
        if not bname:
            report.warnings.append("Found <body> without a name attribute.")
            bname = f"<unnamed_{len(report.bodies)}>"
        report.bodies.append(bname)
        body_count[bname] += 1

        if parent_name is None:
            if report.body_root is None:
                report.body_root = bname
        else:
            report.body_tree[parent_name].append(bname)

        for joint in body.findall("joint"):
            jname = joint.attrib.get("name", "")
            jtype = joint.attrib.get("type", "hinge")  # MuJoCo default
            jrange = joint.attrib.get("range", "")
            jaxis = joint.attrib.get("axis", "")
            if jname:
                joint_count[jname] += 1
            else:
                report.warnings.append(
                    f"Body '{bname}' has <joint> without a name."
                )
            report.joints.append({
                "name": jname,
                "type": jtype,
                "range": jrange,
                "axis": jaxis,
                "body": bname,
            })

        # freejoint: special MuJoCo joint
        for fj in body.findall("freejoint"):
            jname = fj.attrib.get("name", "")
            report.joints.append({
                "name": jname or f"{bname}_freejoint",
                "type": "free",
                "range": "",
                "axis": "",
                "body": bname,
            })

        _walk_bodies(body, bname, report, body_count, joint_count)


def inspect_mjcf(file_path: str) -> MJCFReport:
    report = MJCFReport(file_path=file_path)

    if not os.path.isfile(file_path):
        report.errors.append(f"File does not exist: {file_path}")
        return report

    try:
        tree = ET.parse(file_path)
    except ET.ParseError as exc:
        report.errors.append(f"XML parse error: {exc}")
        return report

    root = tree.getroot()
    if root.tag != "mujoco":
        report.errors.append(
            f"Expected <mujoco> root element, got <{root.tag}>"
        )
        return report

    report.model_name = root.attrib.get("model", "<unnamed>")
    mjcf_dir = os.path.dirname(os.path.abspath(file_path))

    compiler = root.find("compiler")
    if compiler is not None:
        report.meshdir = compiler.attrib.get("meshdir", "")
    abs_meshdir = os.path.normpath(os.path.join(mjcf_dir, report.meshdir or "."))

    # ---- assets / meshes -----------------------------------------------
    asset = root.find("asset")
    declared_meshes: Dict[str, str] = {}
    if asset is not None:
        for mesh in asset.findall("mesh"):
            mname = mesh.attrib.get("name", "")
            mfile = mesh.attrib.get("file", "")
            if not mfile:
                report.warnings.append(
                    f"<mesh name='{mname}'> in <asset> has no file attribute."
                )
                continue
            resolved = mfile
            if not os.path.isabs(mfile):
                resolved = os.path.normpath(os.path.join(abs_meshdir, mfile))
            declared_meshes[mname] = resolved
            if os.path.isfile(resolved):
                report.found_meshes.append(resolved)
            else:
                report.missing_meshes.append(f"{mfile}  ->  {resolved}")
    else:
        report.warnings.append("No <asset> section found.")

    # ---- worldbody / bodies / joints -----------------------------------
    body_count: Dict[str, int] = defaultdict(int)
    joint_count: Dict[str, int] = defaultdict(int)
    worldbody = root.find("worldbody")
    if worldbody is None:
        report.errors.append("No <worldbody> section found.")
        return report

    _walk_bodies(worldbody, None, report, body_count, joint_count)

    report.duplicate_bodies = [n for n, c in body_count.items() if c > 1]
    report.duplicate_joints = [n for n, c in joint_count.items() if c > 1]

    # ---- check geom mesh references inside bodies ----------------------
    for geom in worldbody.iter("geom"):
        if geom.attrib.get("type") == "mesh":
            mref = geom.attrib.get("mesh", "")
            if mref and mref not in declared_meshes:
                report.warnings.append(
                    f"<geom> references mesh '{mref}' not declared in <asset>."
                )

    # ---- actuators ------------------------------------------------------
    joint_names = {j["name"] for j in report.joints if j["name"]}
    actuator_section = root.find("actuator")
    if actuator_section is not None:
        for act in list(actuator_section):
            aname = act.attrib.get("name", "")
            atype = act.tag
            target = act.attrib.get("joint", "")
            ctrl_range = act.attrib.get("ctrlrange", "")
            report.actuators.append({
                "name": aname,
                "type": atype,
                "joint": target,
                "ctrlrange": ctrl_range,
            })
            if target and target not in joint_names:
                report.actuator_issues.append(
                    f"Actuator '{aname}' targets unknown joint '{target}'."
                )

    # ---- elevate to errors ---------------------------------------------
    if report.duplicate_bodies:
        report.errors.append(
            f"Duplicate body names: {sorted(set(report.duplicate_bodies))}"
        )
    if report.duplicate_joints:
        report.errors.append(
            f"Duplicate joint names: {sorted(set(report.duplicate_joints))}"
        )
    if report.missing_meshes:
        report.errors.append(
            f"{len(report.missing_meshes)} mesh file(s) missing on disk."
        )
    if report.actuator_issues:
        report.errors.append(
            f"{len(report.actuator_issues)} actuator/joint mismatch(es)."
        )

    return report


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def _print_body_tree(report: MJCFReport) -> None:
    def walk(name: str, prefix: str = "", is_last: bool = True) -> None:
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{name}")
        new_prefix = prefix + ("    " if is_last else "│   ")
        kids = report.body_tree.get(name, [])
        for i, c in enumerate(kids):
            walk(c, new_prefix, i == len(kids) - 1)

    if report.body_root:
        walk(report.body_root)


def print_report(report: MJCFReport, verbose: bool = True) -> None:
    bar = "=" * 78
    print(bar)
    print(f"MJCF: {report.file_path}")
    print(f"Model: {report.model_name}")
    print(f"meshdir: {report.meshdir or '(unset)'}")
    print(bar)
    if report.errors:
        print("STATUS: FAIL")
    elif report.warnings or report.actuator_issues:
        print("STATUS: PASS (with warnings)")
    else:
        print("STATUS: PASS")
    print(f"  bodies:       {len(report.bodies)}")
    print(f"  joints:       {len(report.joints)}")
    print(f"  actuators:    {len(report.actuators)}")
    print(f"  meshes found: {len(report.found_meshes)}")
    print(f"  meshes miss:  {len(report.missing_meshes)}")
    print(f"  warnings:     {len(report.warnings)}")
    print(f"  errors:       {len(report.errors)}")

    if verbose:
        print("\n-- Joints --")
        header = f"{'name':<32}{'type':<10}{'range':<22}{'axis':<14}{'body':<22}"
        print(header)
        print("-" * len(header))
        for j in report.joints:
            print(
                f"{j['name']:<32}{j['type']:<10}"
                f"{j['range']:<22}{j['axis']:<14}{j['body']:<22}"
            )

        if report.actuators:
            print("\n-- Actuators --")
            for a in report.actuators:
                print(
                    f"  {a['name']:<28} type={a['type']:<10} "
                    f"joint={a['joint']:<28} ctrlrange={a['ctrlrange']}"
                )

        print("\n-- Body tree --")
        _print_body_tree(report)

    if report.missing_meshes:
        print("\n-- Missing meshes --")
        for m in report.missing_meshes:
            print(f"  ! {m}")
    if report.actuator_issues:
        print("\n-- Actuator issues --")
        for issue in report.actuator_issues:
            print(f"  ! {issue}")
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
                        help="MJCF file path. Repeat to inspect multiple.")
    parser.add_argument("--all", action="store_true",
                        help="Inspect all default DR02 MJCF files.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print summary only.")
    args = parser.parse_args()

    files = args.file or []
    if args.all or not files:
        files = list(DEFAULT_MJCF_FILES) + files

    overall_ok = True
    for f in files:
        report = inspect_mjcf(f)
        print_report(report, verbose=not args.quiet)
        overall_ok = overall_ok and report.passed

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
