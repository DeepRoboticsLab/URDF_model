#!/usr/bin/env python3
"""
inspect_urdf.py
================

URDF validation utility for DR02 robot models.

Validates URDF XML files by:
    * Parsing the XML and checking well-formedness.
    * Listing every link and joint (name, type, parent, child).
    * Inspecting joint <limit> tags (lower, upper, effort, velocity).
    * Resolving every <mesh filename="..."> relative path and checking it
      actually exists on disk.
    * Detecting duplicate link / joint names.
    * Printing a summary table of the kinematic tree.
    * Reporting warnings and errors at the end.

Only the Python standard library is used (xml.etree.ElementTree), so this
script runs even when mujoco / pybullet are not installed.

Usage
-----
    conda run -n isaaclab python inspect_urdf.py
    conda run -n isaaclab python inspect_urdf.py --file path/to/robot.urdf
    conda run -n isaaclab python inspect_urdf.py --all

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

# ---------------------------------------------------------------------------
# Default DR02 URDF files
# ---------------------------------------------------------------------------
DR02_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
DEFAULT_URDF_FILES = [
    os.path.join(DR02_ROOT, "urdf", "pro", "DR02-pro.urdf"),
    os.path.join(DR02_ROOT, "urdf", "standard", "DR02-STD.urdf"),
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class URDFReport:
    file_path: str
    robot_name: str = ""
    links: List[str] = field(default_factory=list)
    joints: List[Dict[str, str]] = field(default_factory=list)
    duplicate_links: List[str] = field(default_factory=list)
    duplicate_joints: List[str] = field(default_factory=list)
    missing_meshes: List[str] = field(default_factory=list)
    found_meshes: List[str] = field(default_factory=list)
    joint_limit_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Core inspection
# ---------------------------------------------------------------------------
def _resolve_mesh_path(urdf_dir: str, raw_filename: str) -> str:
    """Resolve a URDF mesh filename to an absolute path.

    Supports the two cases used in DR02 URDFs:
        * relative path (./meshes/foo.STL)
        * package://pkg/meshes/foo.STL (treated as relative to urdf dir)
    """
    if raw_filename.startswith("package://"):
        # strip "package://<pkg>/"
        without_scheme = raw_filename[len("package://"):]
        parts = without_scheme.split("/", 1)
        rel = parts[1] if len(parts) > 1 else without_scheme
        return os.path.normpath(os.path.join(urdf_dir, rel))
    if os.path.isabs(raw_filename):
        return raw_filename
    return os.path.normpath(os.path.join(urdf_dir, raw_filename))


def inspect_urdf(file_path: str) -> URDFReport:
    report = URDFReport(file_path=file_path)

    if not os.path.isfile(file_path):
        report.errors.append(f"File does not exist: {file_path}")
        return report

    try:
        tree = ET.parse(file_path)
    except ET.ParseError as exc:
        report.errors.append(f"XML parse error: {exc}")
        return report

    root = tree.getroot()
    if root.tag != "robot":
        report.errors.append(
            f"Expected <robot> root element, got <{root.tag}>"
        )
        return report

    report.robot_name = root.attrib.get("name", "<unnamed>")
    urdf_dir = os.path.dirname(os.path.abspath(file_path))

    # ---- links ----------------------------------------------------------
    link_count: Dict[str, int] = defaultdict(int)
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if not name:
            report.warnings.append("Found <link> without a name attribute.")
            continue
        report.links.append(name)
        link_count[name] += 1

        # mesh references
        for mesh in link.iter("mesh"):
            fn = mesh.attrib.get("filename")
            if not fn:
                report.warnings.append(
                    f"Link '{name}' has <mesh> with no filename."
                )
                continue
            resolved = _resolve_mesh_path(urdf_dir, fn)
            if os.path.isfile(resolved):
                report.found_meshes.append(resolved)
            else:
                report.missing_meshes.append(f"{fn}  ->  {resolved}")

    report.duplicate_links = [
        n for n, c in link_count.items() if c > 1
    ]

    # ---- joints ---------------------------------------------------------
    joint_count: Dict[str, int] = defaultdict(int)
    for joint in root.findall("joint"):
        jname = joint.attrib.get("name", "")
        jtype = joint.attrib.get("type", "")
        if not jname:
            report.warnings.append("Found <joint> without a name attribute.")
            continue
        joint_count[jname] += 1

        parent_el = joint.find("parent")
        child_el = joint.find("child")
        parent = parent_el.attrib.get("link", "") if parent_el is not None else ""
        child = child_el.attrib.get("link", "") if child_el is not None else ""

        limit_el = joint.find("limit")
        if limit_el is not None:
            lower = limit_el.attrib.get("lower")
            upper = limit_el.attrib.get("upper")
            effort = limit_el.attrib.get("effort")
            velocity = limit_el.attrib.get("velocity")
        else:
            lower = upper = effort = velocity = None

        if jtype in {"revolute", "prismatic"}:
            if limit_el is None:
                report.joint_limit_issues.append(
                    f"Joint '{jname}' (type={jtype}) is missing <limit>."
                )
            else:
                try:
                    if lower is not None and upper is not None and float(lower) > float(upper):
                        report.joint_limit_issues.append(
                            f"Joint '{jname}' has lower>upper "
                            f"({lower} > {upper})."
                        )
                except ValueError:
                    report.joint_limit_issues.append(
                        f"Joint '{jname}' has non-numeric limit values."
                    )
                if effort is None or velocity is None:
                    report.warnings.append(
                        f"Joint '{jname}' missing effort/velocity in <limit>."
                    )

        if parent and parent not in link_count:
            report.warnings.append(
                f"Joint '{jname}' references unknown parent link '{parent}'."
            )
        if child and child not in link_count:
            report.warnings.append(
                f"Joint '{jname}' references unknown child link '{child}'."
            )

        report.joints.append(
            {
                "name": jname,
                "type": jtype,
                "parent": parent,
                "child": child,
                "lower": lower or "",
                "upper": upper or "",
                "effort": effort or "",
                "velocity": velocity or "",
            }
        )

    report.duplicate_joints = [
        n for n, c in joint_count.items() if c > 1
    ]

    if report.duplicate_links:
        report.errors.append(
            f"Duplicate link names: {sorted(set(report.duplicate_links))}"
        )
    if report.duplicate_joints:
        report.errors.append(
            f"Duplicate joint names: {sorted(set(report.duplicate_joints))}"
        )
    if report.missing_meshes:
        report.errors.append(
            f"{len(report.missing_meshes)} mesh file(s) missing on disk."
        )

    return report


# ---------------------------------------------------------------------------
# Tree printing
# ---------------------------------------------------------------------------
def _print_tree(report: URDFReport) -> None:
    children: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    parents = set()
    all_children = set()
    for j in report.joints:
        children[j["parent"]].append((j["child"], j["name"]))
        parents.add(j["parent"])
        all_children.add(j["child"])

    roots = [l for l in report.links if l not in all_children]
    if not roots:
        roots = report.links[:1]

    def walk(link: str, prefix: str = "", is_last: bool = True) -> None:
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{link}")
        new_prefix = prefix + ("    " if is_last else "│   ")
        kids = children.get(link, [])
        for i, (child, jname) in enumerate(kids):
            last = i == len(kids) - 1
            jconn = "└── " if last else "├── "
            print(f"{new_prefix}{jconn}[joint: {jname}]")
            walk(
                child,
                new_prefix + ("    " if last else "│   "),
                is_last=True,
            )

    for root in roots:
        walk(root)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_report(report: URDFReport, verbose: bool = True) -> None:
    bar = "=" * 78
    print(bar)
    print(f"URDF: {report.file_path}")
    print(f"Robot name: {report.robot_name}")
    print(bar)
    if report.errors:
        print("STATUS: FAIL")
    elif report.warnings or report.joint_limit_issues:
        print("STATUS: PASS (with warnings)")
    else:
        print("STATUS: PASS")
    print(f"  links:        {len(report.links)}")
    print(f"  joints:       {len(report.joints)}")
    print(f"  meshes found: {len(report.found_meshes)}")
    print(f"  meshes miss:  {len(report.missing_meshes)}")
    print(f"  warnings:     {len(report.warnings) + len(report.joint_limit_issues)}")
    print(f"  errors:       {len(report.errors)}")

    if verbose:
        print("\n-- Joints --")
        header = f"{'name':<32}{'type':<12}{'parent':<22}{'child':<22}"
        print(header)
        print("-" * len(header))
        for j in report.joints:
            print(
                f"{j['name']:<32}{j['type']:<12}"
                f"{j['parent']:<22}{j['child']:<22}"
            )

        print("\n-- Joint limits --")
        for j in report.joints:
            if j["type"] in {"revolute", "prismatic"}:
                print(
                    f"  {j['name']:<32} "
                    f"lower={j['lower']:<10} upper={j['upper']:<10} "
                    f"effort={j['effort']:<8} velocity={j['velocity']:<8}"
                )

        print("\n-- Kinematic tree --")
        _print_tree(report)

    if report.missing_meshes:
        print("\n-- Missing meshes --")
        for m in report.missing_meshes:
            print(f"  ! {m}")
    if report.joint_limit_issues:
        print("\n-- Joint limit issues --")
        for issue in report.joint_limit_issues:
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
                        help="URDF file path. Repeat to inspect multiple.")
    parser.add_argument("--all", action="store_true",
                        help="Inspect all default DR02 URDF files.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print summary only, skip detailed tables.")
    args = parser.parse_args()

    files = args.file or []
    if args.all or not files:
        files = list(DEFAULT_URDF_FILES) + files

    overall_ok = True
    for f in files:
        report = inspect_urdf(f)
        print_report(report, verbose=not args.quiet)
        overall_ok = overall_ok and report.passed

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
