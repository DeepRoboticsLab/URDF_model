#!/usr/bin/env python3
"""
inspect_all.py
===============

Master DR02 inspection driver.

Imports the four sibling validators
(:mod:`inspect_urdf`, :mod:`inspect_mjcf`, :mod:`inspect_usd`,
:mod:`inspect_meshes`) and produces a consolidated, color-coded
pass / fail report for every model file in the DR02 tree.

Usage
-----
    conda run -n isaaclab python inspect_all.py
    conda run -n isaaclab python inspect_all.py --all
    conda run -n isaaclab python inspect_all.py --urdf
    conda run -n isaaclab python inspect_all.py --mjcf --mesh
    conda run -n isaaclab python inspect_all.py --usd --quiet

Color output is auto-disabled when stdout is not a TTY or
``NO_COLOR`` is set.

Exit codes
----------
    0 -> all selected checks passed
    1 -> at least one check failed
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# Make sibling modules importable when run from anywhere.
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import inspect_urdf as urdf_mod  # noqa: E402
import inspect_mjcf as mjcf_mod  # noqa: E402
import inspect_meshes as mesh_mod  # noqa: E402

# USD module is imported lazily because it depends on pxr.
usd_mod = None
USD_IMPORT_ERROR: Optional[str] = None
try:
    import inspect_usd as _usd_mod  # noqa: E402
    usd_mod = _usd_mod
    if not getattr(usd_mod, "PXR_AVAILABLE", False):
        USD_IMPORT_ERROR = getattr(usd_mod, "PXR_IMPORT_ERROR", "pxr unavailable")
except Exception as exc:
    USD_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


COLOR = _color_enabled()
GREEN = "\033[32m" if COLOR else ""
YELLOW = "\033[33m" if COLOR else ""
RED = "\033[31m" if COLOR else ""
CYAN = "\033[36m" if COLOR else ""
BOLD = "\033[1m" if COLOR else ""
RESET = "\033[0m" if COLOR else ""


def _status_token(status: str) -> str:
    if status == "PASS":
        return f"{GREEN}PASS{RESET}"
    if status == "WARN":
        return f"{YELLOW}WARN{RESET}"
    if status == "FAIL":
        return f"{RED}FAIL{RESET}"
    if status == "SKIP":
        return f"{CYAN}SKIP{RESET}"
    return status


# ---------------------------------------------------------------------------
# Consolidated row
# ---------------------------------------------------------------------------
@dataclass
class Row:
    category: str
    name: str
    status: str  # PASS / WARN / FAIL / SKIP
    detail: str = ""


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_urdf(verbose: bool) -> List[Row]:
    rows: List[Row] = []
    print(f"{BOLD}{CYAN}>>> URDF inspection{RESET}")
    for f in urdf_mod.DEFAULT_URDF_FILES:
        report = urdf_mod.inspect_urdf(f)
        urdf_mod.print_report(report, verbose=verbose)
        if report.errors:
            status = "FAIL"
        elif report.warnings or report.joint_limit_issues:
            status = "WARN"
        else:
            status = "PASS"
        detail = (
            f"links={len(report.links)} joints={len(report.joints)} "
            f"missing_meshes={len(report.missing_meshes)}"
        )
        rows.append(Row("URDF", os.path.basename(f), status, detail))
    return rows


def run_mjcf(verbose: bool) -> List[Row]:
    rows: List[Row] = []
    print(f"{BOLD}{CYAN}>>> MJCF inspection{RESET}")
    for f in mjcf_mod.DEFAULT_MJCF_FILES:
        report = mjcf_mod.inspect_mjcf(f)
        mjcf_mod.print_report(report, verbose=verbose)
        if report.errors:
            status = "FAIL"
        elif report.warnings or report.actuator_issues:
            status = "WARN"
        else:
            status = "PASS"
        detail = (
            f"bodies={len(report.bodies)} joints={len(report.joints)} "
            f"missing_meshes={len(report.missing_meshes)}"
        )
        rows.append(Row("MJCF", os.path.basename(f), status, detail))
    return rows


def run_usd(verbose: bool, include_subfiles: bool = True) -> List[Row]:
    rows: List[Row] = []
    print(f"{BOLD}{CYAN}>>> USD inspection{RESET}")
    if usd_mod is None or not getattr(usd_mod, "PXR_AVAILABLE", False):
        msg = USD_IMPORT_ERROR or "pxr unavailable"
        print(
            f"{YELLOW}USD inspection skipped: {msg}{RESET}\n"
            f"Run with: conda run -n isaaclab python inspect_all.py\n"
        )
        rows.append(Row("USD", "(all)", "SKIP", msg))
        return rows

    files: List[str] = list(usd_mod.DEFAULT_USD_MAIN_FILES)
    if include_subfiles:
        files += list(usd_mod.DEFAULT_USD_SUBFILES)
    for f in files:
        report = usd_mod.inspect_usd(f)
        usd_mod.print_report(report, verbose=verbose)
        if report.errors:
            status = "FAIL"
        elif report.warnings:
            status = "WARN"
        else:
            status = "PASS"
        detail = (
            f"prims={report.prim_count} joints={len(report.joints)} "
            f"missing={len(report.missing_assets)}"
        )
        rows.append(Row("USD", os.path.basename(f), status, detail))
    return rows


def run_meshes(verbose: bool) -> List[Row]:
    rows: List[Row] = []
    print(f"{BOLD}{CYAN}>>> Mesh inspection{RESET}")
    if not mesh_mod.TRIMESH_AVAILABLE:
        msg = mesh_mod.TRIMESH_IMPORT_ERROR or "trimesh unavailable"
        print(f"{YELLOW}Mesh inspection skipped: {msg}{RESET}\n")
        rows.append(Row("MESH", "(all)", "SKIP", msg))
        return rows

    for label, d in mesh_mod.DEFAULT_MESH_DIRS:
        group = mesh_mod.inspect_directory(label, d)
        mesh_mod.print_group(group, verbose=verbose)
        if not os.path.isdir(d) or not group.meshes:
            status = "SKIP"
        elif any(m.errors for m in group.meshes):
            status = "FAIL"
        elif any(m.warnings for m in group.meshes):
            status = "WARN"
        else:
            status = "PASS"
        detail = (
            f"files={len(group.meshes)} "
            f"verts={group.total_vertices} faces={group.total_faces}"
        )
        rows.append(Row("MESH", label, status, detail))
    return rows


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def print_summary(rows: List[Row]) -> bool:
    bar = "=" * 78
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"{BOLD}DR02 inspection summary{RESET}")
    print(f"{BOLD}{bar}{RESET}")
    print(
        f"{'category':<8}{'item':<40}{'status':<14}{'detail'}"
    )
    print("-" * 78)
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        print(
            f"{row.category:<8}{row.name:<40}"
            f"{_status_token(row.status):<24}{row.detail}"
        )
    print("-" * 78)
    print(
        f"PASS={counts.get('PASS',0)}  WARN={counts.get('WARN',0)}  "
        f"FAIL={counts.get('FAIL',0)}  SKIP={counts.get('SKIP',0)}"
    )
    return counts.get("FAIL", 0) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", action="store_true", help="Run URDF checks.")
    parser.add_argument("--mjcf", action="store_true", help="Run MJCF checks.")
    parser.add_argument("--usd", action="store_true", help="Run USD checks.")
    parser.add_argument("--mesh", action="store_true", help="Run mesh checks.")
    parser.add_argument("--all", action="store_true",
                        help="Run every check (default when no flag given).")
    parser.add_argument("--no-usd-subfiles", action="store_true",
                        help="Skip the USD configuration sub-files.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print summaries only.")
    args = parser.parse_args()

    none_selected = not (args.urdf or args.mjcf or args.usd or args.mesh)
    run_all = args.all or none_selected
    verbose = not args.quiet

    rows: List[Row] = []
    if run_all or args.urdf:
        rows.extend(run_urdf(verbose))
    if run_all or args.mjcf:
        rows.extend(run_mjcf(verbose))
    if run_all or args.usd:
        rows.extend(run_usd(verbose, include_subfiles=not args.no_usd_subfiles))
    if run_all or args.mesh:
        rows.extend(run_meshes(verbose))

    ok = print_summary(rows)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
