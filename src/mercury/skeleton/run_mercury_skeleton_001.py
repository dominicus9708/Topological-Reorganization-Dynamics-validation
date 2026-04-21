#!/usr/bin/env python3
"""
run_mercury_skeleton_001.py

Purpose
-------
Minimal skeleton pipeline for the Mercury validation case.

Placement
---------
src/mercury/skeleton/run_mercury_skeleton_001.py

Input
-----
Reads prepared input files from:
  data/derived/mercury/input/

Expected input files
--------------------
- mercury_elements_input_2026.csv
- mercury_vectors_input_2026.csv
- mercury_orbit_summary_input_2026.csv

Output
------
Writes timestamped skeleton results to:
  results/mercury/output/skeleton/YYYYMMDD_HHMMSS/

Generated files
---------------
- mercury_skeleton_elements_preview.csv
- mercury_skeleton_vectors_preview.csv
- mercury_skeleton_summary_copy.csv
- mercury_skeleton_report.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=str,
        required=True,
        help="Path to the project root.",
    )
    parser.add_argument(
        "--elements-input",
        type=str,
        default="mercury_elements_input_2026.csv",
        help="Elements input filename inside data/derived/mercury/input/",
    )
    parser.add_argument(
        "--vectors-input",
        type=str,
        default="mercury_vectors_input_2026.csv",
        help="Vectors input filename inside data/derived/mercury/input/",
    )
    parser.add_argument(
        "--summary-input",
        type=str,
        default="mercury_orbit_summary_input_2026.csv",
        help="Summary input filename inside data/derived/mercury/input/",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="Number of preview rows to save for elements and vectors CSV.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def build_report(
    elements_df: pd.DataFrame,
    vectors_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("Mercury skeleton report")
    lines.append("======================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This is a skeleton-stage execution report. It verifies that the prepared Mercury "
        "input files can be loaded successfully and that a timestamped results folder can be created."
    )
    lines.append("")
    lines.append("Input status")
    lines.append("------------")
    lines.append(f"elements_rows: {len(elements_df)}")
    lines.append(f"vectors_rows: {len(vectors_df)}")
    lines.append(f"summary_rows: {len(summary_df)}")
    lines.append("")

    if not elements_df.empty:
        lines.append(f"elements_first_date: {elements_df.iloc[0]['Calendar Date (TDB)']}")
        lines.append(f"elements_last_date: {elements_df.iloc[-1]['Calendar Date (TDB)']}")
    if not vectors_df.empty:
        lines.append(f"vectors_first_date: {vectors_df.iloc[0]['Calendar Date (TDB)']}")
        lines.append(f"vectors_last_date: {vectors_df.iloc[-1]['Calendar Date (TDB)']}")

    lines.append("")
    lines.append("Basic diagnostics")
    lines.append("-----------------")
    if "EC" in elements_df.columns:
        lines.append(f"mean_eccentricity: {elements_df['EC'].astype(float).mean():.15f}")
    if "A" in elements_df.columns:
        lines.append(f"mean_semi_major_axis_au: {elements_df['A'].astype(float).mean():.15f}")
    if "PR" in elements_df.columns:
        lines.append(f"mean_sidereal_period_day: {elements_df['PR'].astype(float).mean():.15f}")
    if "R_AU" in vectors_df.columns:
        lines.append(f"min_radius_au: {vectors_df['R_AU'].astype(float).min():.15f}")
        lines.append(f"max_radius_au: {vectors_df['R_AU'].astype(float).max():.15f}")
    if "V_AU_PER_DAY" in vectors_df.columns:
        lines.append(f"min_speed_au_per_day: {vectors_df['V_AU_PER_DAY'].astype(float).min():.15f}")
        lines.append(f"max_speed_au_per_day: {vectors_df['V_AU_PER_DAY'].astype(float).max():.15f}")

    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append(
        "This run does not perform weak-field fitting or perihelion-advance reconstruction. "
        "It only confirms that the Mercury case has reached a reproducible skeleton stage."
    )
    lines.append("")
    lines.append("Output directory")
    lines.append("----------------")
    lines.append(str(output_dir))

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    input_dir = project_root / "data" / "derived" / "mercury" / "input"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "mercury" / "output" / "skeleton" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    elements_path = input_dir / args.elements_input
    vectors_path = input_dir / args.vectors_input
    summary_path = input_dir / args.summary_input

    if not elements_path.exists():
        raise FileNotFoundError(f"Elements input not found: {elements_path}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors input not found: {vectors_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary input not found: {summary_path}")

    elements_df = pd.read_csv(elements_path)
    vectors_df = pd.read_csv(vectors_path)
    summary_df = pd.read_csv(summary_path)

    ensure_columns(
        elements_df,
        ["JDTDB", "Calendar Date (TDB)", "EC", "QR", "A", "AD", "PR", "W", "OM", "MA", "TA"],
        "elements_df",
    )
    ensure_columns(
        vectors_df,
        ["JDTDB", "Calendar Date (TDB)", "X", "Y", "Z", "VX", "VY", "VZ", "R_AU", "V_AU_PER_DAY"],
        "vectors_df",
    )
    ensure_columns(
        summary_df,
        ["metric", "value", "note"],
        "summary_df",
    )

    elements_preview = elements_df.head(args.preview_rows).copy()
    vectors_preview = vectors_df.head(args.preview_rows).copy()

    elements_preview_path = output_dir / "mercury_skeleton_elements_preview.csv"
    vectors_preview_path = output_dir / "mercury_skeleton_vectors_preview.csv"
    summary_copy_path = output_dir / "mercury_skeleton_summary_copy.csv"
    report_path = output_dir / "mercury_skeleton_report.txt"

    elements_preview.to_csv(elements_preview_path, index=False, encoding="utf-8-sig")
    vectors_preview.to_csv(vectors_preview_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_copy_path, index=False, encoding="utf-8-sig")

    report_text = build_report(
        elements_df=elements_df,
        vectors_df=vectors_df,
        summary_df=summary_df,
        output_dir=output_dir,
    )
    report_path.write_text(report_text, encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"elements_preview: {elements_preview_path}")
    print(f"vectors_preview: {vectors_preview_path}")
    print(f"summary_copy: {summary_copy_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
