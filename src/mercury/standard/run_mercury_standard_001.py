#!/usr/bin/env python3
"""
run_mercury_standard_001.py

Purpose
-------
Standard baseline pipeline for the Mercury validation case.

Placement
---------
src/mercury/standard/run_mercury_standard_001.py

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
Writes timestamped standard results to:
  results/mercury/output/standard/YYYYMMDD_HHMMSS/

Generated files
---------------
- mercury_standard_elements_full.csv
- mercury_standard_vectors_full.csv
- mercury_standard_baseline_summary.csv
- mercury_standard_report.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    parser.add_argument("--elements-input", type=str, default="mercury_elements_input_2026.csv")
    parser.add_argument("--vectors-input", type=str, default="mercury_vectors_input_2026.csv")
    parser.add_argument("--summary-input", type=str, default="mercury_orbit_summary_input_2026.csv")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def build_baseline_summary(elements_df: pd.DataFrame, vectors_df: pd.DataFrame) -> pd.DataFrame:
    elements_df = elements_df.sort_values("JDTDB").reset_index(drop=True)
    vectors_df = vectors_df.sort_values("JDTDB").reset_index(drop=True)

    idx_min_r = vectors_df["R_AU"].astype(float).idxmin()
    idx_max_r = vectors_df["R_AU"].astype(float).idxmax()

    peri_vec = vectors_df.loc[idx_min_r]
    aphe_vec = vectors_df.loc[idx_max_r]

    mean_e = float(elements_df["EC"].astype(float).mean())
    mean_a = float(elements_df["A"].astype(float).mean())
    mean_pr = float(elements_df["PR"].astype(float).mean())
    min_r = float(vectors_df["R_AU"].astype(float).min())
    max_r = float(vectors_df["R_AU"].astype(float).max())
    min_v = float(vectors_df["V_AU_PER_DAY"].astype(float).min())
    max_v = float(vectors_df["V_AU_PER_DAY"].astype(float).max())

    q_mean = float(elements_df["QR"].astype(float).mean())
    ad_mean = float(elements_df["AD"].astype(float).mean())

    q_vs_rmin_abs = abs(q_mean - min_r)
    ad_vs_rmax_abs = abs(ad_mean - max_r)

    rows = [
        {"metric": "elements_rows", "value": len(elements_df), "note": "Number of elements input rows."},
        {"metric": "vectors_rows", "value": len(vectors_df), "note": "Number of vectors input rows."},
        {"metric": "mean_eccentricity", "value": mean_e, "note": "Mean eccentricity from elements baseline."},
        {"metric": "mean_semi_major_axis_au", "value": mean_a, "note": "Mean semi-major axis from elements baseline."},
        {"metric": "mean_sidereal_period_day", "value": mean_pr, "note": "Mean sidereal period from elements baseline."},
        {"metric": "mean_periapsis_distance_qr_au", "value": q_mean, "note": "Mean periapsis distance QR from elements baseline."},
        {"metric": "mean_apoapsis_distance_ad_au", "value": ad_mean, "note": "Mean apoapsis distance AD from elements baseline."},
        {"metric": "min_radius_au", "value": min_r, "note": f"Minimum vector radius; date={peri_vec['Calendar Date (TDB)']}"},
        {"metric": "max_radius_au", "value": max_r, "note": f"Maximum vector radius; date={aphe_vec['Calendar Date (TDB)']}"},
        {"metric": "min_speed_au_per_day", "value": min_v, "note": "Minimum vector speed."},
        {"metric": "max_speed_au_per_day", "value": max_v, "note": "Maximum vector speed."},
        {"metric": "perihelion_like_date", "value": peri_vec["Calendar Date (TDB)"], "note": "Date at minimum vector radius."},
        {"metric": "aphelion_like_date", "value": aphe_vec["Calendar Date (TDB)"], "note": "Date at maximum vector radius."},
        {"metric": "periapsis_vs_min_radius_abs_diff_au", "value": q_vs_rmin_abs, "note": "Absolute difference between mean QR and vector min radius."},
        {"metric": "apoapsis_vs_max_radius_abs_diff_au", "value": ad_vs_rmax_abs, "note": "Absolute difference between mean AD and vector max radius."},
    ]
    return pd.DataFrame(rows)


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> str:
    m = {str(r["metric"]): r["value"] for _, r in summary_df.iterrows()}

    lines: list[str] = []
    lines.append("Mercury standard report")
    lines.append("======================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This is the standard baseline stage for the Mercury validation case. "
        "It does not introduce any topological correction term. "
        "Its role is to preserve the prepared input and summarize the ordinary orbital baseline "
        "in a reproducible weak-field form."
    )
    lines.append("")
    lines.append("Baseline status")
    lines.append("---------------")
    lines.append(f"elements_rows: {m.get('elements_rows')}")
    lines.append(f"vectors_rows: {m.get('vectors_rows')}")
    lines.append(f"mean_eccentricity: {float(m.get('mean_eccentricity')):.15f}")
    lines.append(f"mean_semi_major_axis_au: {float(m.get('mean_semi_major_axis_au')):.15f}")
    lines.append(f"mean_sidereal_period_day: {float(m.get('mean_sidereal_period_day')):.15f}")
    lines.append(f"mean_periapsis_distance_qr_au: {float(m.get('mean_periapsis_distance_qr_au')):.15f}")
    lines.append(f"mean_apoapsis_distance_ad_au: {float(m.get('mean_apoapsis_distance_ad_au')):.15f}")
    lines.append(f"min_radius_au: {float(m.get('min_radius_au')):.15f}")
    lines.append(f"max_radius_au: {float(m.get('max_radius_au')):.15f}")
    lines.append(f"min_speed_au_per_day: {float(m.get('min_speed_au_per_day')):.15f}")
    lines.append(f"max_speed_au_per_day: {float(m.get('max_speed_au_per_day')):.15f}")
    lines.append(f"perihelion_like_date: {m.get('perihelion_like_date')}")
    lines.append(f"aphelion_like_date: {m.get('aphelion_like_date')}")
    lines.append("")
    lines.append("Consistency notes")
    lines.append("-----------------")
    lines.append(
        "The standard stage treats Mercury as a weak-field orbital baseline with stable eccentric motion. "
        "The purpose is not to reconstruct the century-scale anomalous perihelion advance, "
        "but to preserve an ordinary orbital reference against which later interpretation may be compared."
    )
    lines.append(
        "The differences between mean element-based periapsis/apoapsis distances and vector-based "
        "minimum/maximum radius are expected to remain small at this baseline level."
    )
    lines.append(f"periapsis_vs_min_radius_abs_diff_au: {float(m.get('periapsis_vs_min_radius_abs_diff_au')):.15f}")
    lines.append(f"apoapsis_vs_max_radius_abs_diff_au: {float(m.get('apoapsis_vs_max_radius_abs_diff_au')):.15f}")
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
    output_dir = project_root / "results" / "mercury" / "output" / "standard" / timestamp
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
    prepared_summary_df = pd.read_csv(summary_path)

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
    ensure_columns(prepared_summary_df, ["metric", "value", "note"], "prepared_summary_df")

    baseline_summary_df = build_baseline_summary(elements_df, vectors_df)

    elements_out = output_dir / "mercury_standard_elements_full.csv"
    vectors_out = output_dir / "mercury_standard_vectors_full.csv"
    summary_out = output_dir / "mercury_standard_baseline_summary.csv"
    report_out = output_dir / "mercury_standard_report.txt"

    elements_df.to_csv(elements_out, index=False, encoding="utf-8-sig")
    vectors_df.to_csv(vectors_out, index=False, encoding="utf-8-sig")
    baseline_summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    report_out.write_text(build_report(baseline_summary_df, output_dir), encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"elements_out: {elements_out}")
    print(f"vectors_out: {vectors_out}")
    print(f"summary_out: {summary_out}")
    print(f"report_out: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
