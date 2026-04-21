#!/usr/bin/env python3
"""
build_mercury_input_001.py

Purpose
-------
Read derived Mercury CSV files from:
  data/derived/mercury/derived/

and write cleaned input files to:
  data/derived/mercury/input/

Expected input files
--------------------
- mercury_elements_2026.csv
- mercury_vectors_2026.csv
- mercury_elements_vectors_merged_2026.csv

Output files
------------
- mercury_elements_input_2026.csv
- mercury_vectors_input_2026.csv
- mercury_orbit_summary_input_2026.csv
- mercury_input_manifest_2026.txt
"""

from __future__ import annotations

import argparse
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
        "--elements-csv",
        type=str,
        default="mercury_elements_2026.csv",
        help="Elements CSV filename inside data/derived/mercury/derived/",
    )
    parser.add_argument(
        "--vectors-csv",
        type=str,
        default="mercury_vectors_2026.csv",
        help="Vectors CSV filename inside data/derived/mercury/derived/",
    )
    parser.add_argument(
        "--merged-csv",
        type=str,
        default="mercury_elements_vectors_merged_2026.csv",
        help="Merged CSV filename inside data/derived/mercury/derived/",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    derived_dir = project_root / "data" / "derived" / "mercury" / "derived"
    input_dir = project_root / "data" / "derived" / "mercury" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    elements_path = derived_dir / args.elements_csv
    vectors_path = derived_dir / args.vectors_csv
    merged_path = derived_dir / args.merged_csv

    if not elements_path.exists():
        raise FileNotFoundError(f"Elements CSV not found: {elements_path}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors CSV not found: {vectors_path}")
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged CSV not found: {merged_path}")

    elements_df = pd.read_csv(elements_path)
    vectors_df = pd.read_csv(vectors_path)
    merged_df = pd.read_csv(merged_path)

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
        merged_df,
        ["JDTDB", "Calendar Date (TDB)", "EC", "QR", "A", "AD", "PR", "W", "OM", "MA", "TA", "R_AU", "V_AU_PER_DAY"],
        "merged_df",
    )

    # Clean / reorder elements input
    elements_input = elements_df[
        ["JDTDB", "Calendar Date (TDB)", "EC", "QR", "A", "AD", "PR", "W", "OM", "MA", "TA"]
    ].copy()
    elements_input = elements_input.sort_values("JDTDB").reset_index(drop=True)

    # Clean / reorder vectors input
    vectors_input = vectors_df[
        ["JDTDB", "Calendar Date (TDB)", "X", "Y", "Z", "VX", "VY", "VZ", "R_AU", "V_AU_PER_DAY"]
    ].copy()
    vectors_input = vectors_input.sort_values("JDTDB").reset_index(drop=True)

    # Build orbit summary input for weak-field consistency use
    idx_min_r = merged_df["R_AU"].astype(float).idxmin()
    idx_max_r = merged_df["R_AU"].astype(float).idxmax()

    peri_row = merged_df.loc[idx_min_r]
    aphe_row = merged_df.loc[idx_max_r]

    summary = pd.DataFrame(
        [
            {
                "metric": "elements_rows",
                "value": len(elements_input),
                "note": "Number of rows in cleaned elements input.",
            },
            {
                "metric": "vectors_rows",
                "value": len(vectors_input),
                "note": "Number of rows in cleaned vectors input.",
            },
            {
                "metric": "mean_eccentricity",
                "value": float(elements_input["EC"].astype(float).mean()),
                "note": "Mean eccentricity across 2026 elements rows.",
            },
            {
                "metric": "mean_semi_major_axis_au",
                "value": float(elements_input["A"].astype(float).mean()),
                "note": "Mean semi-major axis across 2026 elements rows.",
            },
            {
                "metric": "mean_sidereal_period_day",
                "value": float(elements_input["PR"].astype(float).mean()),
                "note": "Mean sidereal period across 2026 elements rows.",
            },
            {
                "metric": "min_radius_au",
                "value": float(vectors_input["R_AU"].astype(float).min()),
                "note": f"Minimum Sun-Mercury distance from vectors; date={peri_row['Calendar Date (TDB)']}",
            },
            {
                "metric": "max_radius_au",
                "value": float(vectors_input["R_AU"].astype(float).max()),
                "note": f"Maximum Sun-Mercury distance from vectors; date={aphe_row['Calendar Date (TDB)']}",
            },
            {
                "metric": "perihelion_like_date",
                "value": peri_row["Calendar Date (TDB)"],
                "note": "Date associated with minimum R_AU in merged data.",
            },
            {
                "metric": "aphelion_like_date",
                "value": aphe_row["Calendar Date (TDB)"],
                "note": "Date associated with maximum R_AU in merged data.",
            },
            {
                "metric": "perihelion_like_argument_of_perifocus_deg",
                "value": float(peri_row["W"]),
                "note": "Argument of perifocus at perihelion-like date from merged data.",
            },
            {
                "metric": "aphelion_like_argument_of_perifocus_deg",
                "value": float(aphe_row["W"]),
                "note": "Argument of perifocus at aphelion-like date from merged data.",
            },
        ]
    )

    elements_out = input_dir / "mercury_elements_input_2026.csv"
    vectors_out = input_dir / "mercury_vectors_input_2026.csv"
    summary_out = input_dir / "mercury_orbit_summary_input_2026.csv"
    manifest_out = input_dir / "mercury_input_manifest_2026.txt"

    elements_input.to_csv(elements_out, index=False, encoding="utf-8-sig")
    vectors_input.to_csv(vectors_out, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    manifest_lines = [
        "Mercury input manifest",
        "=====================",
        "",
        f"elements_input_rows: {len(elements_input)}",
        f"vectors_input_rows: {len(vectors_input)}",
        "",
        f"elements_input_file: {elements_out}",
        f"vectors_input_file: {vectors_out}",
        f"summary_input_file: {summary_out}",
        "",
        f"mean_eccentricity: {float(elements_input['EC'].astype(float).mean()):.15f}",
        f"mean_semi_major_axis_au: {float(elements_input['A'].astype(float).mean()):.15f}",
        f"mean_sidereal_period_day: {float(elements_input['PR'].astype(float).mean()):.15f}",
        f"min_radius_au: {float(vectors_input['R_AU'].astype(float).min()):.15f}",
        f"max_radius_au: {float(vectors_input['R_AU'].astype(float).max()):.15f}",
        f"perihelion_like_date: {peri_row['Calendar Date (TDB)']}",
        f"aphelion_like_date: {aphe_row['Calendar Date (TDB)']}",
    ]
    manifest_out.write_text("\n".join(manifest_lines), encoding="utf-8")

    print("Done.")
    print(f"elements_out: {elements_out}")
    print(f"vectors_out: {vectors_out}")
    print(f"summary_out: {summary_out}")
    print(f"manifest_out: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
