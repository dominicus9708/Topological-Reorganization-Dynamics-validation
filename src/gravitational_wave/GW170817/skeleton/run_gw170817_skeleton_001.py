#!/usr/bin/env python3
from __future__ import annotations

"""
run_gw170817_skeleton_001.py

Purpose
-------
Skeleton-stage validation pipeline for the GW170817 case.

Placement
---------
src/gravitational_wave/GW170817/skeleton/run_gw170817_skeleton_001.py

Input
-----
Reads prepared input files from:
  data/derived/gravitational waves/GW170817/input/

Expected input files
--------------------
- gw170817_h1_input_4khz_4096s.csv
- gw170817_l1_input_4khz_4096s.csv
- gw170817_v1_input_4khz_4096s.csv
- gw170817_h1_l1_v1_input_4khz_4096s.csv
- gw170817_event_summary_input.csv

Output
------
Writes timestamped skeleton results to:
  results/gravitational_wave/GW170817/output/skeleton/YYYYMMDD_HHMMSS/

Generated files
---------------
- gw170817_skeleton_h1_preview.csv
- gw170817_skeleton_l1_preview.csv
- gw170817_skeleton_v1_preview.csv
- gw170817_skeleton_merged_preview.csv
- gw170817_skeleton_summary_copy.csv
- gw170817_skeleton_report.txt
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    parser.add_argument("--h1-input", type=str, default="gw170817_h1_input_4khz_4096s.csv")
    parser.add_argument("--l1-input", type=str, default="gw170817_l1_input_4khz_4096s.csv")
    parser.add_argument("--v1-input", type=str, default="gw170817_v1_input_4khz_4096s.csv")
    parser.add_argument("--merged-input", type=str, default="gw170817_h1_l1_v1_input_4khz_4096s.csv")
    parser.add_argument("--summary-input", type=str, default="gw170817_event_summary_input.csv")
    parser.add_argument("--preview-rows", type=int, default=2000, help="Rows to save for preview CSV files.")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def get_summary_value(summary_df: pd.DataFrame, metric: str):
    mask = summary_df["metric"].astype(str) == metric
    if not mask.any():
        return None
    return summary_df.loc[mask, "value"].iloc[0]


def build_report(
    h1_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    v1_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("GW170817 skeleton report")
    lines.append("=======================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This is the skeleton stage for the GW170817 validation case. "
        "It checks that the H1/L1/V1 detector inputs and the merged input are readable, "
        "structurally consistent, and ready for later standard/topological processing."
    )
    lines.append("")
    lines.append("Input status")
    lines.append("------------")
    lines.append(f"H1 rows: {len(h1_df)}")
    lines.append(f"L1 rows: {len(l1_df)}")
    lines.append(f"V1 rows: {len(v1_df)}")
    lines.append(f"Merged rows: {len(merged_df)}")
    lines.append("")
    lines.append("Summary values")
    lines.append("--------------")
    for key in [
        "h1_sample_rate_hz",
        "l1_sample_rate_hz",
        "v1_sample_rate_hz",
        "h1_duration_s",
        "l1_duration_s",
        "v1_duration_s",
        "h1_gps_start",
        "l1_gps_start",
        "v1_gps_start",
        "h1_peak_time_centered_s",
        "l1_peak_time_centered_s",
        "v1_peak_time_centered_s",
        "merged_peak_time_centered_s",
        "h1_peak_abs_strain",
        "l1_peak_abs_strain",
        "v1_peak_abs_strain",
        "merged_peak_abs_mean_strain",
        "mean_abs_diff_h1_l1",
        "mean_abs_diff_h1_v1",
        "mean_abs_diff_l1_v1",
        "max_abs_diff_h1_l1",
        "max_abs_diff_h1_v1",
        "max_abs_diff_l1_v1",
    ]:
        value = get_summary_value(summary_df, key)
        if value is not None:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append(
        "The skeleton stage does not perform physical interpretation. "
        "Its purpose is to confirm that the three-detector GW170817 input bundle "
        "is consistently assembled and that later baseline / response stages can proceed."
    )
    lines.append("")
    lines.append("Output directory")
    lines.append("----------------")
    lines.append(str(output_dir))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    input_dir = project_root / "data" / "derived" / "gravitational waves" / "GW170817" / "input"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "gravitational_wave" / "GW170817" / "output" / "skeleton" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    h1_path = input_dir / args.h1_input
    l1_path = input_dir / args.l1_input
    v1_path = input_dir / args.v1_input
    merged_path = input_dir / args.merged_input
    summary_path = input_dir / args.summary_input

    for p, label in [
        (h1_path, "H1 input"),
        (l1_path, "L1 input"),
        (v1_path, "V1 input"),
        (merged_path, "Merged input"),
        (summary_path, "Summary input"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    h1_df = pd.read_csv(h1_path)
    l1_df = pd.read_csv(l1_path)
    v1_df = pd.read_csv(v1_path)
    merged_df = pd.read_csv(merged_path)
    summary_df = pd.read_csv(summary_path)

    ensure_columns(
        h1_df,
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "detector", "strain_abs", "strain_sign", "strain_sq"],
        "h1_df",
    )
    ensure_columns(
        l1_df,
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "detector", "strain_abs", "strain_sign", "strain_sq"],
        "l1_df",
    )
    ensure_columns(
        v1_df,
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "detector", "strain_abs", "strain_sign", "strain_sq"],
        "v1_df",
    )
    ensure_columns(
        merged_df,
        [
            "sample_index", "gps_time", "time_since_start_s", "time_centered_s",
            "strain_h1", "strain_l1", "strain_v1",
            "strain_mean_h1_l1", "strain_mean_h1_l1_v1",
            "strain_abs_mean_h1_l1", "strain_abs_mean_h1_l1_v1",
            "strain_abs_diff_h1_l1", "strain_abs_diff_h1_v1", "strain_abs_diff_l1_v1",
            "time_from_peak_s",
        ],
        "merged_df",
    )
    ensure_columns(summary_df, ["metric", "value", "note"], "summary_df")

    n = int(args.preview_rows)
    h1_preview = h1_df.head(n).copy()
    l1_preview = l1_df.head(n).copy()
    v1_preview = v1_df.head(n).copy()
    merged_preview = merged_df.head(n).copy()

    h1_out = output_dir / "gw170817_skeleton_h1_preview.csv"
    l1_out = output_dir / "gw170817_skeleton_l1_preview.csv"
    v1_out = output_dir / "gw170817_skeleton_v1_preview.csv"
    merged_out = output_dir / "gw170817_skeleton_merged_preview.csv"
    summary_out = output_dir / "gw170817_skeleton_summary_copy.csv"
    report_out = output_dir / "gw170817_skeleton_report.txt"

    h1_preview.to_csv(h1_out, index=False, encoding="utf-8-sig")
    l1_preview.to_csv(l1_out, index=False, encoding="utf-8-sig")
    v1_preview.to_csv(v1_out, index=False, encoding="utf-8-sig")
    merged_preview.to_csv(merged_out, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    report_out.write_text(build_report(h1_df, l1_df, v1_df, merged_df, summary_df, output_dir), encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"h1_out: {h1_out}")
    print(f"l1_out: {l1_out}")
    print(f"v1_out: {v1_out}")
    print(f"merged_out: {merged_out}")
    print(f"summary_out: {summary_out}")
    print(f"report_out: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
