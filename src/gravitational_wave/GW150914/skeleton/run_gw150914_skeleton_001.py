#!/usr/bin/env python3
"""
run_gw150914_skeleton_001.py

Purpose
-------
Minimal skeleton pipeline for the GW150914 validation case.

Placement
---------
src/gravitational_wave/GW150914/skeleton/run_gw150914_skeleton_001.py

Input
-----
Reads prepared input files from:
  data/derived/gravitational waves/GW150914/input/

Expected input files
--------------------
- gw150914_h1_input_4khz_32s.csv
- gw150914_l1_input_4khz_32s.csv
- gw150914_h1_l1_input_4khz_32s.csv
- gw150914_event_summary_input.csv

Output
------
Writes timestamped skeleton results to:
  results/gravitational_wave/GW150914/output/skeleton/YYYYMMDD_HHMMSS/

Generated files
---------------
- gw150914_skeleton_h1_preview.csv
- gw150914_skeleton_l1_preview.csv
- gw150914_skeleton_merged_preview.csv
- gw150914_skeleton_summary_copy.csv
- gw150914_skeleton_report.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to the project root.")
    parser.add_argument("--h1-input", type=str, default="gw150914_h1_input_4khz_32s.csv")
    parser.add_argument("--l1-input", type=str, default="gw150914_l1_input_4khz_32s.csv")
    parser.add_argument("--merged-input", type=str, default="gw150914_h1_l1_input_4khz_32s.csv")
    parser.add_argument("--summary-input", type=str, default="gw150914_event_summary_input.csv")
    parser.add_argument("--preview-rows", type=int, default=20)
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
    merged_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("GW150914 skeleton report")
    lines.append("========================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This is a skeleton-stage execution report. It verifies that the prepared GW150914 "
        "input files can be loaded successfully and that a timestamped results folder can be created."
    )
    lines.append("")
    lines.append("Input status")
    lines.append("------------")
    lines.append(f"h1_rows: {len(h1_df)}")
    lines.append(f"l1_rows: {len(l1_df)}")
    lines.append(f"merged_rows: {len(merged_df)}")
    lines.append(f"summary_rows: {len(summary_df)}")
    lines.append("")
    lines.append("Basic diagnostics")
    lines.append("-----------------")
    h1_peak = get_summary_value(summary_df, "h1_peak_abs_strain")
    l1_peak = get_summary_value(summary_df, "l1_peak_abs_strain")
    merged_peak = get_summary_value(summary_df, "merged_peak_abs_mean_strain")
    h1_peak_t = get_summary_value(summary_df, "h1_peak_time_centered_s")
    l1_peak_t = get_summary_value(summary_df, "l1_peak_time_centered_s")
    merged_peak_t = get_summary_value(summary_df, "merged_peak_time_centered_s")
    h1_sr = get_summary_value(summary_df, "h1_sample_rate_hz")
    l1_sr = get_summary_value(summary_df, "l1_sample_rate_hz")

    if h1_sr is not None:
        lines.append(f"h1_sample_rate_hz: {h1_sr}")
    if l1_sr is not None:
        lines.append(f"l1_sample_rate_hz: {l1_sr}")
    if h1_peak is not None:
        lines.append(f"h1_peak_abs_strain: {h1_peak}")
    if l1_peak is not None:
        lines.append(f"l1_peak_abs_strain: {l1_peak}")
    if merged_peak is not None:
        lines.append(f"merged_peak_abs_mean_strain: {merged_peak}")
    if h1_peak_t is not None:
        lines.append(f"h1_peak_time_centered_s: {h1_peak_t}")
    if l1_peak_t is not None:
        lines.append(f"l1_peak_time_centered_s: {l1_peak_t}")
    if merged_peak_t is not None:
        lines.append(f"merged_peak_time_centered_s: {merged_peak_t}")

    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append(
        "This run does not yet perform waveform standardization, topological response modeling, "
        "or integration graphics. It only confirms that the GW150914 case has reached a reproducible "
        "skeleton stage."
    )
    lines.append("")
    lines.append("Output directory")
    lines.append("----------------")
    lines.append(str(output_dir))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    input_dir = project_root / "data" / "derived" / "gravitational waves" / "GW150914" / "input"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "gravitational_wave" / "GW150914" / "output" / "skeleton" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    h1_path = input_dir / args.h1_input
    l1_path = input_dir / args.l1_input
    merged_path = input_dir / args.merged_input
    summary_path = input_dir / args.summary_input

    for p, label in [
        (h1_path, "H1 input"),
        (l1_path, "L1 input"),
        (merged_path, "Merged input"),
        (summary_path, "Summary input"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    h1_df = pd.read_csv(h1_path)
    l1_df = pd.read_csv(l1_path)
    merged_df = pd.read_csv(merged_path)
    summary_df = pd.read_csv(summary_path)

    ensure_columns(
        h1_df,
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "strain_abs", "strain_sign", "strain_sq", "detector"],
        "h1_df",
    )
    ensure_columns(
        l1_df,
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "strain_abs", "strain_sign", "strain_sq", "detector"],
        "l1_df",
    )
    ensure_columns(
        merged_df,
        [
            "sample_index",
            "gps_time",
            "time_since_start_s",
            "time_centered_s",
            "strain_h1",
            "strain_l1",
            "strain_mean_h1_l1",
            "strain_diff_h1_l1",
            "strain_abs_mean_h1_l1",
            "strain_abs_diff_h1_l1",
            "time_from_peak_s",
        ],
        "merged_df",
    )
    ensure_columns(summary_df, ["metric", "value", "note"], "summary_df")

    h1_preview = h1_df.head(args.preview_rows).copy()
    l1_preview = l1_df.head(args.preview_rows).copy()
    merged_preview = merged_df.head(args.preview_rows).copy()

    h1_preview_path = output_dir / "gw150914_skeleton_h1_preview.csv"
    l1_preview_path = output_dir / "gw150914_skeleton_l1_preview.csv"
    merged_preview_path = output_dir / "gw150914_skeleton_merged_preview.csv"
    summary_copy_path = output_dir / "gw150914_skeleton_summary_copy.csv"
    report_path = output_dir / "gw150914_skeleton_report.txt"

    h1_preview.to_csv(h1_preview_path, index=False, encoding="utf-8-sig")
    l1_preview.to_csv(l1_preview_path, index=False, encoding="utf-8-sig")
    merged_preview.to_csv(merged_preview_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_copy_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(h1_df, l1_df, merged_df, summary_df, output_dir), encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"h1_preview: {h1_preview_path}")
    print(f"l1_preview: {l1_preview_path}")
    print(f"merged_preview: {merged_preview_path}")
    print(f"summary_copy: {summary_copy_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
