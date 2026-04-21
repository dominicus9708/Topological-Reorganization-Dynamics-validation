#!/usr/bin/env python3
"""
run_gw150914_standard_001.py

Purpose
-------
Standard baseline pipeline for the GW150914 validation case.

Placement
---------
src/gravitational_wave/GW150914/standard/run_gw150914_standard_001.py

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
Writes timestamped standard results to:
  results/gravitational_wave/GW150914/output/standard/YYYYMMDD_HHMMSS/

Generated files
---------------
- gw150914_standard_h1_full.csv
- gw150914_standard_l1_full.csv
- gw150914_standard_merged_full.csv
- gw150914_standard_baseline_summary.csv
- gw150914_standard_report.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    parser.add_argument("--h1-input", type=str, default="gw150914_h1_input_4khz_32s.csv")
    parser.add_argument("--l1-input", type=str, default="gw150914_l1_input_4khz_32s.csv")
    parser.add_argument("--merged-input", type=str, default="gw150914_h1_l1_input_4khz_32s.csv")
    parser.add_argument("--summary-input", type=str, default="gw150914_event_summary_input.csv")
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


def build_baseline_summary(
    h1_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    h1_peak_idx = h1_df["strain_abs"].astype(float).idxmax()
    l1_peak_idx = l1_df["strain_abs"].astype(float).idxmax()
    merged_peak_idx = merged_df["strain_abs_mean_h1_l1"].astype(float).idxmax()

    h1_peak_row = h1_df.loc[h1_peak_idx]
    l1_peak_row = l1_df.loc[l1_peak_idx]
    merged_peak_row = merged_df.loc[merged_peak_idx]

    rows = [
        {"metric": "h1_rows", "value": len(h1_df), "note": "Number of rows in H1 baseline."},
        {"metric": "l1_rows", "value": len(l1_df), "note": "Number of rows in L1 baseline."},
        {"metric": "merged_rows", "value": len(merged_df), "note": "Number of rows in merged baseline."},
        {"metric": "h1_sample_rate_hz", "value": get_summary_value(summary_df, "h1_sample_rate_hz"), "note": "H1 sample rate copied from input summary."},
        {"metric": "l1_sample_rate_hz", "value": get_summary_value(summary_df, "l1_sample_rate_hz"), "note": "L1 sample rate copied from input summary."},
        {"metric": "h1_peak_time_centered_s", "value": float(h1_peak_row["time_centered_s"]), "note": "Time-centered coordinate of max absolute H1 strain."},
        {"metric": "l1_peak_time_centered_s", "value": float(l1_peak_row["time_centered_s"]), "note": "Time-centered coordinate of max absolute L1 strain."},
        {"metric": "merged_peak_time_centered_s", "value": float(merged_peak_row["time_centered_s"]), "note": "Time-centered coordinate of max absolute mean(H1,L1) strain."},
        {"metric": "h1_peak_abs_strain", "value": float(h1_peak_row["strain_abs"]), "note": "Maximum absolute H1 strain."},
        {"metric": "l1_peak_abs_strain", "value": float(l1_peak_row["strain_abs"]), "note": "Maximum absolute L1 strain."},
        {"metric": "merged_peak_abs_mean_strain", "value": float(merged_peak_row["strain_abs_mean_h1_l1"]), "note": "Maximum absolute mean(H1,L1) strain."},
        {"metric": "h1_strain_std", "value": float(h1_df["strain"].astype(float).std()), "note": "Standard deviation of H1 strain."},
        {"metric": "l1_strain_std", "value": float(l1_df["strain"].astype(float).std()), "note": "Standard deviation of L1 strain."},
        {"metric": "merged_mean_strain_std", "value": float(merged_df["strain_mean_h1_l1"].astype(float).std()), "note": "Standard deviation of merged mean strain."},
        {"metric": "h1_l1_peak_time_offset_s", "value": abs(float(h1_peak_row["time_centered_s"]) - float(l1_peak_row["time_centered_s"])), "note": "Absolute difference between H1 and L1 peak times."},
        {"metric": "mean_abs_detector_difference", "value": float(merged_df["strain_abs_diff_h1_l1"].astype(float).mean()), "note": "Mean absolute difference |H1-L1| over the event window."},
        {"metric": "max_abs_detector_difference", "value": float(merged_df["strain_abs_diff_h1_l1"].astype(float).max()), "note": "Maximum absolute difference |H1-L1| over the event window."},
    ]
    return pd.DataFrame(rows)


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> str:
    m = {str(r["metric"]): r["value"] for _, r in summary_df.iterrows()}

    lines: list[str] = []
    lines.append("GW150914 standard report")
    lines.append("=======================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This is the standard baseline stage for the GW150914 validation case. "
        "It preserves the ordinary detector strain baseline without introducing any topological correction term."
    )
    lines.append("")
    lines.append("Baseline status")
    lines.append("---------------")
    for k in [
        "h1_rows", "l1_rows", "merged_rows",
        "h1_sample_rate_hz", "l1_sample_rate_hz",
        "h1_peak_time_centered_s", "l1_peak_time_centered_s", "merged_peak_time_centered_s",
        "h1_peak_abs_strain", "l1_peak_abs_strain", "merged_peak_abs_mean_strain",
        "h1_strain_std", "l1_strain_std", "merged_mean_strain_std",
        "h1_l1_peak_time_offset_s", "mean_abs_detector_difference", "max_abs_detector_difference",
    ]:
        if k in m:
            lines.append(f"{k}: {m[k]}")
    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append(
        "The standard stage treats GW150914 as an ordinary detector-baseline waveform case. "
        "Its role is to preserve the observed strain structure, peak timing, and detector-level difference "
        "before any structural propagation interpretation is introduced."
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
    output_dir = project_root / "results" / "gravitational_wave" / "GW150914" / "output" / "standard" / timestamp
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
            "sample_index", "gps_time", "time_since_start_s", "time_centered_s",
            "strain_h1", "strain_l1", "strain_mean_h1_l1", "strain_diff_h1_l1",
            "strain_abs_mean_h1_l1", "strain_abs_diff_h1_l1", "time_from_peak_s",
        ],
        "merged_df",
    )
    ensure_columns(summary_df, ["metric", "value", "note"], "summary_df")

    baseline_summary_df = build_baseline_summary(h1_df, l1_df, merged_df, summary_df)

    h1_out = output_dir / "gw150914_standard_h1_full.csv"
    l1_out = output_dir / "gw150914_standard_l1_full.csv"
    merged_out = output_dir / "gw150914_standard_merged_full.csv"
    summary_out = output_dir / "gw150914_standard_baseline_summary.csv"
    report_out = output_dir / "gw150914_standard_report.txt"

    h1_df.to_csv(h1_out, index=False, encoding="utf-8-sig")
    l1_df.to_csv(l1_out, index=False, encoding="utf-8-sig")
    merged_df.to_csv(merged_out, index=False, encoding="utf-8-sig")
    baseline_summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    report_out.write_text(build_report(baseline_summary_df, output_dir), encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"h1_out: {h1_out}")
    print(f"l1_out: {l1_out}")
    print(f"merged_out: {merged_out}")
    print(f"summary_out: {summary_out}")
    print(f"report_out: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
