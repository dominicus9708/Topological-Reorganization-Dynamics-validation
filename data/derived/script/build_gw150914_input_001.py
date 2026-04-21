#!/usr/bin/env python3
"""
build_gw150914_input_001.py

Purpose
-------
Read derived GW150914 CSV files from:
  data/derived/gravitational waves/GW150914/derived/

and write cleaned input files to:
  data/derived/gravitational waves/GW150914/input/

Placement
---------
data/derived/script/build_gw150914_input_001.py

Expected input files
--------------------
- gw150914_h1_strain_4khz_32s.csv
- gw150914_l1_strain_4khz_32s.csv
- gw150914_h1_l1_strain_merged_4khz_32s.csv
- gw150914_metadata_summary.csv

Output files
------------
- gw150914_h1_input_4khz_32s.csv
- gw150914_l1_input_4khz_32s.csv
- gw150914_h1_l1_input_4khz_32s.csv
- gw150914_event_summary_input.csv
- gw150914_input_manifest.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
        "--h1-csv",
        type=str,
        default="gw150914_h1_strain_4khz_32s.csv",
        help="H1 derived CSV filename inside data/derived/gravitational waves/GW150914/derived/",
    )
    parser.add_argument(
        "--l1-csv",
        type=str,
        default="gw150914_l1_strain_4khz_32s.csv",
        help="L1 derived CSV filename inside data/derived/gravitational waves/GW150914/derived/",
    )
    parser.add_argument(
        "--merged-csv",
        type=str,
        default="gw150914_h1_l1_strain_merged_4khz_32s.csv",
        help="Merged derived CSV filename inside data/derived/gravitational waves/GW150914/derived/",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="gw150914_metadata_summary.csv",
        help="Metadata summary CSV filename inside data/derived/gravitational waves/GW150914/derived/",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def extract_metadata_value(metadata_df: pd.DataFrame, detector_label: str, metric: str):
    mask = (
        metadata_df["detector_label"].astype(str).eq(detector_label)
        & metadata_df["metric"].astype(str).eq(metric)
    )
    if not mask.any():
        return None
    return metadata_df.loc[mask, "value"].iloc[0]


def build_detector_input(df: pd.DataFrame, detector_name: str) -> pd.DataFrame:
    out = df[
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain"]
    ].copy()
    out = out.sort_values("sample_index").reset_index(drop=True)

    out["detector"] = detector_name
    out["strain_abs"] = out["strain"].astype(float).abs()
    out["strain_sign"] = np.sign(out["strain"].astype(float))
    out["strain_sq"] = out["strain"].astype(float) ** 2
    return out


def build_merged_input(merged_df: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(
        merged_df,
        [
            "sample_index",
            "gps_time",
            "time_since_start_s",
            "time_centered_s",
            "strain_h1",
            "strain_l1",
        ],
        "merged_df",
    )
    out = merged_df[
        [
            "sample_index",
            "gps_time",
            "time_since_start_s",
            "time_centered_s",
            "strain_h1",
            "strain_l1",
        ]
    ].copy()
    out = out.sort_values("sample_index").reset_index(drop=True)

    out["strain_mean_h1_l1"] = (
        out["strain_h1"].astype(float) + out["strain_l1"].astype(float)
    ) / 2.0
    out["strain_diff_h1_l1"] = (
        out["strain_h1"].astype(float) - out["strain_l1"].astype(float)
    )
    out["strain_abs_mean_h1_l1"] = out["strain_mean_h1_l1"].abs()
    out["strain_abs_diff_h1_l1"] = out["strain_diff_h1_l1"].abs()

    peak_idx = out["strain_abs_mean_h1_l1"].astype(float).idxmax()
    peak_time = float(out.loc[peak_idx, "time_centered_s"])
    out["time_from_peak_s"] = out["time_centered_s"].astype(float) - peak_time
    return out


def build_event_summary(
    h1_input: pd.DataFrame,
    l1_input: pd.DataFrame,
    merged_input: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    h1_peak_idx = h1_input["strain_abs"].astype(float).idxmax()
    l1_peak_idx = l1_input["strain_abs"].astype(float).idxmax()
    merged_peak_idx = merged_input["strain_abs_mean_h1_l1"].astype(float).idxmax()

    h1_peak_row = h1_input.loc[h1_peak_idx]
    l1_peak_row = l1_input.loc[l1_peak_idx]
    merged_peak_row = merged_input.loc[merged_peak_idx]

    h1_sample_rate = extract_metadata_value(metadata_df, "H1", "sample_rate_hz")
    l1_sample_rate = extract_metadata_value(metadata_df, "L1", "sample_rate_hz")
    h1_duration = extract_metadata_value(metadata_df, "H1", "Duration")
    l1_duration = extract_metadata_value(metadata_df, "L1", "Duration")
    gps_start = extract_metadata_value(metadata_df, "H1", "GPSstart")

    rows = [
        {
            "metric": "h1_rows",
            "value": len(h1_input),
            "note": "Number of rows in H1 input.",
        },
        {
            "metric": "l1_rows",
            "value": len(l1_input),
            "note": "Number of rows in L1 input.",
        },
        {
            "metric": "merged_rows",
            "value": len(merged_input),
            "note": "Number of rows in merged input.",
        },
        {
            "metric": "gps_start",
            "value": gps_start,
            "note": "Detector GPS start from metadata.",
        },
        {
            "metric": "h1_sample_rate_hz",
            "value": h1_sample_rate,
            "note": "H1 sample rate from metadata.",
        },
        {
            "metric": "l1_sample_rate_hz",
            "value": l1_sample_rate,
            "note": "L1 sample rate from metadata.",
        },
        {
            "metric": "h1_duration_s",
            "value": h1_duration,
            "note": "H1 duration from metadata.",
        },
        {
            "metric": "l1_duration_s",
            "value": l1_duration,
            "note": "L1 duration from metadata.",
        },
        {
            "metric": "h1_peak_time_centered_s",
            "value": float(h1_peak_row["time_centered_s"]),
            "note": "Time-centered coordinate of maximum absolute H1 strain.",
        },
        {
            "metric": "l1_peak_time_centered_s",
            "value": float(l1_peak_row["time_centered_s"]),
            "note": "Time-centered coordinate of maximum absolute L1 strain.",
        },
        {
            "metric": "merged_peak_time_centered_s",
            "value": float(merged_peak_row["time_centered_s"]),
            "note": "Time-centered coordinate of maximum absolute mean(H1,L1) strain.",
        },
        {
            "metric": "h1_peak_abs_strain",
            "value": float(h1_peak_row["strain_abs"]),
            "note": "Maximum absolute H1 strain.",
        },
        {
            "metric": "l1_peak_abs_strain",
            "value": float(l1_peak_row["strain_abs"]),
            "note": "Maximum absolute L1 strain.",
        },
        {
            "metric": "merged_peak_abs_mean_strain",
            "value": float(merged_peak_row["strain_abs_mean_h1_l1"]),
            "note": "Maximum absolute mean(H1,L1) strain.",
        },
        {
            "metric": "h1_strain_std",
            "value": float(h1_input["strain"].astype(float).std()),
            "note": "Standard deviation of H1 strain.",
        },
        {
            "metric": "l1_strain_std",
            "value": float(l1_input["strain"].astype(float).std()),
            "note": "Standard deviation of L1 strain.",
        },
        {
            "metric": "merged_mean_strain_std",
            "value": float(merged_input["strain_mean_h1_l1"].astype(float).std()),
            "note": "Standard deviation of mean(H1,L1) strain.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    derived_dir = project_root / "data" / "derived" / "gravitational waves" / "GW150914" / "derived"
    input_dir = project_root / "data" / "derived" / "gravitational waves" / "GW150914" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    h1_path = derived_dir / args.h1_csv
    l1_path = derived_dir / args.l1_csv
    merged_path = derived_dir / args.merged_csv
    metadata_path = derived_dir / args.metadata_csv

    if not h1_path.exists():
        raise FileNotFoundError(f"H1 derived CSV not found: {h1_path}")
    if not l1_path.exists():
        raise FileNotFoundError(f"L1 derived CSV not found: {l1_path}")
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged derived CSV not found: {merged_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata derived CSV not found: {metadata_path}")

    h1_df = pd.read_csv(h1_path)
    l1_df = pd.read_csv(l1_path)
    merged_df = pd.read_csv(merged_path)
    metadata_df = pd.read_csv(metadata_path)

    ensure_columns(
        h1_df,
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain"],
        "h1_df",
    )
    ensure_columns(
        l1_df,
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain"],
        "l1_df",
    )
    ensure_columns(
        metadata_df,
        ["detector_label", "metric", "value"],
        "metadata_df",
    )

    h1_input = build_detector_input(h1_df, "H1")
    l1_input = build_detector_input(l1_df, "L1")
    merged_input = build_merged_input(merged_df)
    event_summary = build_event_summary(h1_input, l1_input, merged_input, metadata_df)

    h1_out = input_dir / "gw150914_h1_input_4khz_32s.csv"
    l1_out = input_dir / "gw150914_l1_input_4khz_32s.csv"
    merged_out = input_dir / "gw150914_h1_l1_input_4khz_32s.csv"
    summary_out = input_dir / "gw150914_event_summary_input.csv"
    manifest_out = input_dir / "gw150914_input_manifest.txt"

    h1_input.to_csv(h1_out, index=False, encoding="utf-8-sig")
    l1_input.to_csv(l1_out, index=False, encoding="utf-8-sig")
    merged_input.to_csv(merged_out, index=False, encoding="utf-8-sig")
    event_summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    manifest_lines = [
        "GW150914 input manifest",
        "======================",
        "",
        f"h1_input_rows: {len(h1_input)}",
        f"l1_input_rows: {len(l1_input)}",
        f"merged_input_rows: {len(merged_input)}",
        f"summary_rows: {len(event_summary)}",
        "",
        f"h1_out: {h1_out}",
        f"l1_out: {l1_out}",
        f"merged_out: {merged_out}",
        f"summary_out: {summary_out}",
    ]
    manifest_out.write_text("\n".join(manifest_lines), encoding="utf-8")

    print("Done.")
    print(f"h1_out: {h1_out}")
    print(f"l1_out: {l1_out}")
    print(f"merged_out: {merged_out}")
    print(f"summary_out: {summary_out}")
    print(f"manifest_out: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
