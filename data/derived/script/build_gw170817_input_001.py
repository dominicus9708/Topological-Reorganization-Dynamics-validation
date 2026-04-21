#!/usr/bin/env python3
from __future__ import annotations

"""
build_gw170817_input_001.py

Purpose
-------
Read derived GW170817 CSV files from:
  data/derived/gravitational waves/GW170817/derived/

and write cleaned input files to:
  data/derived/gravitational waves/GW170817/input/

Placement
---------
data/derived/script/build_gw170817_input_001.py

Expected input files
--------------------
- gw170817_h1_strain_4khz_4096s.csv
- gw170817_l1_strain_4khz_4096s.csv
- gw170817_v1_strain_4khz_4096s.csv
- gw170817_h1_l1_v1_strain_merged_4khz_4096s.csv
- gw170817_metadata_summary.csv

Output files
------------
- gw170817_h1_input_4khz_4096s.csv
- gw170817_l1_input_4khz_4096s.csv
- gw170817_v1_input_4khz_4096s.csv
- gw170817_h1_l1_v1_input_4khz_4096s.csv
- gw170817_event_summary_input.csv
- gw170817_input_manifest.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to the project root.")
    parser.add_argument("--h1-csv", type=str, default="gw170817_h1_strain_4khz_4096s.csv")
    parser.add_argument("--l1-csv", type=str, default="gw170817_l1_strain_4khz_4096s.csv")
    parser.add_argument("--v1-csv", type=str, default="gw170817_v1_strain_4khz_4096s.csv")
    parser.add_argument("--merged-csv", type=str, default="gw170817_h1_l1_v1_strain_merged_4khz_4096s.csv")
    parser.add_argument("--metadata-csv", type=str, default="gw170817_metadata_summary.csv")
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
            "sample_index", "gps_time", "time_since_start_s", "time_centered_s",
            "strain_h1", "strain_l1", "strain_v1",
        ],
        "merged_df",
    )

    out = merged_df[
        [
            "sample_index", "gps_time", "time_since_start_s", "time_centered_s",
            "strain_h1", "strain_l1", "strain_v1",
        ]
    ].copy().sort_values("sample_index").reset_index(drop=True)

    out["strain_mean_h1_l1"] = (
        out["strain_h1"].astype(float) + out["strain_l1"].astype(float)
    ) / 2.0
    out["strain_mean_h1_l1_v1"] = (
        out["strain_h1"].astype(float) + out["strain_l1"].astype(float) + out["strain_v1"].astype(float)
    ) / 3.0

    out["strain_abs_mean_h1_l1"] = out["strain_mean_h1_l1"].abs()
    out["strain_abs_mean_h1_l1_v1"] = out["strain_mean_h1_l1_v1"].abs()

    out["strain_abs_diff_h1_l1"] = (
        out["strain_h1"].astype(float) - out["strain_l1"].astype(float)
    ).abs()
    out["strain_abs_diff_h1_v1"] = (
        out["strain_h1"].astype(float) - out["strain_v1"].astype(float)
    ).abs()
    out["strain_abs_diff_l1_v1"] = (
        out["strain_l1"].astype(float) - out["strain_v1"].astype(float)
    ).abs()

    peak_idx = out["strain_abs_mean_h1_l1_v1"].astype(float).idxmax()
    peak_time = float(out.loc[peak_idx, "time_centered_s"])
    out["time_from_peak_s"] = out["time_centered_s"].astype(float) - peak_time
    return out


def build_event_summary(
    h1_input: pd.DataFrame,
    l1_input: pd.DataFrame,
    v1_input: pd.DataFrame,
    merged_input: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    h1_peak_idx = h1_input["strain_abs"].astype(float).idxmax()
    l1_peak_idx = l1_input["strain_abs"].astype(float).idxmax()
    v1_peak_idx = v1_input["strain_abs"].astype(float).idxmax()
    merged_peak_idx = merged_input["strain_abs_mean_h1_l1_v1"].astype(float).idxmax()

    h1_peak_row = h1_input.loc[h1_peak_idx]
    l1_peak_row = l1_input.loc[l1_peak_idx]
    v1_peak_row = v1_input.loc[v1_peak_idx]
    merged_peak_row = merged_input.loc[merged_peak_idx]

    rows = []

    for det in ("H1", "L1", "V1"):
        rows.extend(
            [
                {
                    "metric": f"{det.lower()}_rows",
                    "value": len({"H1": h1_input, "L1": l1_input, "V1": v1_input}[det]),
                    "note": f"Number of rows in {det} input.",
                },
                {
                    "metric": f"{det.lower()}_sample_rate_hz",
                    "value": extract_metadata_value(metadata_df, det, "sample_rate_hz"),
                    "note": f"{det} sample rate from metadata.",
                },
                {
                    "metric": f"{det.lower()}_duration_s",
                    "value": extract_metadata_value(metadata_df, det, "Duration"),
                    "note": f"{det} duration from metadata.",
                },
                {
                    "metric": f"{det.lower()}_gps_start",
                    "value": extract_metadata_value(metadata_df, det, "GPSstart"),
                    "note": f"{det} GPS start from metadata.",
                },
            ]
        )

    rows.extend(
        [
            {"metric": "merged_rows", "value": len(merged_input), "note": "Number of rows in merged input."},
            {"metric": "h1_peak_time_centered_s", "value": float(h1_peak_row["time_centered_s"]), "note": "Peak time of maximum absolute H1 strain."},
            {"metric": "l1_peak_time_centered_s", "value": float(l1_peak_row["time_centered_s"]), "note": "Peak time of maximum absolute L1 strain."},
            {"metric": "v1_peak_time_centered_s", "value": float(v1_peak_row["time_centered_s"]), "note": "Peak time of maximum absolute V1 strain."},
            {"metric": "merged_peak_time_centered_s", "value": float(merged_peak_row["time_centered_s"]), "note": "Peak time of maximum absolute mean(H1,L1,V1) strain."},
            {"metric": "h1_peak_abs_strain", "value": float(h1_peak_row["strain_abs"]), "note": "Maximum absolute H1 strain."},
            {"metric": "l1_peak_abs_strain", "value": float(l1_peak_row["strain_abs"]), "note": "Maximum absolute L1 strain."},
            {"metric": "v1_peak_abs_strain", "value": float(v1_peak_row["strain_abs"]), "note": "Maximum absolute V1 strain."},
            {"metric": "merged_peak_abs_mean_strain", "value": float(merged_peak_row["strain_abs_mean_h1_l1_v1"]), "note": "Maximum absolute mean(H1,L1,V1) strain."},
            {"metric": "h1_strain_std", "value": float(h1_input["strain"].astype(float).std()), "note": "Standard deviation of H1 strain."},
            {"metric": "l1_strain_std", "value": float(l1_input["strain"].astype(float).std()), "note": "Standard deviation of L1 strain."},
            {"metric": "v1_strain_std", "value": float(v1_input["strain"].astype(float).std()), "note": "Standard deviation of V1 strain."},
            {"metric": "merged_mean_strain_std", "value": float(merged_input["strain_mean_h1_l1_v1"].astype(float).std()), "note": "Standard deviation of mean(H1,L1,V1) strain."},
            {"metric": "mean_abs_diff_h1_l1", "value": float(merged_input["strain_abs_diff_h1_l1"].astype(float).mean()), "note": "Mean absolute detector difference |H1-L1|."},
            {"metric": "mean_abs_diff_h1_v1", "value": float(merged_input["strain_abs_diff_h1_v1"].astype(float).mean()), "note": "Mean absolute detector difference |H1-V1|."},
            {"metric": "mean_abs_diff_l1_v1", "value": float(merged_input["strain_abs_diff_l1_v1"].astype(float).mean()), "note": "Mean absolute detector difference |L1-V1|."},
            {"metric": "max_abs_diff_h1_l1", "value": float(merged_input["strain_abs_diff_h1_l1"].astype(float).max()), "note": "Maximum absolute detector difference |H1-L1|."},
            {"metric": "max_abs_diff_h1_v1", "value": float(merged_input["strain_abs_diff_h1_v1"].astype(float).max()), "note": "Maximum absolute detector difference |H1-V1|."},
            {"metric": "max_abs_diff_l1_v1", "value": float(merged_input["strain_abs_diff_l1_v1"].astype(float).max()), "note": "Maximum absolute detector difference |L1-V1|."},
        ]
    )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    derived_dir = project_root / "data" / "derived" / "gravitational waves" / "GW170817" / "derived"
    input_dir = project_root / "data" / "derived" / "gravitational waves" / "GW170817" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "h1": derived_dir / args.h1_csv,
        "l1": derived_dir / args.l1_csv,
        "v1": derived_dir / args.v1_csv,
        "merged": derived_dir / args.merged_csv,
        "metadata": derived_dir / args.metadata_csv,
    }

    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{key} derived CSV not found: {p}")

    h1_df = pd.read_csv(paths["h1"])
    l1_df = pd.read_csv(paths["l1"])
    v1_df = pd.read_csv(paths["v1"])
    merged_df = pd.read_csv(paths["merged"])
    metadata_df = pd.read_csv(paths["metadata"])

    base_required = ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain"]
    ensure_columns(h1_df, base_required, "h1_df")
    ensure_columns(l1_df, base_required, "l1_df")
    ensure_columns(v1_df, base_required, "v1_df")
    ensure_columns(metadata_df, ["detector_label", "metric", "value"], "metadata_df")

    h1_input = build_detector_input(h1_df, "H1")
    l1_input = build_detector_input(l1_df, "L1")
    v1_input = build_detector_input(v1_df, "V1")
    merged_input = build_merged_input(merged_df)
    event_summary = build_event_summary(h1_input, l1_input, v1_input, merged_input, metadata_df)

    out_h1 = input_dir / "gw170817_h1_input_4khz_4096s.csv"
    out_l1 = input_dir / "gw170817_l1_input_4khz_4096s.csv"
    out_v1 = input_dir / "gw170817_v1_input_4khz_4096s.csv"
    out_merged = input_dir / "gw170817_h1_l1_v1_input_4khz_4096s.csv"
    out_summary = input_dir / "gw170817_event_summary_input.csv"
    out_manifest = input_dir / "gw170817_input_manifest.txt"

    h1_input.to_csv(out_h1, index=False, encoding="utf-8-sig")
    l1_input.to_csv(out_l1, index=False, encoding="utf-8-sig")
    v1_input.to_csv(out_v1, index=False, encoding="utf-8-sig")
    merged_input.to_csv(out_merged, index=False, encoding="utf-8-sig")
    event_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    manifest_lines = [
        "GW170817 input manifest",
        "======================",
        "",
        f"h1_input_rows: {len(h1_input)}",
        f"l1_input_rows: {len(l1_input)}",
        f"v1_input_rows: {len(v1_input)}",
        f"merged_input_rows: {len(merged_input)}",
        f"summary_rows: {len(event_summary)}",
        "",
        f"h1_out: {out_h1}",
        f"l1_out: {out_l1}",
        f"v1_out: {out_v1}",
        f"merged_out: {out_merged}",
        f"summary_out: {out_summary}",
    ]
    out_manifest.write_text("\n".join(manifest_lines), encoding="utf-8")

    print("Done.")
    print(f"h1_out: {out_h1}")
    print(f"l1_out: {out_l1}")
    print(f"v1_out: {out_v1}")
    print(f"merged_out: {out_merged}")
    print(f"summary_out: {out_summary}")
    print(f"manifest_out: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
