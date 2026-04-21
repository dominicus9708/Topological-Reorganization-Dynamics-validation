#!/usr/bin/env python3
"""
build_gw150914_derived_csv_001.py

Purpose
-------
Parse GW150914 raw HDF5 strain files from:
  data/raw/gravitational waves/GW150914/strain/

and write derived CSV files to:
  data/derived/gravitational waves/GW150914/derived/

Placement
---------
data/derived/script/build_gw150914_derived_csv_001.py

Expected input files
--------------------
- H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5
- L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5

Output files
------------
- gw150914_h1_strain_4khz_32s.csv
- gw150914_l1_strain_4khz_32s.csv
- gw150914_h1_l1_strain_merged_4khz_32s.csv
- gw150914_metadata_summary.csv
- gw150914_derived_manifest.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_H1_FILE = "H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5"
DEFAULT_L1_FILE = "L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to the project root.")
    parser.add_argument("--h1-file", type=str, default=DEFAULT_H1_FILE, help="H1 raw HDF5 filename.")
    parser.add_argument("--l1-file", type=str, default=DEFAULT_L1_FILE, help="L1 raw HDF5 filename.")
    return parser.parse_args()


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def parse_gwosc_hdf5(path: Path) -> tuple[pd.DataFrame, dict]:
    with h5py.File(path, "r") as f:
        strain = np.array(f["strain/Strain"], dtype=float)

        meta = {
            "Description": decode_if_bytes(f["meta/Description"][()]),
            "DescriptionURL": decode_if_bytes(f["meta/DescriptionURL"][()]),
            "Detector": decode_if_bytes(f["meta/Detector"][()]),
            "Duration": int(f["meta/Duration"][()]),
            "GPSstart": int(f["meta/GPSstart"][()]),
            "Observatory": decode_if_bytes(f["meta/Observatory"][()]),
            "Type": decode_if_bytes(f["meta/Type"][()]),
            "UTCstart": decode_if_bytes(f["meta/UTCstart"][()]),
        }

        duration = meta["Duration"]
        n = len(strain)
        sample_rate_hz = n / duration
        dt = 1.0 / sample_rate_hz

        gps_time = meta["GPSstart"] + np.arange(n) * dt
        time_since_start = np.arange(n) * dt
        time_centered = time_since_start - duration / 2.0

        df = pd.DataFrame(
            {
                "sample_index": np.arange(n, dtype=int),
                "gps_time": gps_time,
                "time_since_start_s": time_since_start,
                "time_centered_s": time_centered,
                "strain": strain,
            }
        )

        meta["sample_count"] = n
        meta["sample_rate_hz"] = sample_rate_hz
        meta["dt_s"] = dt

    return df, meta


def build_metadata_summary(h1_meta: dict, l1_meta: dict) -> pd.DataFrame:
    rows = []
    for label, meta in [("H1", h1_meta), ("L1", l1_meta)]:
        for key, value in meta.items():
            rows.append(
                {
                    "detector_label": label,
                    "metric": key,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    raw_dir = project_root / "data" / "raw" / "gravitational waves" / "GW150914" / "strain"
    derived_dir = project_root / "data" / "derived" / "gravitational waves" / "GW150914" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    h1_path = raw_dir / args.h1_file
    l1_path = raw_dir / args.l1_file

    if not h1_path.exists():
        raise FileNotFoundError(f"H1 raw file not found: {h1_path}")
    if not l1_path.exists():
        raise FileNotFoundError(f"L1 raw file not found: {l1_path}")

    h1_df, h1_meta = parse_gwosc_hdf5(h1_path)
    l1_df, l1_meta = parse_gwosc_hdf5(l1_path)

    h1_df["detector"] = "H1"
    l1_df["detector"] = "L1"

    merged_df = h1_df.merge(
        l1_df,
        on=["sample_index", "gps_time", "time_since_start_s", "time_centered_s"],
        how="outer",
        suffixes=("_h1", "_l1"),
    )

    metadata_summary_df = build_metadata_summary(h1_meta, l1_meta)

    h1_out = derived_dir / "gw150914_h1_strain_4khz_32s.csv"
    l1_out = derived_dir / "gw150914_l1_strain_4khz_32s.csv"
    merged_out = derived_dir / "gw150914_h1_l1_strain_merged_4khz_32s.csv"
    meta_out = derived_dir / "gw150914_metadata_summary.csv"
    manifest_out = derived_dir / "gw150914_derived_manifest.txt"

    h1_df.to_csv(h1_out, index=False, encoding="utf-8-sig")
    l1_df.to_csv(l1_out, index=False, encoding="utf-8-sig")
    merged_df.to_csv(merged_out, index=False, encoding="utf-8-sig")
    metadata_summary_df.to_csv(meta_out, index=False, encoding="utf-8-sig")

    manifest_lines = [
        "GW150914 derived manifest",
        "========================",
        "",
        f"h1_input: {h1_path}",
        f"l1_input: {l1_path}",
        "",
        f"h1_rows: {len(h1_df)}",
        f"l1_rows: {len(l1_df)}",
        f"merged_rows: {len(merged_df)}",
        "",
        f"h1_sample_rate_hz: {h1_meta['sample_rate_hz']}",
        f"l1_sample_rate_hz: {l1_meta['sample_rate_hz']}",
        f"h1_duration_s: {h1_meta['Duration']}",
        f"l1_duration_s: {l1_meta['Duration']}",
        f"h1_gps_start: {h1_meta['GPSstart']}",
        f"l1_gps_start: {l1_meta['GPSstart']}",
        "",
        f"h1_output: {h1_out}",
        f"l1_output: {l1_out}",
        f"merged_output: {merged_out}",
        f"metadata_output: {meta_out}",
    ]
    manifest_out.write_text("\n".join(manifest_lines), encoding="utf-8")

    print("Done.")
    print(f"h1_out: {h1_out}")
    print(f"l1_out: {l1_out}")
    print(f"merged_out: {merged_out}")
    print(f"meta_out: {meta_out}")
    print(f"manifest_out: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
