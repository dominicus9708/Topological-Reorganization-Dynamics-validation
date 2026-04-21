#!/usr/bin/env python3
from __future__ import annotations

"""
build_gw170817_derived_csv_001.py

Purpose
-------
Parse GW170817 raw HDF5 strain files from:
  data/raw/gravitational waves/GW170817/strain/

and write derived CSV files to:
  data/derived/gravitational waves/GW170817/derived/

Placement
---------
data/derived/script/build_gw170817_derived_csv_001.py

Expected input files
--------------------
- H-H1_LOSC_C00_4_V1-1187006834-4096.hdf5
- L-L1_LOSC_C00_4_V1-1187006834-4096.hdf5
- V-V1_LOSC_C00_4_V1-1187006834-4096.hdf5

Output files
------------
- gw170817_h1_strain_4khz_4096s.csv
- gw170817_l1_strain_4khz_4096s.csv
- gw170817_v1_strain_4khz_4096s.csv
- gw170817_h1_l1_v1_strain_merged_4khz_4096s.csv
- gw170817_metadata_summary.csv
- gw170817_derived_manifest.txt
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_H1_FILE = "H-H1_LOSC_C00_4_V1-1187006834-4096.hdf5"
DEFAULT_L1_FILE = "L-L1_LOSC_C00_4_V1-1187006834-4096.hdf5"
DEFAULT_V1_FILE = "V-V1_LOSC_C00_4_V1-1187006834-4096.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to the project root.")
    parser.add_argument("--h1-file", type=str, default=DEFAULT_H1_FILE)
    parser.add_argument("--l1-file", type=str, default=DEFAULT_L1_FILE)
    parser.add_argument("--v1-file", type=str, default=DEFAULT_V1_FILE)
    return parser.parse_args()


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def read_scalar_if_exists(group, key, default=None):
    try:
        return group[key][()]
    except Exception:
        return default


def parse_gwosc_hdf5(path: Path) -> tuple[pd.DataFrame, dict]:
    with h5py.File(path, "r") as f:
        strain = np.array(f["strain/Strain"], dtype=float)
        meta_group = f["meta"]

        detector = decode_if_bytes(read_scalar_if_exists(meta_group, "Detector", path.name[:2]))
        duration = read_scalar_if_exists(meta_group, "Duration", None)
        gps_start = read_scalar_if_exists(meta_group, "GPSstart", None)
        utc_start = decode_if_bytes(read_scalar_if_exists(meta_group, "UTCstart", ""))
        description = decode_if_bytes(read_scalar_if_exists(meta_group, "Description", ""))
        description_url = decode_if_bytes(read_scalar_if_exists(meta_group, "DescriptionURL", ""))
        file_type = decode_if_bytes(read_scalar_if_exists(meta_group, "Type", ""))
        observatory = decode_if_bytes(read_scalar_if_exists(meta_group, "Observatory", detector))

        n = len(strain)

        if duration is None:
            # fallback from filename if needed
            name = path.name
            try:
                duration = int(name.split("-")[-1].split(".")[0])
            except Exception:
                raise ValueError(f"Could not derive duration from file metadata or filename: {path}")

        duration = int(duration)

        sample_rate_hz = n / duration
        dt = 1.0 / sample_rate_hz

        if gps_start is None:
            try:
                gps_start = int(path.name.split("-")[-2])
            except Exception:
                gps_start = 0

        gps_start = int(gps_start)
        gps_time = gps_start + np.arange(n) * dt
        time_since_start = np.arange(n) * dt
        time_centered = time_since_start - duration / 2.0

        df = pd.DataFrame(
            {
                "sample_index": np.arange(n, dtype=np.int64),
                "gps_time": gps_time,
                "time_since_start_s": time_since_start,
                "time_centered_s": time_centered,
                "strain": strain,
            }
        )

        meta = {
            "Detector": detector,
            "Duration": duration,
            "GPSstart": gps_start,
            "UTCstart": utc_start,
            "Description": description,
            "DescriptionURL": description_url,
            "Type": file_type,
            "Observatory": observatory,
            "sample_count": n,
            "sample_rate_hz": sample_rate_hz,
            "dt_s": dt,
        }

    return df, meta


def build_metadata_summary(metas: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for label, meta in metas.items():
        for key, value in meta.items():
            rows.append({"detector_label": label, "metric": key, "value": value})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    raw_dir = project_root / "data" / "raw" / "gravitational waves" / "GW170817" / "strain"
    derived_dir = project_root / "data" / "derived" / "gravitational waves" / "GW170817" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "H1": raw_dir / args.h1_file,
        "L1": raw_dir / args.l1_file,
        "V1": raw_dir / args.v1_file,
    }

    for label, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{label} raw file not found: {p}")

    frames = {}
    metas = {}
    for label, p in paths.items():
        df, meta = parse_gwosc_hdf5(p)
        df["detector"] = label
        frames[label] = df
        metas[label] = meta

    merged_df = frames["H1"][
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain"]
    ].rename(columns={"strain": "strain_h1"}).merge(
        frames["L1"][["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain"]].rename(columns={"strain": "strain_l1"}),
        on=["sample_index", "gps_time", "time_since_start_s", "time_centered_s"],
        how="outer",
    ).merge(
        frames["V1"][["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain"]].rename(columns={"strain": "strain_v1"}),
        on=["sample_index", "gps_time", "time_since_start_s", "time_centered_s"],
        how="outer",
    )

    metadata_summary_df = build_metadata_summary(metas)

    out_h1 = derived_dir / "gw170817_h1_strain_4khz_4096s.csv"
    out_l1 = derived_dir / "gw170817_l1_strain_4khz_4096s.csv"
    out_v1 = derived_dir / "gw170817_v1_strain_4khz_4096s.csv"
    out_merged = derived_dir / "gw170817_h1_l1_v1_strain_merged_4khz_4096s.csv"
    out_meta = derived_dir / "gw170817_metadata_summary.csv"
    out_manifest = derived_dir / "gw170817_derived_manifest.txt"

    frames["H1"].to_csv(out_h1, index=False, encoding="utf-8-sig")
    frames["L1"].to_csv(out_l1, index=False, encoding="utf-8-sig")
    frames["V1"].to_csv(out_v1, index=False, encoding="utf-8-sig")
    merged_df.to_csv(out_merged, index=False, encoding="utf-8-sig")
    metadata_summary_df.to_csv(out_meta, index=False, encoding="utf-8-sig")

    lines = [
        "GW170817 derived manifest",
        "========================",
        "",
    ]
    for label, p in paths.items():
        lines.append(f"{label.lower()}_input: {p}")
    lines.append("")
    for label in ("H1", "L1", "V1"):
        lines.append(f"{label.lower()}_rows: {len(frames[label])}")
        lines.append(f"{label.lower()}_sample_rate_hz: {metas[label]['sample_rate_hz']}")
        lines.append(f"{label.lower()}_duration_s: {metas[label]['Duration']}")
        lines.append(f"{label.lower()}_gps_start: {metas[label]['GPSstart']}")
    lines.extend(
        [
            "",
            f"merged_rows: {len(merged_df)}",
            "",
            f"h1_output: {out_h1}",
            f"l1_output: {out_l1}",
            f"v1_output: {out_v1}",
            f"merged_output: {out_merged}",
            f"metadata_output: {out_meta}",
        ]
    )
    out_manifest.write_text("\n".join(lines), encoding="utf-8")

    print("Done.")
    print(f"h1_out: {out_h1}")
    print(f"l1_out: {out_l1}")
    print(f"v1_out: {out_v1}")
    print(f"merged_out: {out_merged}")
    print(f"meta_out: {out_meta}")
    print(f"manifest_out: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
