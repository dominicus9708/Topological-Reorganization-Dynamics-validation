#!/usr/bin/env python3
"""
build_gw150914_raw_bundle_001.py

Purpose
-------
Download the first raw-data bundle for the GW150914 gravitational-wave case.

Placement
---------
data/raw/script/build_gw150914_raw_bundle_001.py

Output
------
Writes raw files to:
  data/raw/gravitational waves/GW150914/

Folder structure
----------------
data/raw/gravitational waves/GW150914/
├─ metadata/
├─ strain/
├─ registry/
└─ manifest/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class DownloadItem:
    source_id: str
    title: str
    url: str
    category: str
    filename: str
    note: str


GW150914_ITEMS = [
    DownloadItem(
        source_id="gw150914_metadata_v4",
        title="GW150914 event metadata (GWOSC API v2)",
        url="https://gwosc.org/api/v2/event-versions/GW150914-v4?format=api",
        category="metadata",
        filename="GW150914_event_metadata_v4.json",
        note="Official GWOSC event detail endpoint for GW150914.",
    ),
    DownloadItem(
        source_id="gw150914_strain_h1_32s_4khz",
        title="GW150914 H1 32-second 4 kHz strain file",
        url="https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
        category="strain",
        filename="H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
        note="GWOSC event strain file for detector H1.",
    ),
    DownloadItem(
        source_id="gw150914_strain_l1_32s_4khz",
        title="GW150914 L1 32-second 4 kHz strain file",
        url="https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
        category="strain",
        filename="L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5",
        note="GWOSC event strain file for detector L1.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to the project root.")
    parser.add_argument("--timeout", type=int, default=90, help="Download timeout in seconds.")
    return parser.parse_args()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_binary(url: str, out_path: Path, timeout: int) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            out_path.write_bytes(response.read())
        return "downloaded"
    except Exception as exc:
        return f"failed: {exc}"


def ensure_dirs(base_dir: Path) -> dict:
    paths = {
        "metadata": base_dir / "metadata",
        "strain": base_dir / "strain",
        "registry": base_dir / "registry",
        "manifest": base_dir / "manifest",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_registry_snapshot(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base_dir / "registry" / f"gw150914_source_registry_snapshot_{timestamp}.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_id", "title", "url", "category", "filename", "note"],
        )
        writer.writeheader()
        for item in GW150914_ITEMS:
            writer.writerow(
                {
                    "source_id": item.source_id,
                    "title": item.title,
                    "url": item.url,
                    "category": item.category,
                    "filename": item.filename,
                    "note": item.note,
                }
            )
    return out_path


def write_readme(base_dir: Path) -> Path:
    readme_path = base_dir / "README_gw150914_raw_structure.txt"
    content = """GW150914 raw-data folder structure
=================================

metadata/
- Event metadata JSON obtained from the official GWOSC event detail API.

strain/
- Raw detector strain files for GW150914 (H1 and L1, 32 s, 4 kHz, HDF5).

registry/
- Snapshot of the internal source registry used by the pipeline.

manifest/
- Download status, file paths, sizes, and hashes.
"""
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    base_dir = project_root / "data" / "raw" / "gravitational waves" / "GW150914"
    dirs = ensure_dirs(base_dir)
    registry_snapshot = save_registry_snapshot(base_dir)
    write_readme(base_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = dirs["manifest"] / f"gw150914_download_manifest_{timestamp}.csv"

    fieldnames = [
        "source_id",
        "title",
        "url",
        "category",
        "filename",
        "note",
        "status",
        "local_path",
        "size_bytes",
        "sha256",
    ]

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in GW150914_ITEMS:
            target_dir = dirs[item.category]
            out_path = target_dir / item.filename

            status = download_binary(item.url, out_path, timeout=args.timeout)
            size_bytes = out_path.stat().st_size if out_path.exists() else ""
            sha256 = sha256_of_file(out_path) if out_path.exists() else ""

            writer.writerow(
                {
                    "source_id": item.source_id,
                    "title": item.title,
                    "url": item.url,
                    "category": item.category,
                    "filename": item.filename,
                    "note": item.note,
                    "status": status,
                    "local_path": str(out_path),
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
            )

    print("Done.")
    print(f"GW150914 raw dir: {base_dir}")
    print(f"Registry snapshot: {registry_snapshot}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
