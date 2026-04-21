#!/usr/bin/env python3
from __future__ import annotations

"""
build_gw170817_raw_bundle_005.py

Purpose
-------
Download the GW170817 raw-data bundle using the strain listing structure
confirmed from 004.

Confirmed GWOSC listing facts
-----------------------------
- duration is 4096 s, not 32 s
- sample rate field is sample_rate_kHz with values 4 or 16
- file format field is GWF / HDF / TXT
- HDF entries download .hdf5 files

Placement
---------
data/raw/script/build_gw170817_raw_bundle_005.py
"""

import argparse
import csv
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


EVENT_VERSION = "GW170817-v1"
GWOSC_HOST = "https://gwosc.org"


@dataclass
class DownloadRecord:
    source_id: str
    title: str
    url: str
    category: str
    filename: str
    note: str
    status: str = ""
    local_path: str = ""
    size_bytes: str = ""
    sha256: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to the project root.")
    parser.add_argument("--timeout", type=int, default=90, help="Download timeout in seconds.")
    parser.add_argument("--sample-rate-khz", type=int, default=4, choices=[4, 16], help="GWOSC sample rate in kHz.")
    parser.add_argument("--duration", type=int, default=4096, choices=[4096], help="GWOSC duration in seconds.")
    parser.add_argument("--file-format", type=str, default="HDF", choices=["HDF", "GWF", "TXT"], help="GWOSC file format code.")
    return parser.parse_args()


def fetch_bytes(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json,*/*", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def fetch_text(url: str, timeout: int) -> str:
    return fetch_bytes(url, timeout).decode("utf-8", errors="replace")


def fetch_json(url: str, timeout: int):
    return json.loads(fetch_text(url, timeout))


def download_binary(url: str, out_path: Path, timeout: int) -> str:
    try:
        out_path.write_bytes(fetch_bytes(url, timeout))
        return "downloaded"
    except Exception as exc:
        return f"failed: {exc}"


def download_text(url: str, out_path: Path, timeout: int) -> str:
    try:
        out_path.write_text(fetch_text(url, timeout), encoding="utf-8")
        return "downloaded"
    except Exception as exc:
        return f"failed: {exc}"


def sha256_of_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs(base_dir: Path) -> dict:
    paths = {
        "metadata": base_dir / "metadata",
        "strain": base_dir / "strain",
        "registry": base_dir / "registry",
        "manifest": base_dir / "manifest",
        "debug": base_dir / "metadata" / "debug",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_readme(base_dir: Path) -> Path:
    readme_path = base_dir / "README_gw170817_raw_structure_005.txt"
    content = """GW170817 raw-data folder structure (005)

metadata/
- Event metadata JSON obtained from the official GWOSC event detail API.
- GRB 170817A timing note from NASA GCN.
- Strain listing debug dumps.

strain/
- H1, L1, V1 strain files downloaded from the confirmed GWOSC listing.
- Default target in this version: 4096 s, 4 kHz, HDF.

registry/
- Snapshot of the internal source registry used by the pipeline.

manifest/
- Download status, file paths, sizes, and hashes.
"""
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def save_registry_snapshot(base_dir: Path, rows: list[dict]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base_dir / "registry" / f"gw170817_source_registry_snapshot_005_{timestamp}.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_id", "title", "url", "category", "filename", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def listing_to_rows(payload: dict) -> list[dict]:
    results = payload.get("results", [])
    return [r for r in results if isinstance(r, dict)]


def choose_rows(rows: list[dict], sample_rate_khz: int, duration: int, file_format: str) -> dict[str, dict]:
    selected = {}
    for detector in ("H1", "L1", "V1"):
        matches = [
            row for row in rows
            if str(row.get("detector", "")).upper() == detector
            and int(row.get("duration", -1)) == duration
            and int(row.get("sample_rate_kHz", -1)) == sample_rate_khz
            and str(row.get("file_format", "")).upper() == file_format.upper()
            and bool(row.get("download_url"))
        ]
        if not matches:
            raise ValueError(
                f"No matching listing row found for detector={detector}, duration={duration}, "
                f"sample_rate_kHz={sample_rate_khz}, file_format={file_format}."
            )
        selected[detector] = matches[0]
    return selected


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    base_dir = project_root / "data" / "raw" / "gravitational waves" / "GW170817"
    dirs = ensure_dirs(base_dir)
    write_readme(base_dir)

    metadata_url = f"{GWOSC_HOST}/api/v2/event-versions/{EVENT_VERSION}?format=json"
    grb_note_url = "https://gcn.gsfc.nasa.gov/other/170817.gcn3"

    metadata_out = dirs["metadata"] / "GW170817_event_metadata_v1.json"
    metadata_status = download_binary(metadata_url, metadata_out, args.timeout)
    if not metadata_out.exists():
        raise FileNotFoundError(f"Failed to download metadata JSON: {metadata_url}")
    metadata_payload = json.loads(metadata_out.read_text(encoding="utf-8"))

    strain_files_url = metadata_payload.get("strain_files_url")
    if not strain_files_url:
        raise KeyError("strain_files_url not found in GW170817 metadata payload.")

    listing_url = strain_files_url  # metadata already points to format=json
    listing_payload = fetch_json(listing_url, args.timeout)
    listing_rows = listing_to_rows(listing_payload)

    debug_json = dirs["debug"] / "GW170817_strain_listing_items_005.json"
    debug_csv = dirs["debug"] / "GW170817_strain_listing_items_005.csv"
    debug_json.write_text(json.dumps(listing_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with debug_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["detector", "gps_start", "sample_rate_kHz", "duration", "file_format", "download_url"],
        )
        writer.writeheader()
        for row in listing_rows:
            writer.writerow(
                {
                    "detector": row.get("detector", ""),
                    "gps_start": row.get("gps_start", ""),
                    "sample_rate_kHz": row.get("sample_rate_kHz", ""),
                    "duration": row.get("duration", ""),
                    "file_format": row.get("file_format", ""),
                    "download_url": row.get("download_url", ""),
                }
            )

    selected = choose_rows(
        rows=listing_rows,
        sample_rate_khz=args.sample_rate_khz,
        duration=args.duration,
        file_format=args.file_format,
    )

    registry_rows = [
        {
            "source_id": "gw170817_metadata_v1",
            "title": "GW170817 event metadata (GWOSC API v2)",
            "url": metadata_url,
            "category": "metadata",
            "filename": metadata_out.name,
            "note": "Official GWOSC event detail endpoint for GW170817.",
        },
        {
            "source_id": "grb170817a_gcn_note",
            "title": "GRB 170817A timing note (NASA GCN)",
            "url": grb_note_url,
            "category": "metadata",
            "filename": "GRB170817A_GCN_170817.gcn3.txt",
            "note": "NASA GCN text note containing Fermi GBM trigger timing for GRB 170817A.",
        },
        {
            "source_id": "gw170817_strain_listing_json",
            "title": "GW170817 strain listing JSON",
            "url": listing_url,
            "category": "metadata",
            "filename": debug_json.name,
            "note": "Saved full strain listing JSON used for selection.",
        },
    ]

    for detector, row in selected.items():
        download_url = str(row["download_url"])
        filename = download_url.rstrip("/").split("/")[-1]
        registry_rows.append(
            {
                "source_id": f"gw170817_strain_{detector.lower()}",
                "title": f"GW170817 {detector} {args.duration}-second {args.sample_rate_khz} kHz {args.file_format} strain file",
                "url": download_url,
                "category": "strain",
                "filename": filename,
                "note": "Selected from confirmed GWOSC listing.",
            }
        )

    registry_snapshot = save_registry_snapshot(base_dir, registry_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = dirs["manifest"] / f"gw170817_download_manifest_005_{timestamp}.csv"

    fieldnames = [
        "source_id", "title", "url", "category", "filename", "note",
        "status", "local_path", "size_bytes", "sha256",
    ]

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        rec = DownloadRecord(
            source_id="gw170817_metadata_v1",
            title="GW170817 event metadata (GWOSC API v2)",
            url=metadata_url,
            category="metadata",
            filename=metadata_out.name,
            note="Official GWOSC event detail endpoint for GW170817.",
            status=metadata_status,
        )
        if metadata_out.exists():
            rec.local_path = str(metadata_out)
            rec.size_bytes = str(metadata_out.stat().st_size)
            rec.sha256 = sha256_of_file(metadata_out)
        writer.writerow(rec.__dict__)

        grb_out = dirs["metadata"] / "GRB170817A_GCN_170817.gcn3.txt"
        rec = DownloadRecord(
            source_id="grb170817a_gcn_note",
            title="GRB 170817A timing note (NASA GCN)",
            url=grb_note_url,
            category="metadata",
            filename=grb_out.name,
            note="NASA GCN text note containing Fermi GBM trigger timing for GRB 170817A.",
        )
        rec.status = download_text(rec.url, grb_out, args.timeout)
        if grb_out.exists():
            rec.local_path = str(grb_out)
            rec.size_bytes = str(grb_out.stat().st_size)
            rec.sha256 = sha256_of_file(grb_out)
        writer.writerow(rec.__dict__)

        for p, sid, title, note in [
            (debug_json, "gw170817_strain_listing_items_json", "GW170817 strain listing items JSON", "Saved full strain listing JSON used for selection."),
            (debug_csv, "gw170817_strain_listing_items_csv", "GW170817 strain listing items CSV", "Saved flattened strain listing CSV used for selection."),
        ]:
            rec = DownloadRecord(
                source_id=sid,
                title=title,
                url=str(p),
                category="metadata",
                filename=p.name,
                note=note,
                status="generated",
            )
            if p.exists():
                rec.local_path = str(p)
                rec.size_bytes = str(p.stat().st_size)
                rec.sha256 = sha256_of_file(p)
            writer.writerow(rec.__dict__)

        for detector, row in selected.items():
            url = str(row["download_url"])
            filename = url.rstrip("/").split("/")[-1]
            out_path = dirs["strain"] / filename
            rec = DownloadRecord(
                source_id=f"gw170817_strain_{detector.lower()}",
                title=f"GW170817 {detector} {args.duration}-second {args.sample_rate_khz} kHz {args.file_format} strain file",
                url=url,
                category="strain",
                filename=filename,
                note="Selected from confirmed GWOSC listing.",
            )
            rec.status = download_binary(rec.url, out_path, args.timeout)
            if out_path.exists():
                rec.local_path = str(out_path)
                rec.size_bytes = str(out_path.stat().st_size)
                rec.sha256 = sha256_of_file(out_path)
            writer.writerow(rec.__dict__)

    print("Done.")
    print(f"GW170817 raw dir: {base_dir}")
    print(f"Registry snapshot: {registry_snapshot}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
