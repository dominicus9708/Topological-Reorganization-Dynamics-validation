#!/usr/bin/env python3
from __future__ import annotations

"""
build_gw170817_raw_bundle_002.py

Purpose
-------
Download the GW170817 raw-data bundle with more robust GWOSC strain discovery.

Placement
---------
data/raw/script/build_gw170817_raw_bundle_002.py
"""

import argparse
import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
    parser.add_argument("--sample-rate", type=int, default=4096, help="Sample rate in Hz. Use 4096 or 16384.")
    parser.add_argument("--duration", type=int, default=32, help="Strain duration in seconds. Use 32 or 4096.")
    parser.add_argument("--file-format", type=str, default="hdf5", help="File format. Use hdf5, gwf, or txt.")
    return parser.parse_args()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_bytes(url: str, timeout: int) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def fetch_text(url: str, timeout: int) -> str:
    return fetch_bytes(url, timeout).decode("utf-8", errors="replace")


def fetch_json(url: str, timeout: int) -> Any:
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


def save_registry_snapshot(base_dir: Path, rows: list[dict]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base_dir / "registry" / f"gw170817_source_registry_snapshot_002_{timestamp}.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_id", "title", "url", "category", "filename", "note"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def write_readme(base_dir: Path) -> Path:
    readme_path = base_dir / "README_gw170817_raw_structure_002.txt"
    content = """GW170817 raw-data folder structure

metadata/
- Event metadata JSON obtained from the official GWOSC event detail API.
- GRB 170817A timing note from NASA GCN.

strain/
- Raw detector strain files for GW170817 (H1, L1, V1), discovered dynamically.

registry/
- Snapshot of the internal source registry used by the pipeline.

manifest/
- Download status, file paths, sizes, and hashes.
"""
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def sample_rate_api_value(sample_rate_hz: int) -> str:
    if sample_rate_hz == 4096:
        return "4"
    if sample_rate_hz == 16384:
        return "16"
    return str(sample_rate_hz)


def build_strain_listing_url(strain_files_url: str, detector: str, sample_rate_hz: int, duration_s: int, file_format: str, fmt: str) -> str:
    parsed = urllib.parse.urlparse(strain_files_url)
    params = urllib.parse.parse_qs(parsed.query)
    params["format"] = [fmt]
    params["detector"] = [detector]
    params["duration"] = [str(duration_s)]
    params["file-format"] = [file_format]
    params["sample-rate"] = [sample_rate_api_value(sample_rate_hz)]
    new_query = urllib.parse.urlencode(params, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def choose_download_url_from_json(payload: Any) -> tuple[str, str]:
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            items = payload["results"]
        elif "strain_files" in payload and isinstance(payload["strain_files"], list):
            items = payload["strain_files"]
        else:
            items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    if not items:
        raise ValueError("No strain files returned in JSON payload.")

    item = items[0]
    download_url = item.get("download_url") or item.get("url") or item.get("links", {}).get("download")
    if not download_url:
        raise ValueError("Could not find a downloadable strain URL in JSON payload.")
    filename = item.get("filename") or download_url.rstrip("/").split("/")[-1]
    return download_url, filename


def choose_download_url_from_html(html: str, detector: str, file_format: str) -> tuple[str, str]:
    pattern = re.compile(r'href="([^"]*%s[^"]*\.%s)"' % (re.escape(detector), re.escape(file_format)), re.IGNORECASE)
    matches = pattern.findall(html)
    if not matches:
        raise ValueError("Could not find downloadable strain link in HTML fallback.")
    download_url = matches[0]
    if download_url.startswith("/"):
        download_url = urllib.parse.urljoin(GWOSC_HOST, download_url)
    filename = download_url.rstrip("/").split("/")[-1]
    return download_url, filename


def discover_strain_download(strain_files_url: str, detector: str, sample_rate_hz: int, duration_s: int, file_format: str, timeout: int) -> tuple[str, str, str]:
    listing_json_url = build_strain_listing_url(
        strain_files_url=strain_files_url,
        detector=detector,
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        file_format=file_format,
        fmt="json",
    )
    try:
        payload = fetch_json(listing_json_url, timeout)
        download_url, filename = choose_download_url_from_json(payload)
        return download_url, filename, "json_listing"
    except Exception:
        pass

    listing_api_url = build_strain_listing_url(
        strain_files_url=strain_files_url,
        detector=detector,
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        file_format=file_format,
        fmt="api",
    )
    try:
        html = fetch_text(listing_api_url, timeout)
        download_url, filename = choose_download_url_from_html(html, detector=detector, file_format=file_format)
        return download_url, filename, "html_fallback"
    except Exception as exc:
        raise ValueError(f"Strain discovery failed for {detector}: {exc}") from exc


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
    ]

    discovered_rows = []
    for detector in ("H1", "L1", "V1"):
        source_id = f"gw170817_strain_{detector.lower()}"
        try:
            download_url, filename, discovery_mode = discover_strain_download(
                strain_files_url=strain_files_url,
                detector=detector,
                sample_rate_hz=args.sample_rate,
                duration_s=args.duration,
                file_format=args.file_format,
                timeout=args.timeout,
            )
            discovered_rows.append(
                {
                    "source_id": source_id,
                    "title": f"GW170817 {detector} {args.duration}-second {args.sample_rate} Hz strain file",
                    "url": download_url,
                    "category": "strain",
                    "filename": filename,
                    "note": f"Discovered dynamically from GWOSC strain-files endpoint using {discovery_mode}.",
                }
            )
        except Exception as exc:
            discovered_rows.append(
                {
                    "source_id": source_id,
                    "title": f"GW170817 {detector} {args.duration}-second {args.sample_rate} Hz strain file",
                    "url": strain_files_url,
                    "category": "strain",
                    "filename": f"{detector}_strain_query_failed_002.txt",
                    "note": str(exc),
                }
            )

    registry_rows.extend(discovered_rows)
    registry_snapshot = save_registry_snapshot(base_dir, registry_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = dirs["manifest"] / f"gw170817_download_manifest_002_{timestamp}.csv"

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

        for row in discovered_rows:
            rec = DownloadRecord(**row)
            if row["filename"].endswith("_failed_002.txt"):
                fail_out = dirs["strain"] / row["filename"]
                fail_out.write_text(row["note"], encoding="utf-8")
                rec.status = "discovery_failed"
                rec.local_path = str(fail_out)
                rec.size_bytes = str(fail_out.stat().st_size)
                rec.sha256 = sha256_of_file(fail_out)
                writer.writerow(rec.__dict__)
                continue

            out_path = dirs["strain"] / row["filename"]
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
