#!/usr/bin/env python3
from __future__ import annotations

"""
build_gw170817_raw_bundle_004.py

Purpose
-------
Debug-first GW170817 raw bundle builder.

This version prioritizes inspecting the official GWOSC strain-files listing
before trying to download detector strain files. It saves the raw listing
responses and a flattened CSV dump of discovered items so that the exact
detector / duration / sample-rate / format fields can be inspected.

Placement
---------
data/raw/script/build_gw170817_raw_bundle_004.py
"""

import argparse
import csv
import hashlib
import html
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
    parser.add_argument("--sample-rate", type=int, default=4096, help="Requested sample rate in Hz for later filtering.")
    parser.add_argument("--duration", type=int, default=32, help="Requested duration in seconds for later filtering.")
    parser.add_argument("--file-format", type=str, default="hdf5", help="Requested file format for later filtering.")
    return parser.parse_args()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_bytes(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json,text/html,*/*", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def fetch_text(url: str, timeout: int) -> str:
    return fetch_bytes(url, timeout).decode("utf-8", errors="replace")


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
        "debug": base_dir / "metadata" / "debug",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_registry_snapshot(base_dir: Path, rows: list[dict]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base_dir / "registry" / f"gw170817_source_registry_snapshot_004_{timestamp}.csv"
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
    readme_path = base_dir / "README_gw170817_raw_structure_004.txt"
    content = """GW170817 raw-data folder structure (004)

metadata/
- Event metadata JSON obtained from the official GWOSC event detail API.
- GRB 170817A timing note from NASA GCN.
- Debug copies of strain listing responses and flattened item dumps.

strain/
- Detector strain files, if discoverable after listing inspection.

registry/
- Snapshot of the internal source registry used by the pipeline.

manifest/
- Download status, file paths, sizes, and hashes.

This 004 version is listing-inspection first.
"""
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def normalize_format_url(url: str, fmt: str) -> str:
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    params["format"] = [fmt]
    new_query = urllib.parse.urlencode(params, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def parse_json_direct(text: str) -> Any:
    return json.loads(text)


def extract_json_from_api_html(text: str) -> Any:
    m = re.search(r'<div class="response-info".*?<pre[^>]*class="prettyprint"[^>]*>.*?</span>(.*?)</pre>', text, re.S)
    if not m:
        m = re.search(r'<pre[^>]*class="prettyprint"[^>]*>(.*?)</pre>', text, re.S)
    if not m:
        raise ValueError("No prettyprint <pre> block found in HTML response.")

    block = m.group(1)
    block = html.unescape(block)
    block = re.sub(r'<[^>]+>', '', block)
    block = block.strip()

    json_start = block.find("{")
    if json_start == -1:
        json_start = block.find("[")
    if json_start == -1:
        raise ValueError("No JSON object or array found inside HTML response block.")

    block = block[json_start:]
    return json.loads(block)


def payload_to_items(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return [x for x in payload["results"] if isinstance(x, dict)]
        if isinstance(payload.get("strain_files"), list):
            return [x for x in payload["strain_files"] if isinstance(x, dict)]
        if isinstance(payload.get("files"), list):
            return [x for x in payload["files"] if isinstance(x, dict)]
        return [payload]
    return []


def flatten_item(item: dict) -> dict:
    def safe_get(*keys):
        for k in keys:
            if k in item:
                return item.get(k)
        return ""

    links = item.get("links", {}) if isinstance(item.get("links"), dict) else {}
    out = {
        "detector": safe_get("detector", "observatory"),
        "duration": safe_get("duration"),
        "sample_rate": safe_get("sample_rate", "sample_rate_hz"),
        "file_format": safe_get("file_format", "format"),
        "filename": safe_get("filename"),
        "name": safe_get("name"),
        "download_url": safe_get("download_url", "url") or links.get("download", ""),
        "raw_json": json.dumps(item, ensure_ascii=False),
    }
    if not out["filename"] and out["download_url"]:
        out["filename"] = str(out["download_url"]).rstrip("/").split("/")[-1]
    return out


def local_match(row: dict, detector: str, duration: int, sample_rate: int, file_format: str) -> bool:
    hay = " ".join([str(v) for v in row.values()]).upper()
    detector_ok = detector.upper() in hay
    duration_ok = str(duration) in hay
    sr_ok = any(tok in hay for tok in [str(sample_rate), "4KHZ" if sample_rate == 4096 else "16KHZ", "4096", "16384"])
    fmt_ok = file_format.upper() in hay or f".{file_format.upper()}" in hay
    return detector_ok and duration_ok and sr_ok and fmt_ok and bool(row.get("download_url"))


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
        {
            "source_id": "gw170817_strain_listing_json",
            "title": "GW170817 strain listing JSON response",
            "url": normalize_format_url(strain_files_url, "json"),
            "category": "metadata",
            "filename": "GW170817_strain_listing_response_004.json.txt",
            "note": "Raw strain listing response saved for debugging.",
        },
        {
            "source_id": "gw170817_strain_listing_api",
            "title": "GW170817 strain listing API HTML response",
            "url": normalize_format_url(strain_files_url, "api"),
            "category": "metadata",
            "filename": "GW170817_strain_listing_response_004.api.html",
            "note": "Raw API HTML response saved for debugging.",
        },
    ]
    registry_snapshot = save_registry_snapshot(base_dir, registry_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = dirs["manifest"] / f"gw170817_download_manifest_004_{timestamp}.csv"

    json_listing_url = normalize_format_url(strain_files_url, "json")
    api_listing_url = normalize_format_url(strain_files_url, "api")

    json_listing_path = dirs["debug"] / "GW170817_strain_listing_response_004.json.txt"
    api_listing_path = dirs["debug"] / "GW170817_strain_listing_response_004.api.html"
    parsed_items_json_path = dirs["debug"] / "GW170817_strain_listing_items_004.json"
    parsed_items_csv_path = dirs["debug"] / "GW170817_strain_listing_items_004.csv"

    json_status = download_text(json_listing_url, json_listing_path, args.timeout)
    api_status = download_text(api_listing_url, api_listing_path, args.timeout)

    items = []
    parse_mode = "none"

    if json_listing_path.exists():
        try:
            payload = parse_json_direct(json_listing_path.read_text(encoding="utf-8"))
            items = payload_to_items(payload)
            parse_mode = "json_direct"
        except Exception:
            pass

    if not items and api_listing_path.exists():
        try:
            payload = extract_json_from_api_html(api_listing_path.read_text(encoding="utf-8"))
            items = payload_to_items(payload)
            parse_mode = "api_html_embedded_json"
        except Exception:
            pass

    flattened = [flatten_item(item) for item in items]

    parsed_items_json_path.write_text(
        json.dumps(flattened, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if flattened:
        with parsed_items_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["detector", "duration", "sample_rate", "file_format", "filename", "name", "download_url", "raw_json"],
            )
            writer.writeheader()
            writer.writerows(flattened)
    else:
        with parsed_items_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["detector", "duration", "sample_rate", "file_format", "filename", "name", "download_url", "raw_json"],
            )
            writer.writeheader()

    discovered_rows = []
    for detector in ("H1", "L1", "V1"):
        source_id = f"gw170817_strain_{detector.lower()}"
        matches = [row for row in flattened if local_match(row, detector, args.duration, args.sample_rate, args.file_format)]
        if matches:
            row = matches[0]
            filename = row["filename"] or f"{detector}_download_unknown_name.{args.file_format}"
            discovered_rows.append(
                {
                    "source_id": source_id,
                    "title": f"GW170817 {detector} {args.duration}-second {args.sample_rate} Hz strain file",
                    "url": row["download_url"],
                    "category": "strain",
                    "filename": filename,
                    "note": f"Matched locally from parsed listing using parse_mode={parse_mode}.",
                }
            )
        else:
            discovered_rows.append(
                {
                    "source_id": source_id,
                    "title": f"GW170817 {detector} {args.duration}-second {args.sample_rate} Hz strain file",
                    "url": strain_files_url,
                    "category": "strain",
                    "filename": f"{detector}_strain_query_failed_004.txt",
                    "note": f"No matching listing row found for detector={detector}, duration={args.duration}, sample_rate={args.sample_rate}, format={args.file_format}. parse_mode={parse_mode}",
                }
            )

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

        for status, path, sid, title, url, note in [
            (json_status, json_listing_path, "gw170817_strain_listing_json", "GW170817 strain listing JSON response", json_listing_url, "Raw strain listing response saved for debugging."),
            (api_status, api_listing_path, "gw170817_strain_listing_api", "GW170817 strain listing API HTML response", api_listing_url, "Raw API HTML response saved for debugging."),
            ("generated", parsed_items_json_path, "gw170817_strain_listing_items_json", "GW170817 parsed strain listing items JSON", str(parsed_items_json_path), f"Parsed listing items; parse_mode={parse_mode}."),
            ("generated", parsed_items_csv_path, "gw170817_strain_listing_items_csv", "GW170817 parsed strain listing items CSV", str(parsed_items_csv_path), f"Flattened listing items; parse_mode={parse_mode}."),
        ]:
            rec = DownloadRecord(
                source_id=sid,
                title=title,
                url=url,
                category="metadata",
                filename=path.name,
                note=note,
                status=status,
            )
            if path.exists():
                rec.local_path = str(path)
                rec.size_bytes = str(path.stat().st_size)
                rec.sha256 = sha256_of_file(path)
            writer.writerow(rec.__dict__)

        for row in discovered_rows:
            rec = DownloadRecord(**row)
            if row["filename"].endswith("_failed_004.txt"):
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
    print(f"Parsed listing items JSON: {parsed_items_json_path}")
    print(f"Parsed listing items CSV: {parsed_items_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
