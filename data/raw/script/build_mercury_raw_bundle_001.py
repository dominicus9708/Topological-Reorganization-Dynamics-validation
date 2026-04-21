#!/usr/bin/env python3
"""
build_mercury_raw_bundle_001.py

Purpose
-------
Build a Mercury raw-data bundle by separating literature sources and raw data sources
into different folders under data/raw/mercury/.

Expected project structure
--------------------------
<project_root>/
  data/
    raw/
      script/
        build_mercury_raw_bundle_001.py
        mercury_source_registry.csv
      mercury/
        literature/
        raw_data/
        registry/
        manifest/

Registry CSV required columns
-----------------------------
source_id,title,url,category,filename,note

Allowed category values
-----------------------
paper
raw_data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ALLOWED_CATEGORIES = {"paper", "raw_data"}


@dataclass
class SourceRow:
    source_id: str
    title: str
    url: str
    category: str
    filename: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to the project root.")
    parser.add_argument(
        "--registry-csv",
        type=str,
        default="data/raw/script/mercury_source_registry.csv",
        help="Path to the source registry CSV, relative to project root unless absolute.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Download timeout in seconds.")
    return parser.parse_args()


def resolve_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    paths = {
        "literature": base_dir / "literature",
        "raw_data": base_dir / "raw_data",
        "registry": base_dir / "registry",
        "manifest": base_dir / "manifest",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def read_registry(registry_csv: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with registry_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = ["source_id", "title", "url", "category", "filename", "note"]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in registry CSV: {missing}")
        for idx, row in enumerate(reader, start=2):
            category = (row.get("category") or "").strip()
            if category not in ALLOWED_CATEGORIES:
                raise ValueError(
                    f"Invalid category '{category}' at CSV line {idx}. Allowed values: {sorted(ALLOWED_CATEGORIES)}"
                )
            rows.append(
                SourceRow(
                    source_id=(row.get("source_id") or "").strip(),
                    title=(row.get("title") or "").strip(),
                    url=(row.get("url") or "").strip(),
                    category=category,
                    filename=(row.get("filename") or "").strip(),
                    note=(row.get("note") or "").strip(),
                )
            )
    return rows


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path, timeout: int) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            out_path.write_bytes(response.read())
        return "downloaded"
    except Exception as exc:
        return f"failed: {exc}"


def save_registry_snapshot(registry_csv: Path, registry_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = registry_dir / f"mercury_source_registry_snapshot_{timestamp}.csv"
    shutil.copy2(registry_csv, snapshot_path)
    return snapshot_path


def write_readme(base_dir: Path) -> Path:
    readme_path = base_dir / "README_mercury_raw_structure.txt"
    content = """Mercury raw-data folder structure
================================

literature/
- Stores paper PDFs, article files, or literature-origin source files.

raw_data/
- Stores CSV, TXT, supplementary tables, or machine-readable raw inputs.

registry/
- Stores source registry snapshots used at execution time.

manifest/
- Stores execution manifests describing download results, local paths, and hashes.

This folder is intended to preserve a strict separation between literature provenance
and raw numerical/source data before later processing into derived/input stages.
"""
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    registry_csv = resolve_path(project_root, args.registry_csv)
    if not registry_csv.exists():
        raise FileNotFoundError(f"Registry CSV not found: {registry_csv}")

    mercury_raw_dir = project_root / "data" / "raw" / "mercury"
    dirs = ensure_dirs(mercury_raw_dir)
    rows = read_registry(registry_csv)
    snapshot_path = save_registry_snapshot(registry_csv, dirs["registry"])
    write_readme(mercury_raw_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = dirs["manifest"] / f"mercury_download_manifest_{timestamp}.csv"

    fieldnames = [
        "source_id", "title", "url", "category", "filename", "note",
        "status", "local_path", "size_bytes", "sha256",
    ]

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            target_dir = dirs["literature"] if row.category == "paper" else dirs["raw_data"]
            out_path = target_dir / row.filename
            status = download_file(row.url, out_path, timeout=args.timeout)
            size_bytes = out_path.stat().st_size if out_path.exists() else ""
            sha256 = sha256_of_file(out_path) if out_path.exists() else ""
            writer.writerow(
                {
                    "source_id": row.source_id,
                    "title": row.title,
                    "url": row.url,
                    "category": row.category,
                    "filename": row.filename,
                    "note": row.note,
                    "status": status,
                    "local_path": str(out_path),
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
            )

    print("Done.")
    print(f"Project root: {project_root}")
    print(f"Registry snapshot: {snapshot_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Mercury raw dir: {mercury_raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
