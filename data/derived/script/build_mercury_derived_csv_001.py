#!/usr/bin/env python3
"""
build_mercury_derived_csv_001.py

Purpose
-------
Parse JPL Horizons Mercury raw text files from:
  data/raw/mercury/raw_data/

and write derived CSV files to:
  data/derived/mercury/derived/

Input files expected
--------------------
- jpl_horizons_mercury_elements_2026.txt
- jpl_horizons_mercury_vectors_2026.txt

Output files
------------
- mercury_elements_2026.csv
- mercury_vectors_2026.csv
- mercury_elements_vectors_merged_2026.csv
- mercury_derived_manifest_2026.txt
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple


DEFAULT_ELEMENTS_NAME = "jpl_horizons_mercury_elements_2026.txt"
DEFAULT_VECTORS_NAME = "jpl_horizons_mercury_vectors_2026.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=str,
        required=True,
        help="Path to the project root.",
    )
    parser.add_argument(
        "--elements-file",
        type=str,
        default=DEFAULT_ELEMENTS_NAME,
        help="Elements raw filename inside data/raw/mercury/raw_data/",
    )
    parser.add_argument(
        "--vectors-file",
        type=str,
        default=DEFAULT_VECTORS_NAME,
        help="Vectors raw filename inside data/raw/mercury/raw_data/",
    )
    return parser.parse_args()


def _find_header_and_rows(text: str) -> Tuple[List[str], List[str]]:
    lines = text.splitlines()

    soe_idx = None
    eoe_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "$$SOE":
            soe_idx = i
        elif line.strip() == "$$EOE":
            eoe_idx = i
            break

    if soe_idx is None or eoe_idx is None or eoe_idx <= soe_idx:
        raise ValueError("Could not find $$SOE/$$EOE block.")

    header_idx = None
    for i in range(soe_idx - 1, -1, -1):
        line = lines[i].strip()
        if "JDTDB" in line and "," in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header line above $$SOE.")

    header_line = lines[header_idx].strip()
    raw_rows = [ln.strip() for ln in lines[soe_idx + 1 : eoe_idx] if ln.strip()]
    return parse_header_line(header_line), raw_rows


def parse_header_line(header_line: str) -> List[str]:
    parts = [p.strip() for p in header_line.split(",")]
    return [p for p in parts if p]


def split_csv_preserving_date(row: str) -> List[str]:
    row = row.rstrip(",")
    parts = [p.strip() for p in row.split(",")]
    return parts


def coerce_value(value: str):
    try:
        if any(ch in value for ch in [".", "E", "e"]):
            return float(value)
        return int(value)
    except Exception:
        return value


def parse_horizons_table(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    headers, raw_rows = _find_header_and_rows(text)

    rows = []
    for raw in raw_rows:
        fields = split_csv_preserving_date(raw)
        if len(fields) != len(headers):
            raise ValueError(
                f"Field count mismatch in {path.name}: expected {len(headers)}, got {len(fields)}\n"
                f"Headers={headers}\nRow={fields}"
            )
        row = {h: coerce_value(v) for h, v in zip(headers, fields)}
        rows.append(row)
    return rows


def compute_vector_derived_columns(rows: List[dict]) -> None:
    for row in rows:
        x = float(row["X"])
        y = float(row["Y"])
        z = float(row["Z"])
        vx = float(row["VX"])
        vy = float(row["VY"])
        vz = float(row["VZ"])

        r_au = math.sqrt(x * x + y * y + z * z)
        v_au_per_day = math.sqrt(vx * vx + vy * vy + vz * vz)

        row["R_AU"] = r_au
        row["V_AU_PER_DAY"] = v_au_per_day


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_on_jd_and_calendar(elements_rows: List[dict], vector_rows: List[dict]) -> List[dict]:
    vmap = {}
    for row in vector_rows:
        key = (str(row["JDTDB"]), str(row["Calendar Date (TDB)"]))
        vmap[key] = row

    merged = []
    for erow in elements_rows:
        key = (str(erow["JDTDB"]), str(erow["Calendar Date (TDB)"]))
        out = dict(erow)
        if key in vmap:
            vrow = vmap[key]
            for k, v in vrow.items():
                if k in ("JDTDB", "Calendar Date (TDB)"):
                    continue
                out[k] = v
        merged.append(out)
    return merged


def write_manifest(path: Path, elements_rows: List[dict], vector_rows: List[dict], merged_rows: List[dict]) -> None:
    lines = []
    lines.append("Mercury derived manifest")
    lines.append("=======================")
    lines.append("")
    lines.append(f"elements_rows: {len(elements_rows)}")
    lines.append(f"vector_rows: {len(vector_rows)}")
    lines.append(f"merged_rows: {len(merged_rows)}")
    lines.append("")
    if elements_rows:
        lines.append(f"elements_first_date: {elements_rows[0]['Calendar Date (TDB)']}")
        lines.append(f"elements_last_date: {elements_rows[-1]['Calendar Date (TDB)']}")
    if vector_rows:
        lines.append(f"vectors_first_date: {vector_rows[0]['Calendar Date (TDB)']}")
        lines.append(f"vectors_last_date: {vector_rows[-1]['Calendar Date (TDB)']}")
    if vector_rows:
        r_values = [float(r["R_AU"]) for r in vector_rows]
        lines.append("")
        lines.append(f"min_R_AU: {min(r_values):.12f}")
        lines.append(f"max_R_AU: {max(r_values):.12f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    raw_dir = project_root / "data" / "raw" / "mercury" / "raw_data"
    derived_dir = project_root / "data" / "derived" / "mercury" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    elements_path = raw_dir / args.elements_file
    vectors_path = raw_dir / args.vectors_file

    if not elements_path.exists():
        raise FileNotFoundError(f"Elements raw file not found: {elements_path}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors raw file not found: {vectors_path}")

    elements_rows = parse_horizons_table(elements_path)
    vector_rows = parse_horizons_table(vectors_path)
    compute_vector_derived_columns(vector_rows)
    merged_rows = merge_on_jd_and_calendar(elements_rows, vector_rows)

    elements_csv = derived_dir / "mercury_elements_2026.csv"
    vectors_csv = derived_dir / "mercury_vectors_2026.csv"
    merged_csv = derived_dir / "mercury_elements_vectors_merged_2026.csv"
    manifest_txt = derived_dir / "mercury_derived_manifest_2026.txt"

    write_csv(elements_csv, elements_rows)
    write_csv(vectors_csv, vector_rows)
    write_csv(merged_csv, merged_rows)
    write_manifest(manifest_txt, elements_rows, vector_rows, merged_rows)

    print("Done.")
    print(f"elements_csv: {elements_csv}")
    print(f"vectors_csv: {vectors_csv}")
    print(f"merged_csv: {merged_csv}")
    print(f"manifest_txt: {manifest_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
