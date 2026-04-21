#!/usr/bin/env python3
"""
run_mercury_topological_001.py

Purpose
-------
Topological response pipeline for the Mercury validation case.

Placement
---------
src/mercury/topological/run_mercury_topological_001.py

Input
-----
Reads prepared input files from:
  data/derived/mercury/input/

Expected input files
--------------------
- mercury_elements_input_2026.csv
- mercury_vectors_input_2026.csv
- mercury_orbit_summary_input_2026.csv

Output
------
Writes timestamped topological results to:
  results/mercury/output/topological/YYYYMMDD_HHMMSS/

Generated files
---------------
- mercury_topological_vectors_response.csv
- mercury_topological_elements_reference.csv
- mercury_topological_summary.csv
- mercury_topological_report.txt

Model notes
-----------
This pipeline uses the trial weak-field structural response form

    sigma(r) = beta * ln(r / a)
    d sigma / d r = beta / r
    g_eff(r) = g_N(r) - xi*c_info^2 * d sigma / d r

with
    g_N(r) = mu_sun / r^2

All quantities are kept in AU/day-based units where appropriate.
The topological contribution is treated as a structural-response quantity,
not as a closed observational prediction.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# Keplerian GM in AU^3 / day^2, matching JPL Horizons convention used in the raw file.
MU_SUN_AU3_PER_DAY2 = 2.9591225740912142e-04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    parser.add_argument("--elements-input", type=str, default="mercury_elements_input_2026.csv")
    parser.add_argument("--vectors-input", type=str, default="mercury_vectors_input_2026.csv")
    parser.add_argument("--summary-input", type=str, default="mercury_orbit_summary_input_2026.csv")
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Trial structural response coefficient beta in sigma(r)=beta*ln(r/a).",
    )
    parser.add_argument(
        "--xi-cinfo2",
        type=float,
        default=1.0e-8,
        help="Trial combined scale xi*c_info^2 in AU^2/day^2 units.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def load_a_reference(summary_df: pd.DataFrame, elements_df: pd.DataFrame) -> float:
    mask = summary_df["metric"].astype(str) == "mean_semi_major_axis_au"
    if mask.any():
        return float(summary_df.loc[mask, "value"].iloc[0])
    return float(elements_df["A"].astype(float).mean())


def build_topological_vectors(
    vectors_df: pd.DataFrame,
    a_ref_au: float,
    beta: float,
    xi_cinfo2: float,
) -> pd.DataFrame:
    out = vectors_df.copy()
    out = out.sort_values("JDTDB").reset_index(drop=True)

    r = out["R_AU"].astype(float).to_numpy()
    v = out["V_AU_PER_DAY"].astype(float).to_numpy()

    sigma_trial = beta * np.log(r / a_ref_au)
    grad_sigma_trial = beta / r

    g_newton = MU_SUN_AU3_PER_DAY2 / (r ** 2)
    g_topological = -xi_cinfo2 * grad_sigma_trial
    g_effective = g_newton + g_topological

    topo_to_newton_ratio = np.abs(g_topological) / np.maximum(np.abs(g_newton), 1e-30)
    speed_to_circular_like_ratio = (v ** 2) / np.maximum(r * np.abs(g_newton), 1e-30)

    out["A_REF_AU"] = a_ref_au
    out["SIGMA_TRIAL"] = sigma_trial
    out["GRAD_SIGMA_TRIAL_PER_AU"] = grad_sigma_trial
    out["G_NEWTON_AU_PER_DAY2"] = g_newton
    out["G_TOPOLOGICAL_TRIAL_AU_PER_DAY2"] = g_topological
    out["G_EFFECTIVE_TRIAL_AU_PER_DAY2"] = g_effective
    out["TOPO_TO_NEWTON_RATIO"] = topo_to_newton_ratio
    out["SPEED_TO_NEWTON_CIRCULAR_RATIO"] = speed_to_circular_like_ratio

    return out


def build_summary(
    elements_df: pd.DataFrame,
    topo_vectors_df: pd.DataFrame,
    a_ref_au: float,
    beta: float,
    xi_cinfo2: float,
) -> pd.DataFrame:
    idx_min_r = topo_vectors_df["R_AU"].astype(float).idxmin()
    idx_max_r = topo_vectors_df["R_AU"].astype(float).idxmax()

    peri_row = topo_vectors_df.loc[idx_min_r]
    aphe_row = topo_vectors_df.loc[idx_max_r]

    rows = [
        {"metric": "elements_rows", "value": len(elements_df), "note": "Number of elements input rows."},
        {"metric": "vectors_rows", "value": len(topo_vectors_df), "note": "Number of vectors rows used in topological response stage."},
        {"metric": "a_reference_au", "value": a_ref_au, "note": "Reference semi-major axis used in sigma(r)=beta*ln(r/a)."},
        {"metric": "beta", "value": beta, "note": "Trial structural coefficient beta."},
        {"metric": "xi_cinfo2", "value": xi_cinfo2, "note": "Trial combined scale xi*c_info^2 in AU^2/day^2."},
        {"metric": "sigma_trial_min", "value": float(topo_vectors_df["SIGMA_TRIAL"].astype(float).min()), "note": "Minimum trial structural contrast."},
        {"metric": "sigma_trial_max", "value": float(topo_vectors_df["SIGMA_TRIAL"].astype(float).max()), "note": "Maximum trial structural contrast."},
        {"metric": "grad_sigma_trial_min_per_au", "value": float(topo_vectors_df["GRAD_SIGMA_TRIAL_PER_AU"].astype(float).min()), "note": "Minimum trial structural gradient."},
        {"metric": "grad_sigma_trial_max_per_au", "value": float(topo_vectors_df["GRAD_SIGMA_TRIAL_PER_AU"].astype(float).max()), "note": "Maximum trial structural gradient."},
        {"metric": "g_topological_abs_max", "value": float(np.abs(topo_vectors_df["G_TOPOLOGICAL_TRIAL_AU_PER_DAY2"].astype(float)).max()), "note": "Maximum absolute trial topological response acceleration."},
        {"metric": "topo_to_newton_ratio_max", "value": float(topo_vectors_df["TOPO_TO_NEWTON_RATIO"].astype(float).max()), "note": "Maximum ratio |g_topo|/|g_newton|."},
        {"metric": "topo_to_newton_ratio_mean", "value": float(topo_vectors_df["TOPO_TO_NEWTON_RATIO"].astype(float).mean()), "note": "Mean ratio |g_topo|/|g_newton|."},
        {"metric": "perihelion_like_date", "value": peri_row["Calendar Date (TDB)"], "note": "Date at minimum R_AU in topological vectors response."},
        {"metric": "aphelion_like_date", "value": aphe_row["Calendar Date (TDB)"], "note": "Date at maximum R_AU in topological vectors response."},
        {"metric": "sigma_trial_at_perihelion_like", "value": float(peri_row["SIGMA_TRIAL"]), "note": "Trial structural contrast at perihelion-like date."},
        {"metric": "sigma_trial_at_aphelion_like", "value": float(aphe_row["SIGMA_TRIAL"]), "note": "Trial structural contrast at aphelion-like date."},
        {"metric": "g_topological_at_perihelion_like", "value": float(peri_row["G_TOPOLOGICAL_TRIAL_AU_PER_DAY2"]), "note": "Trial topological response at perihelion-like date."},
        {"metric": "g_topological_at_aphelion_like", "value": float(aphe_row["G_TOPOLOGICAL_TRIAL_AU_PER_DAY2"]), "note": "Trial topological response at aphelion-like date."},
    ]
    return pd.DataFrame(rows)


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> str:
    m = {str(r["metric"]): r["value"] for _, r in summary_df.iterrows()}

    lines: list[str] = []
    lines.append("Mercury topological report")
    lines.append("=========================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This is the topological response stage for the Mercury validation case. "
        "The present run does not claim a closed perihelion-advance reconstruction. "
        "Its role is to compute a reproducible weak-field structural-response quantity "
        "based on the trial form sigma(r)=beta*ln(r/a)."
    )
    lines.append("")
    lines.append("Trial parameters")
    lines.append("----------------")
    lines.append(f"a_reference_au: {float(m.get('a_reference_au')):.15f}")
    lines.append(f"beta: {float(m.get('beta')):.15f}")
    lines.append(f"xi_cinfo2: {float(m.get('xi_cinfo2')):.15e}")
    lines.append("")
    lines.append("Response summary")
    lines.append("----------------")
    lines.append(f"sigma_trial_min: {float(m.get('sigma_trial_min')):.15f}")
    lines.append(f"sigma_trial_max: {float(m.get('sigma_trial_max')):.15f}")
    lines.append(f"grad_sigma_trial_min_per_au: {float(m.get('grad_sigma_trial_min_per_au')):.15f}")
    lines.append(f"grad_sigma_trial_max_per_au: {float(m.get('grad_sigma_trial_max_per_au')):.15f}")
    lines.append(f"g_topological_abs_max: {float(m.get('g_topological_abs_max')):.15e}")
    lines.append(f"topo_to_newton_ratio_max: {float(m.get('topo_to_newton_ratio_max')):.15e}")
    lines.append(f"topo_to_newton_ratio_mean: {float(m.get('topo_to_newton_ratio_mean')):.15e}")
    lines.append(f"perihelion_like_date: {m.get('perihelion_like_date')}")
    lines.append(f"aphelion_like_date: {m.get('aphelion_like_date')}")
    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append(
        "The present topological output should be read as a structure-response quantity. "
        "It is not yet a direct observational prediction in the same status as the standard baseline."
    )
    lines.append(
        "The main consistency question at this stage is whether the trial structural term remains "
        "small relative to the Newtonian weak-field baseline while still yielding a nonzero, traceable response."
    )
    lines.append("")
    lines.append("Output directory")
    lines.append("----------------")
    lines.append(str(output_dir))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    input_dir = project_root / "data" / "derived" / "mercury" / "input"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "mercury" / "output" / "topological" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    elements_path = input_dir / args.elements_input
    vectors_path = input_dir / args.vectors_input
    summary_path = input_dir / args.summary_input

    if not elements_path.exists():
        raise FileNotFoundError(f"Elements input not found: {elements_path}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors input not found: {vectors_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary input not found: {summary_path}")

    elements_df = pd.read_csv(elements_path)
    vectors_df = pd.read_csv(vectors_path)
    summary_df = pd.read_csv(summary_path)

    ensure_columns(
        elements_df,
        ["JDTDB", "Calendar Date (TDB)", "EC", "QR", "A", "AD", "PR", "W", "OM", "MA", "TA"],
        "elements_df",
    )
    ensure_columns(
        vectors_df,
        ["JDTDB", "Calendar Date (TDB)", "X", "Y", "Z", "VX", "VY", "VZ", "R_AU", "V_AU_PER_DAY"],
        "vectors_df",
    )
    ensure_columns(summary_df, ["metric", "value", "note"], "summary_df")

    a_ref_au = load_a_reference(summary_df, elements_df)
    topo_vectors_df = build_topological_vectors(
        vectors_df=vectors_df,
        a_ref_au=a_ref_au,
        beta=float(args.beta),
        xi_cinfo2=float(args.xi_cinfo2),
    )
    topo_summary_df = build_summary(
        elements_df=elements_df,
        topo_vectors_df=topo_vectors_df,
        a_ref_au=a_ref_au,
        beta=float(args.beta),
        xi_cinfo2=float(args.xi_cinfo2),
    )

    vectors_out = output_dir / "mercury_topological_vectors_response.csv"
    elements_out = output_dir / "mercury_topological_elements_reference.csv"
    summary_out = output_dir / "mercury_topological_summary.csv"
    report_out = output_dir / "mercury_topological_report.txt"

    topo_vectors_df.to_csv(vectors_out, index=False, encoding="utf-8-sig")
    elements_df.to_csv(elements_out, index=False, encoding="utf-8-sig")
    topo_summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    report_out.write_text(build_report(topo_summary_df, output_dir), encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"vectors_out: {vectors_out}")
    print(f"elements_out: {elements_out}")
    print(f"summary_out: {summary_out}")
    print(f"report_out: {report_out}")
    return 0


def load_a_reference(summary_df: pd.DataFrame, elements_df: pd.DataFrame) -> float:
    mask = summary_df["metric"].astype(str) == "mean_semi_major_axis_au"
    if mask.any():
        return float(summary_df.loc[mask, "value"].iloc[0])
    return float(elements_df["A"].astype(float).mean())


if __name__ == "__main__":
    raise SystemExit(main())
