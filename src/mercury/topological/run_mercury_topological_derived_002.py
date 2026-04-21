#!/usr/bin/env python3
"""
run_mercury_topological_derived_002.py

Purpose
-------
Derived-parameter topological response pipeline for the Mercury validation case.

Placement
---------
src/mercury/topological/run_mercury_topological_derived_002.py

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
- mercury_topological_vectors_response_derived_002.csv
- mercury_topological_elements_reference_derived_002.csv
- mercury_topological_summary_derived_002.csv
- mercury_topological_report_derived_002.txt

Model notes
-----------
This version removes manual trial parameters and derives all free response scales
from the Mercury baseline itself.

Definitions
-----------
1) Reference semi-major axis:
       a_ref := mean semi-major axis from prepared summary/input

2) Structural profile normalization:
       sigma_raw(r) = ln(r / a_ref)
       beta_derived = 1 / max_i |sigma_raw(r_i)|

   so that
       sigma(r) = beta_derived * ln(r / a_ref)
   remains normalized to approximately [-1, 1] over the available orbit sample.

3) Structural mismatch scale from standard baseline:
       delta_q  = | mean(QR) - min(R_AU) |
       delta_ad = | mean(AD) - max(R_AU) |
       delta_r  = max(delta_q, delta_ad)
       epsilon_struct = delta_r / a_ref

4) Derived weak-field structural scale:
       xi_cinfo2_derived = (mu_sun / (a_ref * beta_derived)) * epsilon_struct

   so that the induced topological/Newtonian ratio is anchored to the
   observed baseline mismatch scale rather than an externally imposed number.

5) Effective response:
       d sigma / d r = beta_derived / r
       g_N(r)        = mu_sun / r^2
       g_topo(r)     = - xi_cinfo2_derived * d sigma / d r
       g_eff(r)      = g_N(r) + g_topo(r)

All quantities are kept in AU/day-based units where appropriate.
The output remains a structure-response quantity, not a closed observational prediction.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


MU_SUN_AU3_PER_DAY2 = 2.9591225740912142e-04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    parser.add_argument("--elements-input", type=str, default="mercury_elements_input_2026.csv")
    parser.add_argument("--vectors-input", type=str, default="mercury_vectors_input_2026.csv")
    parser.add_argument("--summary-input", type=str, default="mercury_orbit_summary_input_2026.csv")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def load_metric(summary_df: pd.DataFrame, metric: str, fallback: float | None = None) -> float:
    mask = summary_df["metric"].astype(str) == metric
    if mask.any():
        return float(summary_df.loc[mask, "value"].iloc[0])
    if fallback is None:
        raise KeyError(f"Metric not found in summary: {metric}")
    return float(fallback)


def derive_parameters(elements_df: pd.DataFrame, vectors_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, float]:
    a_ref = load_metric(summary_df, "mean_semi_major_axis_au", fallback=float(elements_df["A"].astype(float).mean()))
    q_mean = float(elements_df["QR"].astype(float).mean())
    ad_mean = float(elements_df["AD"].astype(float).mean())

    r = vectors_df["R_AU"].astype(float).to_numpy()
    r_min = float(np.min(r))
    r_max = float(np.max(r))

    sigma_raw = np.log(r / a_ref)
    sigma_raw_abs_max = float(np.max(np.abs(sigma_raw)))
    if sigma_raw_abs_max <= 0:
        raise ValueError("sigma_raw_abs_max is non-positive; cannot derive beta.")

    beta_derived = 1.0 / sigma_raw_abs_max

    delta_q = abs(q_mean - r_min)
    delta_ad = abs(ad_mean - r_max)
    delta_r = max(delta_q, delta_ad)
    epsilon_struct = delta_r / a_ref

    xi_cinfo2_derived = (MU_SUN_AU3_PER_DAY2 / (a_ref * beta_derived)) * epsilon_struct

    return {
        "a_ref": a_ref,
        "q_mean": q_mean,
        "ad_mean": ad_mean,
        "r_min": r_min,
        "r_max": r_max,
        "sigma_raw_abs_max": sigma_raw_abs_max,
        "beta_derived": beta_derived,
        "delta_q": delta_q,
        "delta_ad": delta_ad,
        "delta_r": delta_r,
        "epsilon_struct": epsilon_struct,
        "xi_cinfo2_derived": xi_cinfo2_derived,
    }


def build_topological_vectors(vectors_df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    out = vectors_df.copy().sort_values("JDTDB").reset_index(drop=True)

    a_ref = params["a_ref"]
    beta = params["beta_derived"]
    xi_cinfo2 = params["xi_cinfo2_derived"]

    r = out["R_AU"].astype(float).to_numpy()
    v = out["V_AU_PER_DAY"].astype(float).to_numpy()

    sigma_raw = np.log(r / a_ref)
    sigma = beta * sigma_raw
    grad_sigma = beta / r

    g_newton = MU_SUN_AU3_PER_DAY2 / (r ** 2)
    g_topo = -xi_cinfo2 * grad_sigma
    g_eff = g_newton + g_topo
    topo_to_newton_ratio = np.abs(g_topo) / np.maximum(np.abs(g_newton), 1e-30)
    speed_to_newton_circular_ratio = (v ** 2) / np.maximum(r * np.abs(g_newton), 1e-30)

    out["A_REF_AU"] = a_ref
    out["SIGMA_RAW"] = sigma_raw
    out["BETA_DERIVED"] = beta
    out["SIGMA_DERIVED"] = sigma
    out["GRAD_SIGMA_DERIVED_PER_AU"] = grad_sigma
    out["XI_CINFO2_DERIVED_AU2_PER_DAY2"] = xi_cinfo2
    out["G_NEWTON_AU_PER_DAY2"] = g_newton
    out["G_TOPOLOGICAL_DERIVED_AU_PER_DAY2"] = g_topo
    out["G_EFFECTIVE_DERIVED_AU_PER_DAY2"] = g_eff
    out["TOPO_TO_NEWTON_RATIO_DERIVED"] = topo_to_newton_ratio
    out["SPEED_TO_NEWTON_CIRCULAR_RATIO"] = speed_to_newton_circular_ratio

    return out


def build_summary(elements_df: pd.DataFrame, topo_vectors_df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    idx_min_r = topo_vectors_df["R_AU"].astype(float).idxmin()
    idx_max_r = topo_vectors_df["R_AU"].astype(float).idxmax()

    peri_row = topo_vectors_df.loc[idx_min_r]
    aphe_row = topo_vectors_df.loc[idx_max_r]

    rows = [
        {"metric": "elements_rows", "value": len(elements_df), "note": "Number of elements input rows."},
        {"metric": "vectors_rows", "value": len(topo_vectors_df), "note": "Number of vectors rows in derived topological stage."},
        {"metric": "a_reference_au", "value": params["a_ref"], "note": "Reference semi-major axis derived from baseline input."},
        {"metric": "q_mean_au", "value": params["q_mean"], "note": "Mean element-based periapsis distance."},
        {"metric": "ad_mean_au", "value": params["ad_mean"], "note": "Mean element-based apoapsis distance."},
        {"metric": "r_min_au", "value": params["r_min"], "note": "Minimum vector radius."},
        {"metric": "r_max_au", "value": params["r_max"], "note": "Maximum vector radius."},
        {"metric": "delta_q_au", "value": params["delta_q"], "note": "Absolute mismatch |mean(QR)-min(R_AU)|."},
        {"metric": "delta_ad_au", "value": params["delta_ad"], "note": "Absolute mismatch |mean(AD)-max(R_AU)|."},
        {"metric": "delta_r_au", "value": params["delta_r"], "note": "Maximum baseline mismatch scale."},
        {"metric": "epsilon_struct", "value": params["epsilon_struct"], "note": "Dimensionless structural mismatch scale delta_r/a_ref."},
        {"metric": "sigma_raw_abs_max", "value": params["sigma_raw_abs_max"], "note": "Maximum absolute unnormalized sigma_raw."},
        {"metric": "beta_derived", "value": params["beta_derived"], "note": "Derived normalization coefficient for sigma."},
        {"metric": "xi_cinfo2_derived", "value": params["xi_cinfo2_derived"], "note": "Derived structural scale from baseline mismatch."},
        {"metric": "sigma_derived_min", "value": float(topo_vectors_df["SIGMA_DERIVED"].astype(float).min()), "note": "Minimum normalized structural contrast."},
        {"metric": "sigma_derived_max", "value": float(topo_vectors_df["SIGMA_DERIVED"].astype(float).max()), "note": "Maximum normalized structural contrast."},
        {"metric": "g_topological_abs_max", "value": float(np.abs(topo_vectors_df["G_TOPOLOGICAL_DERIVED_AU_PER_DAY2"].astype(float)).max()), "note": "Maximum absolute derived topological response acceleration."},
        {"metric": "topo_to_newton_ratio_max", "value": float(topo_vectors_df["TOPO_TO_NEWTON_RATIO_DERIVED"].astype(float).max()), "note": "Maximum |g_topo|/|g_newton| in derived form."},
        {"metric": "topo_to_newton_ratio_mean", "value": float(topo_vectors_df["TOPO_TO_NEWTON_RATIO_DERIVED"].astype(float).mean()), "note": "Mean |g_topo|/|g_newton| in derived form."},
        {"metric": "perihelion_like_date", "value": peri_row["Calendar Date (TDB)"], "note": "Date at minimum R_AU in derived topological vectors response."},
        {"metric": "aphelion_like_date", "value": aphe_row["Calendar Date (TDB)"], "note": "Date at maximum R_AU in derived topological vectors response."},
    ]
    return pd.DataFrame(rows)


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> str:
    m = {str(r["metric"]): r["value"] for _, r in summary_df.iterrows()}

    lines: list[str] = []
    lines.append("Mercury topological report (derived-parameter version)")
    lines.append("=====================================================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This run removes manual trial coefficients and derives all topological response scales "
        "from the Mercury baseline itself. The output remains a structure-response quantity "
        "rather than a closed perihelion-advance reconstruction."
    )
    lines.append("")
    lines.append("Derived parameters")
    lines.append("------------------")
    lines.append(f"a_reference_au: {float(m.get('a_reference_au')):.15f}")
    lines.append(f"delta_q_au: {float(m.get('delta_q_au')):.15f}")
    lines.append(f"delta_ad_au: {float(m.get('delta_ad_au')):.15f}")
    lines.append(f"delta_r_au: {float(m.get('delta_r_au')):.15f}")
    lines.append(f"epsilon_struct: {float(m.get('epsilon_struct')):.15e}")
    lines.append(f"sigma_raw_abs_max: {float(m.get('sigma_raw_abs_max')):.15f}")
    lines.append(f"beta_derived: {float(m.get('beta_derived')):.15f}")
    lines.append(f"xi_cinfo2_derived: {float(m.get('xi_cinfo2_derived')):.15e}")
    lines.append("")
    lines.append("Response summary")
    lines.append("----------------")
    lines.append(f"sigma_derived_min: {float(m.get('sigma_derived_min')):.15f}")
    lines.append(f"sigma_derived_max: {float(m.get('sigma_derived_max')):.15f}")
    lines.append(f"g_topological_abs_max: {float(m.get('g_topological_abs_max')):.15e}")
    lines.append(f"topo_to_newton_ratio_max: {float(m.get('topo_to_newton_ratio_max')):.15e}")
    lines.append(f"topo_to_newton_ratio_mean: {float(m.get('topo_to_newton_ratio_mean')):.15e}")
    lines.append(f"perihelion_like_date: {m.get('perihelion_like_date')}")
    lines.append(f"aphelion_like_date: {m.get('aphelion_like_date')}")
    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append(
        "The structural response scale is now anchored to the internal mismatch between the "
        "element-based periapsis/apoapsis distances and the vector-based minimum/maximum orbital radius. "
        "Accordingly, the remaining free response scales are data-derived rather than manually imposed."
    )
    lines.append(
        "The main consistency question remains whether the derived structural term stays subleading "
        "relative to the Newtonian weak-field baseline while still yielding a nonzero, traceable response."
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

    params = derive_parameters(elements_df, vectors_df, summary_df)
    topo_vectors_df = build_topological_vectors(vectors_df, params)
    topo_summary_df = build_summary(elements_df, topo_vectors_df, params)

    vectors_out = output_dir / "mercury_topological_vectors_response_derived_002.csv"
    elements_out = output_dir / "mercury_topological_elements_reference_derived_002.csv"
    summary_out = output_dir / "mercury_topological_summary_derived_002.csv"
    report_out = output_dir / "mercury_topological_report_derived_002.txt"

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


if __name__ == "__main__":
    raise SystemExit(main())
