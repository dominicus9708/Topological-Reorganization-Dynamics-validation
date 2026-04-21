#!/usr/bin/env python3
"""
run_gw150914_topological_derived_001.py

Purpose
-------
Derived-parameter topological response pipeline for the GW150914 validation case.

Placement
---------
src/gravitational_wave/GW150914/topological/run_gw150914_topological_derived_001.py

Input
-----
Reads prepared input files from:
  data/derived/gravitational waves/GW150914/input/

Expected input files
--------------------
- gw150914_h1_input_4khz_32s.csv
- gw150914_l1_input_4khz_32s.csv
- gw150914_h1_l1_input_4khz_32s.csv
- gw150914_event_summary_input.csv

Output
------
Writes timestamped topological results to:
  results/gravitational_wave/GW150914/output/topological/YYYYMMDD_HHMMSS/

Generated files
---------------
- gw150914_topological_response_derived_001.csv
- gw150914_topological_summary_derived_001.csv
- gw150914_topological_report_derived_001.txt

Model notes
-----------
This version uses only data-derived quantities.

Definitions
-----------
1) Mean strain baseline:
       h_mean(t) = (h_H1(t) + h_L1(t)) / 2

2) Structural amplitude:
       A(t) = |h_mean(t)|

3) Reference amplitude from pre-event baseline:
       A_ref = median(A(t)) over pre-event samples (time_centered_s < 0)

4) Observed structural contrast:
       sigma_obs(t) = ln(A(t) / A_ref)

5) Peak time:
       t_peak = argmax_t A(t)

6) e-fold decay time:
       t_e = first t >= t_peak such that A(t) <= A_peak / e
       tau_decay = t_e - t_peak
       kappa_derived = 1 / tau_decay

7) Post-peak model:
       sigma_model(t) = sigma_peak * exp[-kappa_derived * (t - t_peak)]   for t >= t_peak

8) Residual:
       delta_sigma(t) = sigma_obs(t) - sigma_model(t)   for t >= t_peak

9) Propagation proxy:
       P(t) = | d sigma_obs / d t |

All floors or stabilizers are derived from the data itself:
- A_floor = min positive A(t) in the full event window.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    parser.add_argument("--h1-input", type=str, default="gw150914_h1_input_4khz_32s.csv")
    parser.add_argument("--l1-input", type=str, default="gw150914_l1_input_4khz_32s.csv")
    parser.add_argument("--merged-input", type=str, default="gw150914_h1_l1_input_4khz_32s.csv")
    parser.add_argument("--summary-input", type=str, default="gw150914_event_summary_input.csv")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def build_topological_response(
    merged_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    out = merged_df.copy().sort_values("sample_index").reset_index(drop=True)

    h_mean = out["strain_mean_h1_l1"].astype(float).to_numpy()
    t = out["time_centered_s"].astype(float).to_numpy()
    A = np.abs(h_mean)

    pre_mask = t < 0.0
    if not np.any(pre_mask):
        raise ValueError("No pre-event samples found for deriving A_ref.")

    A_pre = A[pre_mask]
    A_ref = float(np.median(A_pre))

    positive_A = A[A > 0]
    if len(positive_A) == 0:
        raise ValueError("No positive amplitudes found; cannot derive A_floor.")
    A_floor = float(np.min(positive_A))

    A_safe = np.maximum(A, A_floor)
    sigma_obs = np.log(A_safe / A_ref)

    peak_idx = int(np.argmax(A))
    t_peak = float(t[peak_idx])
    A_peak = float(A[peak_idx])
    sigma_peak = float(sigma_obs[peak_idx])

    decay_threshold = A_peak / np.e
    post_idx = np.where(np.arange(len(A)) >= peak_idx)[0]
    decay_candidates = [i for i in post_idx if A[i] <= decay_threshold]

    if decay_candidates:
        e_idx = int(decay_candidates[0])
        decay_status = "crossed"
    else:
        e_idx = int(len(A) - 1)
        decay_status = "not_crossed_within_window"

    t_e = float(t[e_idx])
    tau_decay = float(t_e - t_peak)

    if tau_decay <= 0:
        # Derived fallback from sampling interval if the threshold is reached immediately or not separable.
        dt_candidates = np.diff(t)
        positive_dt = dt_candidates[dt_candidates > 0]
        if len(positive_dt) == 0:
            raise ValueError("Could not derive positive sampling interval for tau_decay fallback.")
        tau_decay = float(np.min(positive_dt))
        decay_status = decay_status + "_fallback_to_dt"

    kappa_derived = 1.0 / tau_decay

    sigma_model = np.full(len(out), np.nan, dtype=float)
    post_peak_mask = t >= t_peak
    sigma_model[post_peak_mask] = sigma_peak * np.exp(-kappa_derived * (t[post_peak_mask] - t_peak))

    delta_sigma = np.full(len(out), np.nan, dtype=float)
    delta_sigma[post_peak_mask] = sigma_obs[post_peak_mask] - sigma_model[post_peak_mask]

    # Derived propagation proxy
    dt = np.gradient(t)
    dsigma_dt = np.gradient(sigma_obs, t)
    propagation_proxy = np.abs(dsigma_dt)

    # Detector-difference auxiliary quantity
    detector_abs_diff = out["strain_abs_diff_h1_l1"].astype(float).to_numpy()
    detector_mismatch_ratio = detector_abs_diff / np.maximum(A_safe, A_floor)

    out["h_mean"] = h_mean
    out["A_obs"] = A
    out["A_safe"] = A_safe
    out["sigma_obs"] = sigma_obs
    out["sigma_model_postpeak"] = sigma_model
    out["delta_sigma_postpeak"] = delta_sigma
    out["dsigma_dt"] = dsigma_dt
    out["propagation_proxy_abs_dsigma_dt"] = propagation_proxy
    out["detector_mismatch_ratio"] = detector_mismatch_ratio
    out["time_from_peak_derived_s"] = t - t_peak

    summary = {
        "row_count": len(out),
        "A_ref": A_ref,
        "A_floor": A_floor,
        "t_peak": t_peak,
        "A_peak": A_peak,
        "sigma_peak": sigma_peak,
        "decay_threshold": decay_threshold,
        "t_e": t_e,
        "tau_decay": tau_decay,
        "kappa_derived": kappa_derived,
        "decay_status": decay_status,
        "sigma_obs_min": float(np.min(sigma_obs)),
        "sigma_obs_max": float(np.max(sigma_obs)),
        "propagation_proxy_max": float(np.max(propagation_proxy)),
        "propagation_proxy_mean": float(np.mean(propagation_proxy)),
        "detector_mismatch_ratio_max": float(np.max(detector_mismatch_ratio)),
        "detector_mismatch_ratio_mean": float(np.mean(detector_mismatch_ratio)),
        "postpeak_rmse_sigma": float(
            np.sqrt(np.nanmean((delta_sigma[post_peak_mask]) ** 2))
        ),
        "postpeak_mae_sigma": float(
            np.nanmean(np.abs(delta_sigma[post_peak_mask]))
        ),
    }
    return out, summary


def build_summary_df(summary: dict) -> pd.DataFrame:
    rows = [
        {"metric": "row_count", "value": summary["row_count"], "note": "Number of rows in topological response table."},
        {"metric": "A_ref", "value": summary["A_ref"], "note": "Pre-event median structural amplitude baseline."},
        {"metric": "A_floor", "value": summary["A_floor"], "note": "Minimum positive structural amplitude used for safe logarithm."},
        {"metric": "t_peak", "value": summary["t_peak"], "note": "Derived peak time from A(t)=|h_mean(t)|."},
        {"metric": "A_peak", "value": summary["A_peak"], "note": "Derived peak structural amplitude."},
        {"metric": "sigma_peak", "value": summary["sigma_peak"], "note": "Observed structural contrast at the peak."},
        {"metric": "decay_threshold", "value": summary["decay_threshold"], "note": "e-fold threshold A_peak/e."},
        {"metric": "t_e", "value": summary["t_e"], "note": "First post-peak time where A(t)<=A_peak/e, if available."},
        {"metric": "tau_decay", "value": summary["tau_decay"], "note": "Derived post-peak e-fold decay time."},
        {"metric": "kappa_derived", "value": summary["kappa_derived"], "note": "Derived structural relaxation rate 1/tau_decay."},
        {"metric": "decay_status", "value": summary["decay_status"], "note": "Status of the e-fold threshold crossing."},
        {"metric": "sigma_obs_min", "value": summary["sigma_obs_min"], "note": "Minimum observed structural contrast."},
        {"metric": "sigma_obs_max", "value": summary["sigma_obs_max"], "note": "Maximum observed structural contrast."},
        {"metric": "propagation_proxy_max", "value": summary["propagation_proxy_max"], "note": "Maximum absolute structural-contrast time derivative."},
        {"metric": "propagation_proxy_mean", "value": summary["propagation_proxy_mean"], "note": "Mean absolute structural-contrast time derivative."},
        {"metric": "detector_mismatch_ratio_max", "value": summary["detector_mismatch_ratio_max"], "note": "Maximum detector mismatch ratio |H1-L1|/A_safe."},
        {"metric": "detector_mismatch_ratio_mean", "value": summary["detector_mismatch_ratio_mean"], "note": "Mean detector mismatch ratio |H1-L1|/A_safe."},
        {"metric": "postpeak_rmse_sigma", "value": summary["postpeak_rmse_sigma"], "note": "RMSE between observed and modeled sigma after the peak."},
        {"metric": "postpeak_mae_sigma", "value": summary["postpeak_mae_sigma"], "note": "MAE between observed and modeled sigma after the peak."},
    ]
    return pd.DataFrame(rows)


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> str:
    metric_map = {str(r["metric"]): r["value"] for _, r in summary_df.iterrows()}

    lines: list[str] = []
    lines.append("GW150914 topological report (derived-parameter version)")
    lines.append("======================================================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This run derives the GW150914 structural-response quantities entirely from the observed "
        "mean detector strain baseline. No manual coefficients are introduced."
    )
    lines.append("")
    lines.append("Derived parameters")
    lines.append("------------------")
    for key in [
        "A_ref",
        "A_floor",
        "t_peak",
        "A_peak",
        "sigma_peak",
        "decay_threshold",
        "t_e",
        "tau_decay",
        "kappa_derived",
        "decay_status",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Response summary")
    lines.append("----------------")
    for key in [
        "sigma_obs_min",
        "sigma_obs_max",
        "propagation_proxy_max",
        "propagation_proxy_mean",
        "detector_mismatch_ratio_max",
        "detector_mismatch_ratio_mean",
        "postpeak_rmse_sigma",
        "postpeak_mae_sigma",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append(
        "The present output should be read as a structure-response quantity derived from the observed "
        "GW150914 waveform. It is not a direct full waveform replacement model."
    )
    lines.append(
        "The main consistency question is whether the post-peak observed structural contrast can be "
        "represented by a derived exponential relaxation model with a non-arbitrary decay scale."
    )
    lines.append("")
    lines.append("Output directory")
    lines.append("----------------")
    lines.append(str(output_dir))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    input_dir = project_root / "data" / "derived" / "gravitational waves" / "GW150914" / "input"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "gravitational_wave" / "GW150914" / "output" / "topological" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    h1_path = input_dir / args.h1_input
    l1_path = input_dir / args.l1_input
    merged_path = input_dir / args.merged_input
    summary_path = input_dir / args.summary_input

    for p, label in [
        (h1_path, "H1 input"),
        (l1_path, "L1 input"),
        (merged_path, "Merged input"),
        (summary_path, "Summary input"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    h1_df = pd.read_csv(h1_path)
    l1_df = pd.read_csv(l1_path)
    merged_df = pd.read_csv(merged_path)
    summary_input_df = pd.read_csv(summary_path)

    ensure_columns(
        h1_df,
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "strain_abs", "strain_sign", "strain_sq", "detector"],
        "h1_df",
    )
    ensure_columns(
        l1_df,
        ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "strain_abs", "strain_sign", "strain_sq", "detector"],
        "l1_df",
    )
    ensure_columns(
        merged_df,
        [
            "sample_index",
            "gps_time",
            "time_since_start_s",
            "time_centered_s",
            "strain_h1",
            "strain_l1",
            "strain_mean_h1_l1",
            "strain_diff_h1_l1",
            "strain_abs_mean_h1_l1",
            "strain_abs_diff_h1_l1",
            "time_from_peak_s",
        ],
        "merged_df",
    )
    ensure_columns(summary_input_df, ["metric", "value", "note"], "summary_input_df")

    topo_df, summary = build_topological_response(merged_df)
    summary_df = build_summary_df(summary)

    response_out = output_dir / "gw150914_topological_response_derived_001.csv"
    summary_out = output_dir / "gw150914_topological_summary_derived_001.csv"
    report_out = output_dir / "gw150914_topological_report_derived_001.txt"

    topo_df.to_csv(response_out, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    report_out.write_text(build_report(summary_df, output_dir), encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"response_out: {response_out}")
    print(f"summary_out: {summary_out}")
    print(f"report_out: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
