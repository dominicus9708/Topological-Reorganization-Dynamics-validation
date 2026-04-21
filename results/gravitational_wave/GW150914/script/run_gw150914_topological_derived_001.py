#!/usr/bin/env python3
"""
run_gw150914_topological_derived_002.py

Purpose
-------
Revised derived-parameter topological response pipeline for the GW150914 validation case.

Placement
---------
src/gravitational_wave/GW150914/topological/run_gw150914_topological_derived_002.py

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
- gw150914_topological_response_derived_002.csv
- gw150914_topological_summary_derived_002.csv
- gw150914_topological_report_derived_002.txt

Model notes
-----------
This version treats the observed detector outputs as limited projections of an
underlying propagating structure and therefore uses:

1) projected common amplitude:
       A_proj(t) = (|h_H1(t)| + |h_L1(t)|) / 2

2) local baseline:
       A_base(t) = rolling median of A_proj(t)

3) observed structural contrast:
       sigma_obs(t) = ln(A_proj_safe(t) / A_base_safe(t))

4) envelope:
       E(t) = rolling max of A_proj(t)

5) envelope structural contrast:
       sigma_env(t) = ln(E_safe(t) / E_ref)

6) post-peak envelope relaxation model:
       sigma_env_model(t) = sigma_env_peak * exp[-kappa_derived * (t - t_peak)]  for t >= t_peak

7) propagation proxy:
       P_env(t) = |d sigma_env / dt|

8) detector mismatch:
       M_env(t) = ||h_H1|-|h_L1|| / (E_safe(t) + epsilon_E)

All stabilizers are data-derived.
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
    parser.add_argument(
        "--baseline-window-s",
        type=float,
        default=0.25,
        help="Local rolling-median baseline window in seconds.",
    )
    parser.add_argument(
        "--envelope-window-s",
        type=float,
        default=0.02,
        help="Rolling-max envelope window in seconds.",
    )
    parser.add_argument(
        "--postpeak-window-s",
        type=float,
        default=0.2,
        help="Window length after the peak used for post-peak envelope modeling.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def get_summary_value(summary_df: pd.DataFrame, metric: str):
    mask = summary_df["metric"].astype(str) == metric
    if not mask.any():
        return None
    return summary_df.loc[mask, "value"].iloc[0]


def _samples_from_seconds(sample_rate_hz: float, seconds: float, minimum: int = 3) -> int:
    return max(minimum, int(round(sample_rate_hz * seconds)))


def build_topological_response(
    h1_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    summary_input_df: pd.DataFrame,
    baseline_window_s: float,
    envelope_window_s: float,
    postpeak_window_s: float,
) -> tuple[pd.DataFrame, dict]:
    out = merged_df.copy().sort_values("sample_index").reset_index(drop=True)

    sample_rate_hz = float(get_summary_value(summary_input_df, "h1_sample_rate_hz"))
    baseline_window_n = _samples_from_seconds(sample_rate_hz, baseline_window_s, minimum=9)
    envelope_window_n = _samples_from_seconds(sample_rate_hz, envelope_window_s, minimum=5)
    postpeak_window_n = _samples_from_seconds(sample_rate_hz, postpeak_window_s, minimum=17)

    t = out["time_centered_s"].astype(float).to_numpy()
    h1 = np.abs(out["strain_h1"].astype(float).to_numpy())
    l1 = np.abs(out["strain_l1"].astype(float).to_numpy())

    A_proj = (h1 + l1) / 2.0
    A_proj_series = pd.Series(A_proj)

    A_base = (
        A_proj_series.rolling(window=baseline_window_n, center=True, min_periods=1).median()
        .to_numpy()
    )

    A_floor = float(np.min(A_proj[A_proj > 0])) if np.any(A_proj > 0) else 1e-30
    A_proj_safe = np.maximum(A_proj, A_floor)
    A_base_safe = np.maximum(A_base, A_floor)

    sigma_obs = np.log(A_proj_safe / A_base_safe)

    # Envelope from projected common amplitude
    E = (
        A_proj_series.rolling(window=envelope_window_n, center=True, min_periods=1).max()
        .to_numpy()
    )
    E_positive = E[E > 0]
    epsilon_E = float(np.min(E_positive)) if len(E_positive) > 0 else A_floor
    E_safe = np.maximum(E, epsilon_E)

    pre_mask = t < 0.0
    if not np.any(pre_mask):
        raise ValueError("No pre-event samples found for deriving E_ref.")
    E_ref = float(np.median(E_safe[pre_mask]))

    sigma_env = np.log(E_safe / E_ref)

    peak_idx = int(np.argmax(E_safe))
    t_peak = float(t[peak_idx])
    E_peak = float(E_safe[peak_idx])
    sigma_env_peak = float(sigma_env[peak_idx])

    post_end_idx = min(len(E_safe) - 1, peak_idx + postpeak_window_n)
    post_indices = np.arange(peak_idx, post_end_idx + 1)

    decay_threshold = E_peak / np.e
    decay_candidates = [i for i in post_indices if E_safe[i] <= decay_threshold]

    if decay_candidates:
        e_idx = int(decay_candidates[0])
        decay_status = "crossed_within_postpeak_window"
    else:
        e_idx = int(post_end_idx)
        decay_status = "not_crossed_within_postpeak_window"

    t_e = float(t[e_idx])
    tau_decay = float(t_e - t_peak)

    if tau_decay <= 0:
        dt_candidates = np.diff(t)
        positive_dt = dt_candidates[dt_candidates > 0]
        if len(positive_dt) == 0:
            raise ValueError("Could not derive positive dt for fallback tau_decay.")
        tau_decay = float(np.min(positive_dt))
        decay_status = decay_status + "_fallback_to_dt"

    kappa_derived = 1.0 / tau_decay

    sigma_env_model = np.full(len(out), np.nan, dtype=float)
    post_mask = t >= t_peak
    sigma_env_model[post_mask] = sigma_env_peak * np.exp(-kappa_derived * (t[post_mask] - t_peak))

    delta_sigma_env = np.full(len(out), np.nan, dtype=float)
    delta_sigma_env[post_mask] = sigma_env[post_mask] - sigma_env_model[post_mask]

    dsigma_env_dt = np.gradient(sigma_env, t)
    propagation_proxy_env = np.abs(dsigma_env_dt)

    detector_mismatch_env = np.abs(h1 - l1) / (E_safe + epsilon_E)

    out["abs_h1"] = h1
    out["abs_l1"] = l1
    out["A_proj"] = A_proj
    out["A_proj_safe"] = A_proj_safe
    out["A_base"] = A_base
    out["A_base_safe"] = A_base_safe
    out["sigma_obs_local"] = sigma_obs
    out["E_env"] = E
    out["E_env_safe"] = E_safe
    out["sigma_env_obs"] = sigma_env
    out["sigma_env_model_postpeak"] = sigma_env_model
    out["delta_sigma_env_postpeak"] = delta_sigma_env
    out["dsigma_env_dt"] = dsigma_env_dt
    out["propagation_proxy_env"] = propagation_proxy_env
    out["detector_mismatch_env"] = detector_mismatch_env
    out["time_from_peak_env_s"] = t - t_peak

    summary = {
        "row_count": len(out),
        "sample_rate_hz": sample_rate_hz,
        "baseline_window_s": baseline_window_s,
        "baseline_window_n": baseline_window_n,
        "envelope_window_s": envelope_window_s,
        "envelope_window_n": envelope_window_n,
        "postpeak_window_s": postpeak_window_s,
        "postpeak_window_n": postpeak_window_n,
        "A_floor": A_floor,
        "E_ref": E_ref,
        "epsilon_E": epsilon_E,
        "t_peak": t_peak,
        "E_peak": E_peak,
        "sigma_env_peak": sigma_env_peak,
        "decay_threshold": decay_threshold,
        "t_e": t_e,
        "tau_decay": tau_decay,
        "kappa_derived": kappa_derived,
        "decay_status": decay_status,
        "sigma_obs_local_min": float(np.min(sigma_obs)),
        "sigma_obs_local_max": float(np.max(sigma_obs)),
        "sigma_env_min": float(np.min(sigma_env)),
        "sigma_env_max": float(np.max(sigma_env)),
        "propagation_proxy_env_max": float(np.max(propagation_proxy_env)),
        "propagation_proxy_env_mean": float(np.mean(propagation_proxy_env)),
        "detector_mismatch_env_max": float(np.max(detector_mismatch_env)),
        "detector_mismatch_env_mean": float(np.mean(detector_mismatch_env)),
        "postpeak_rmse_sigma_env": float(np.sqrt(np.nanmean((delta_sigma_env[post_mask]) ** 2))),
        "postpeak_mae_sigma_env": float(np.nanmean(np.abs(delta_sigma_env[post_mask]))),
    }
    return out, summary


def build_summary_df(summary: dict) -> pd.DataFrame:
    rows = [
        {"metric": "row_count", "value": summary["row_count"], "note": "Number of rows in revised topological response table."},
        {"metric": "sample_rate_hz", "value": summary["sample_rate_hz"], "note": "Sample rate copied from input summary."},
        {"metric": "baseline_window_s", "value": summary["baseline_window_s"], "note": "Local baseline window in seconds."},
        {"metric": "baseline_window_n", "value": summary["baseline_window_n"], "note": "Local baseline window in samples."},
        {"metric": "envelope_window_s", "value": summary["envelope_window_s"], "note": "Envelope window in seconds."},
        {"metric": "envelope_window_n", "value": summary["envelope_window_n"], "note": "Envelope window in samples."},
        {"metric": "postpeak_window_s", "value": summary["postpeak_window_s"], "note": "Post-peak fitting window in seconds."},
        {"metric": "postpeak_window_n", "value": summary["postpeak_window_n"], "note": "Post-peak fitting window in samples."},
        {"metric": "A_floor", "value": summary["A_floor"], "note": "Minimum positive projected amplitude."},
        {"metric": "E_ref", "value": summary["E_ref"], "note": "Pre-event median envelope baseline."},
        {"metric": "epsilon_E", "value": summary["epsilon_E"], "note": "Minimum positive envelope used for safe normalization."},
        {"metric": "t_peak", "value": summary["t_peak"], "note": "Derived envelope peak time."},
        {"metric": "E_peak", "value": summary["E_peak"], "note": "Derived peak envelope amplitude."},
        {"metric": "sigma_env_peak", "value": summary["sigma_env_peak"], "note": "Envelope structural contrast at the peak."},
        {"metric": "decay_threshold", "value": summary["decay_threshold"], "note": "Envelope e-fold threshold E_peak/e."},
        {"metric": "t_e", "value": summary["t_e"], "note": "First post-peak time where E(t)<=E_peak/e, if available."},
        {"metric": "tau_decay", "value": summary["tau_decay"], "note": "Derived post-peak e-fold decay time from envelope."},
        {"metric": "kappa_derived", "value": summary["kappa_derived"], "note": "Derived relaxation rate 1/tau_decay from envelope."},
        {"metric": "decay_status", "value": summary["decay_status"], "note": "Status of the envelope e-fold threshold crossing."},
        {"metric": "sigma_obs_local_min", "value": summary["sigma_obs_local_min"], "note": "Minimum local-baseline structural contrast."},
        {"metric": "sigma_obs_local_max", "value": summary["sigma_obs_local_max"], "note": "Maximum local-baseline structural contrast."},
        {"metric": "sigma_env_min", "value": summary["sigma_env_min"], "note": "Minimum envelope structural contrast."},
        {"metric": "sigma_env_max", "value": summary["sigma_env_max"], "note": "Maximum envelope structural contrast."},
        {"metric": "propagation_proxy_env_max", "value": summary["propagation_proxy_env_max"], "note": "Maximum absolute envelope-based structural propagation proxy."},
        {"metric": "propagation_proxy_env_mean", "value": summary["propagation_proxy_env_mean"], "note": "Mean absolute envelope-based structural propagation proxy."},
        {"metric": "detector_mismatch_env_max", "value": summary["detector_mismatch_env_max"], "note": "Maximum detector mismatch normalized by envelope."},
        {"metric": "detector_mismatch_env_mean", "value": summary["detector_mismatch_env_mean"], "note": "Mean detector mismatch normalized by envelope."},
        {"metric": "postpeak_rmse_sigma_env", "value": summary["postpeak_rmse_sigma_env"], "note": "RMSE between observed and modeled envelope sigma after the peak."},
        {"metric": "postpeak_mae_sigma_env", "value": summary["postpeak_mae_sigma_env"], "note": "MAE between observed and modeled envelope sigma after the peak."},
    ]
    return pd.DataFrame(rows)


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> str:
    metric_map = {str(r["metric"]): r["value"] for _, r in summary_df.iterrows()}

    lines: list[str] = []
    lines.append("GW150914 topological report (revised derived-parameter version)")
    lines.append("================================================================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This revised run treats the observed detector strains as limited projections of an underlying "
        "propagating structure. It therefore uses projected common amplitude, local baseline normalization, "
        "and envelope relaxation instead of the raw mean-strain logarithm."
    )
    lines.append("")
    lines.append("Derived parameters")
    lines.append("------------------")
    for key in [
        "sample_rate_hz",
        "baseline_window_s",
        "baseline_window_n",
        "envelope_window_s",
        "envelope_window_n",
        "postpeak_window_s",
        "postpeak_window_n",
        "E_ref",
        "t_peak",
        "E_peak",
        "sigma_env_peak",
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
        "sigma_obs_local_min",
        "sigma_obs_local_max",
        "sigma_env_min",
        "sigma_env_max",
        "propagation_proxy_env_max",
        "propagation_proxy_env_mean",
        "detector_mismatch_env_max",
        "detector_mismatch_env_mean",
        "postpeak_rmse_sigma_env",
        "postpeak_mae_sigma_env",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append(
        "The revised output should be read as a structure-response quantity derived from projected detector "
        "amplitudes and their local/envelope organization. It is not a direct full waveform replacement model."
    )
    lines.append(
        "The main consistency question is whether the post-peak envelope-based structural contrast can be "
        "represented more stably by a derived exponential relaxation model than the earlier raw-log version."
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

    topo_df, summary = build_topological_response(
        h1_df=h1_df,
        l1_df=l1_df,
        merged_df=merged_df,
        summary_input_df=summary_input_df,
        baseline_window_s=float(args.baseline_window_s),
        envelope_window_s=float(args.envelope_window_s),
        postpeak_window_s=float(args.postpeak_window_s),
    )
    summary_df = build_summary_df(summary)

    response_out = output_dir / "gw150914_topological_response_derived_002.csv"
    summary_out = output_dir / "gw150914_topological_summary_derived_002.csv"
    report_out = output_dir / "gw150914_topological_report_derived_002.txt"

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
