#!/usr/bin/env python3
"""
run_gw150914_topological_derived_003.py

Purpose
-------
Fully derived-parameter topological response pipeline for the GW150914 validation case.

Placement
---------
src/gravitational_wave/GW150914/topological/run_gw150914_topological_derived_003.py

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
- gw150914_topological_response_derived_003.csv
- gw150914_topological_summary_derived_003.csv
- gw150914_topological_report_derived_003.txt

Model notes
-----------
This version derives even the analysis windows from the event itself.

Stage A. Derived common projected amplitude:
    A_proj(t) = (|h_H1(t)| + |h_L1(t)|) / 2

Stage B. First-pass peak and decay from raw projected amplitude:
    t_peak_raw = argmax A_proj
    t_e_raw = first t >= t_peak_raw such that A_proj(t) <= A_peak_raw / e
    tau_decay_raw = t_e_raw - t_peak_raw

Stage C. Derived windows:
    baseline_window_s = |t_peak_raw| / 50
    envelope_window_s = tau_decay_raw / 2
    postpeak_window_s = 5 * tau_decay_raw

Stage D. Revised observables using derived windows:
    A_base(t) = rolling median of A_proj(t) over baseline_window
    E(t)      = rolling max of A_proj(t) over envelope_window
    sigma_local(t) = ln(A_proj_safe / A_base_safe)
    sigma_env(t)   = ln(E_safe / E_ref)

Stage E. Post-peak envelope relaxation:
    sigma_env_model(t) = sigma_env_peak * exp[-kappa_derived * (t - t_peak_env)],  t >= t_peak_env
    where kappa_derived = 1 / tau_decay_env and tau_decay_env is derived from E(t)

All floors/stabilizers are data-derived.
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


def get_summary_value(summary_df: pd.DataFrame, metric: str):
    mask = summary_df["metric"].astype(str) == metric
    if not mask.any():
        return None
    return summary_df.loc[mask, "value"].iloc[0]


def _samples_from_seconds(sample_rate_hz: float, seconds: float, minimum: int = 3) -> int:
    return max(minimum, int(round(sample_rate_hz * seconds)))


def _first_e_fold_time(t: np.ndarray, A: np.ndarray, peak_idx: int) -> tuple[int, float, str]:
    A_peak = float(A[peak_idx])
    threshold = A_peak / np.e
    post_idx = np.where(np.arange(len(A)) >= peak_idx)[0]
    decay_candidates = [i for i in post_idx if A[i] <= threshold]
    if decay_candidates:
        e_idx = int(decay_candidates[0])
        return e_idx, float(t[e_idx] - t[peak_idx]), "crossed"
    # fallback: first local minimum after peak, else end of window
    if len(post_idx) > 2:
        tail = A[peak_idx:]
        local_min_candidates = []
        for j in range(1, len(tail) - 1):
            if tail[j] <= tail[j - 1] and tail[j] <= tail[j + 1]:
                local_min_candidates.append(peak_idx + j)
        if local_min_candidates:
            e_idx = int(local_min_candidates[0])
            tau = float(t[e_idx] - t[peak_idx])
            if tau > 0:
                return e_idx, tau, "local_minimum_fallback"
    e_idx = len(A) - 1
    tau = float(t[e_idx] - t[peak_idx])
    if tau <= 0:
        dt = np.diff(t)
        pos_dt = dt[dt > 0]
        tau = float(np.min(pos_dt)) if len(pos_dt) else 1.0
        return e_idx, tau, "dt_fallback"
    return e_idx, tau, "end_of_window_fallback"


def build_topological_response(
    h1_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    summary_input_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    out = merged_df.copy().sort_values("sample_index").reset_index(drop=True)

    sample_rate_hz = float(get_summary_value(summary_input_df, "h1_sample_rate_hz"))
    t = out["time_centered_s"].astype(float).to_numpy()
    h1 = np.abs(out["strain_h1"].astype(float).to_numpy())
    l1 = np.abs(out["strain_l1"].astype(float).to_numpy())

    A_proj = (h1 + l1) / 2.0
    A_positive = A_proj[A_proj > 0]
    A_floor = float(np.min(A_positive)) if len(A_positive) else 1e-30
    A_proj_safe = np.maximum(A_proj, A_floor)

    # Stage B: first-pass raw peak/decay
    peak_idx_raw = int(np.argmax(A_proj_safe))
    t_peak_raw = float(t[peak_idx_raw])
    A_peak_raw = float(A_proj_safe[peak_idx_raw])

    e_idx_raw, tau_decay_raw, decay_status_raw = _first_e_fold_time(t, A_proj_safe, peak_idx_raw)
    if tau_decay_raw <= 0:
        raise ValueError("Could not derive positive tau_decay_raw.")

    # Stage C: derive windows from the event itself
    baseline_window_s = abs(t_peak_raw) / 50.0
    envelope_window_s = tau_decay_raw / 2.0
    postpeak_window_s = 5.0 * tau_decay_raw

    baseline_window_n = _samples_from_seconds(sample_rate_hz, baseline_window_s, minimum=9)
    envelope_window_n = _samples_from_seconds(sample_rate_hz, envelope_window_s, minimum=5)
    postpeak_window_n = _samples_from_seconds(sample_rate_hz, postpeak_window_s, minimum=17)

    A_proj_series = pd.Series(A_proj_safe)

    # Local baseline
    A_base = A_proj_series.rolling(window=baseline_window_n, center=True, min_periods=1).median().to_numpy()
    A_base_safe = np.maximum(A_base, A_floor)
    sigma_local = np.log(A_proj_safe / A_base_safe)

    # Envelope
    E = A_proj_series.rolling(window=envelope_window_n, center=True, min_periods=1).max().to_numpy()
    E_positive = E[E > 0]
    epsilon_E = float(np.min(E_positive)) if len(E_positive) else A_floor
    E_safe = np.maximum(E, epsilon_E)

    pre_mask = t < 0.0
    if not np.any(pre_mask):
        raise ValueError("No pre-event samples found for deriving E_ref.")
    E_ref = float(np.median(E_safe[pre_mask]))
    sigma_env = np.log(E_safe / E_ref)

    # Envelope peak and derived decay
    peak_idx_env = int(np.argmax(E_safe))
    t_peak_env = float(t[peak_idx_env])
    E_peak = float(E_safe[peak_idx_env])
    sigma_env_peak = float(sigma_env[peak_idx_env])

    post_end_idx = min(len(E_safe) - 1, peak_idx_env + postpeak_window_n)
    # restrict decay search to postpeak window
    A_post_slice = E_safe[: post_end_idx + 1]
    e_idx_env, tau_decay_env, decay_status_env = _first_e_fold_time(t[: post_end_idx + 1], A_post_slice, peak_idx_env)
    t_e_env = float(t[e_idx_env])
    if tau_decay_env <= 0:
        raise ValueError("Could not derive positive tau_decay_env.")

    kappa_derived = 1.0 / tau_decay_env

    sigma_env_model = np.full(len(out), np.nan, dtype=float)
    post_mask = t >= t_peak_env
    sigma_env_model[post_mask] = sigma_env_peak * np.exp(-kappa_derived * (t[post_mask] - t_peak_env))

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
    out["sigma_obs_local"] = sigma_local
    out["E_env"] = E
    out["E_env_safe"] = E_safe
    out["sigma_env_obs"] = sigma_env
    out["sigma_env_model_postpeak"] = sigma_env_model
    out["delta_sigma_env_postpeak"] = delta_sigma_env
    out["dsigma_env_dt"] = dsigma_env_dt
    out["propagation_proxy_env"] = propagation_proxy_env
    out["detector_mismatch_env"] = detector_mismatch_env
    out["time_from_peak_env_s"] = t - t_peak_env

    summary = {
        "row_count": len(out),
        "sample_rate_hz": sample_rate_hz,
        "t_peak_raw": t_peak_raw,
        "A_peak_raw": A_peak_raw,
        "tau_decay_raw": tau_decay_raw,
        "decay_status_raw": decay_status_raw,
        "baseline_window_s": baseline_window_s,
        "baseline_window_n": baseline_window_n,
        "envelope_window_s": envelope_window_s,
        "envelope_window_n": envelope_window_n,
        "postpeak_window_s": postpeak_window_s,
        "postpeak_window_n": postpeak_window_n,
        "A_floor": A_floor,
        "E_ref": E_ref,
        "epsilon_E": epsilon_E,
        "t_peak_env": t_peak_env,
        "E_peak": E_peak,
        "sigma_env_peak": sigma_env_peak,
        "t_e_env": t_e_env,
        "tau_decay_env": tau_decay_env,
        "kappa_derived": kappa_derived,
        "decay_status_env": decay_status_env,
        "sigma_obs_local_min": float(np.min(sigma_local)),
        "sigma_obs_local_max": float(np.max(sigma_local)),
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
    order = [
        ("row_count", "Number of rows in fully derived topological response table."),
        ("sample_rate_hz", "Sample rate copied from input summary."),
        ("t_peak_raw", "First-pass raw projected-amplitude peak time."),
        ("A_peak_raw", "First-pass raw projected-amplitude peak."),
        ("tau_decay_raw", "First-pass raw projected-amplitude e-fold decay time."),
        ("decay_status_raw", "Status of the raw first-pass decay detection."),
        ("baseline_window_s", "Derived local baseline window in seconds."),
        ("baseline_window_n", "Derived local baseline window in samples."),
        ("envelope_window_s", "Derived envelope window in seconds."),
        ("envelope_window_n", "Derived envelope window in samples."),
        ("postpeak_window_s", "Derived post-peak fitting window in seconds."),
        ("postpeak_window_n", "Derived post-peak fitting window in samples."),
        ("A_floor", "Minimum positive projected amplitude."),
        ("E_ref", "Pre-event median envelope baseline."),
        ("epsilon_E", "Minimum positive envelope used for safe normalization."),
        ("t_peak_env", "Derived envelope peak time."),
        ("E_peak", "Derived peak envelope amplitude."),
        ("sigma_env_peak", "Envelope structural contrast at the peak."),
        ("t_e_env", "Derived envelope e-fold decay crossing time."),
        ("tau_decay_env", "Derived envelope e-fold decay time."),
        ("kappa_derived", "Derived relaxation rate 1/tau_decay_env."),
        ("decay_status_env", "Status of the envelope decay detection."),
        ("sigma_obs_local_min", "Minimum local-baseline structural contrast."),
        ("sigma_obs_local_max", "Maximum local-baseline structural contrast."),
        ("sigma_env_min", "Minimum envelope structural contrast."),
        ("sigma_env_max", "Maximum envelope structural contrast."),
        ("propagation_proxy_env_max", "Maximum absolute envelope-based structural propagation proxy."),
        ("propagation_proxy_env_mean", "Mean absolute envelope-based structural propagation proxy."),
        ("detector_mismatch_env_max", "Maximum detector mismatch normalized by envelope."),
        ("detector_mismatch_env_mean", "Mean detector mismatch normalized by envelope."),
        ("postpeak_rmse_sigma_env", "RMSE between observed and modeled envelope sigma after the peak."),
        ("postpeak_mae_sigma_env", "MAE between observed and modeled envelope sigma after the peak."),
    ]
    rows = [{"metric": k, "value": summary[k], "note": note} for k, note in order]
    return pd.DataFrame(rows)


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> str:
    metric_map = {str(r["metric"]): r["value"] for _, r in summary_df.iterrows()}
    lines: list[str] = []
    lines.append("GW150914 topological report (fully derived version)")
    lines.append("==================================================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This version derives not only the structural quantities but also the analysis windows from the event itself. "
        "It uses projected common amplitude, local baseline normalization, and envelope relaxation."
    )
    lines.append("")
    lines.append("Derived event scales")
    lines.append("--------------------")
    for key in [
        "t_peak_raw", "A_peak_raw", "tau_decay_raw", "decay_status_raw",
        "baseline_window_s", "baseline_window_n",
        "envelope_window_s", "envelope_window_n",
        "postpeak_window_s", "postpeak_window_n",
        "E_ref", "t_peak_env", "E_peak", "sigma_env_peak",
        "t_e_env", "tau_decay_env", "kappa_derived", "decay_status_env",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Response summary")
    lines.append("----------------")
    for key in [
        "sigma_obs_local_min", "sigma_obs_local_max",
        "sigma_env_min", "sigma_env_max",
        "propagation_proxy_env_max", "propagation_proxy_env_mean",
        "detector_mismatch_env_max", "detector_mismatch_env_mean",
        "postpeak_rmse_sigma_env", "postpeak_mae_sigma_env",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append(
        "The fully derived output should be read as a structure-response quantity derived from projected detector amplitudes, "
        "their event-derived local baseline, and their event-derived envelope organization."
    )
    lines.append(
        "The main consistency question is whether this fully derived envelope-based construction improves the post-peak "
        "relaxation consistency and reduces spurious small-amplitude sensitivity."
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
    )
    summary_df = build_summary_df(summary)

    response_out = output_dir / "gw150914_topological_response_derived_003.csv"
    summary_out = output_dir / "gw150914_topological_summary_derived_003.csv"
    report_out = output_dir / "gw150914_topological_report_derived_003.txt"

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
