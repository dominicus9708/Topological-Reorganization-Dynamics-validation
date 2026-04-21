#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

GW170817_GPS = 1187008882.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    parser.add_argument("--h1-input", type=str, default="gw170817_h1_input_4khz_4096s.csv")
    parser.add_argument("--l1-input", type=str, default="gw170817_l1_input_4khz_4096s.csv")
    parser.add_argument("--v1-input", type=str, default="gw170817_v1_input_4khz_4096s.csv")
    parser.add_argument("--merged-input", type=str, default="gw170817_h1_l1_v1_input_4khz_4096s.csv")
    parser.add_argument("--summary-input", type=str, default="gw170817_event_summary_input.csv")
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


def samples_from_seconds(sample_rate_hz: float, seconds: float, minimum: int = 3) -> int:
    return max(minimum, int(round(sample_rate_hz * seconds)))


def first_efold_time(t: np.ndarray, arr: np.ndarray, peak_idx: int) -> tuple[int, float, str]:
    peak_val = float(arr[peak_idx])
    threshold = peak_val / np.e
    post_idx = np.where(np.arange(len(arr)) >= peak_idx)[0]
    hits = [i for i in post_idx if arr[i] <= threshold]
    if hits:
        e_idx = int(hits[0])
        return e_idx, float(t[e_idx] - t[peak_idx]), "crossed"

    tail = arr[peak_idx:]
    if len(tail) > 2:
        local_min = []
        for j in range(1, len(tail) - 1):
            if tail[j] <= tail[j - 1] and tail[j] <= tail[j + 1]:
                local_min.append(peak_idx + j)
        if local_min:
            e_idx = int(local_min[0])
            tau = float(t[e_idx] - t[peak_idx])
            if tau > 0:
                return e_idx, tau, "local_minimum_fallback"

    e_idx = len(arr) - 1
    tau = float(t[e_idx] - t[peak_idx])
    if tau <= 0:
        dt = np.diff(t)
        pos_dt = dt[dt > 0]
        return e_idx, float(np.min(pos_dt)) if len(pos_dt) else 1.0, "dt_fallback"
    return e_idx, tau, "end_of_window_fallback"


def build_topological_response(merged_df: pd.DataFrame, summary_input_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = merged_df.copy().sort_values("sample_index").reset_index(drop=True)

    sample_rate_hz = float(get_summary_value(summary_input_df, "h1_sample_rate_hz"))
    duration_s = float(get_summary_value(summary_input_df, "h1_duration_s"))
    gps_start = float(get_summary_value(summary_input_df, "h1_gps_start"))

    t = out["time_centered_s"].astype(float).to_numpy()
    gps_time = out["gps_time"].astype(float).to_numpy()

    h1 = out["strain_h1"].astype(float).to_numpy()
    l1 = out["strain_l1"].astype(float).to_numpy()
    v1 = out["strain_v1"].astype(float).to_numpy()

    abs_h1 = np.abs(h1)
    abs_l1 = np.abs(l1)
    abs_v1 = np.abs(v1)

    A_gw = (abs_h1 + abs_l1 + abs_v1) / 3.0
    D_gw = (np.abs(h1 - l1) + np.abs(h1 - v1) + np.abs(l1 - v1)) / 3.0

    A_positive = A_gw[A_gw > 0]
    A_floor = float(np.min(A_positive)) if len(A_positive) else 1e-30
    A_gw_safe = np.maximum(A_gw, A_floor)

    baseline_window_s = duration_s / 40.0
    baseline_window_n = samples_from_seconds(sample_rate_hz, baseline_window_s, minimum=33)

    focus_window_s = duration_s / 400.0
    focus_window_n = samples_from_seconds(sample_rate_hz, focus_window_s, minimum=9)

    D_mean = float(np.mean(D_gw))
    A_mean = float(np.mean(A_gw_safe))
    lambda_D = A_mean / D_mean if D_mean > 0 else 1.0

    A_series = pd.Series(A_gw_safe)
    A_base = A_series.rolling(window=baseline_window_n, center=True, min_periods=1).median().to_numpy()
    A_base_safe = np.maximum(A_base, A_floor)

    denom_focus = np.maximum(A_base_safe + lambda_D * D_gw, A_floor)
    sigma_gw_focus = np.log(A_gw_safe / denom_focus)

    sigma_gw_focus_smooth = (
        pd.Series(sigma_gw_focus).rolling(window=focus_window_n, center=True, min_periods=1).mean().to_numpy()
    )

    t_em = float(GW170817_GPS - gps_start - duration_s / 2.0)
    tau_em = max(8.0, duration_s / 512.0)
    P_em = np.exp(-((t - t_em) ** 2) / (2.0 * tau_em ** 2))

    S_event = P_em * sigma_gw_focus_smooth

    focus_idx = int(np.argmax(S_event))
    t_focus = float(t[focus_idx])
    gps_focus = float(gps_time[focus_idx])
    s_event_peak = float(S_event[focus_idx])
    sigma_gw_focus_peak = float(sigma_gw_focus_smooth[focus_idx])

    s_event_shifted = S_event - np.min(S_event) + 1e-12
    e_idx_focus, tau_focus, focus_decay_status = first_efold_time(t, s_event_shifted, focus_idx)
    if tau_focus <= 0:
        raise ValueError("Could not derive positive tau_focus.")

    event_half_window_s = 5.0 * tau_focus
    event_half_window_n = samples_from_seconds(sample_rate_hz, event_half_window_s, minimum=33)

    event_start_idx = max(0, focus_idx - event_half_window_n)
    event_end_idx = min(len(out) - 1, focus_idx + event_half_window_n)
    event_mask = np.zeros(len(out), dtype=bool)
    event_mask[event_start_idx:event_end_idx + 1] = True

    ref_mask = ~event_mask
    if not np.any(ref_mask):
        ref_mask = t < t_focus
    E_ref = float(np.median(A_gw_safe[ref_mask]))

    envelope_window_s = tau_focus
    envelope_window_n = samples_from_seconds(sample_rate_hz, envelope_window_s, minimum=9)

    E_event = (
        pd.Series(A_gw_safe).rolling(window=envelope_window_n, center=True, min_periods=1).max().to_numpy()
    )
    E_positive = E_event[E_event > 0]
    E_floor = float(np.min(E_positive)) if len(E_positive) else A_floor
    E_event_safe = np.maximum(E_event, E_floor)

    sigma_env = np.full(len(out), np.nan, dtype=float)
    sigma_env[event_mask] = np.log(E_event_safe[event_mask] / max(E_ref, E_floor))

    e_idx_env, tau_decay_env, decay_status_env = first_efold_time(t, E_event_safe, focus_idx)
    if tau_decay_env <= 0:
        raise ValueError("Could not derive positive tau_decay_env.")
    kappa_derived = 1.0 / tau_decay_env

    sigma_env_peak = float(sigma_env[focus_idx])

    sigma_env_model = np.full(len(out), np.nan, dtype=float)
    post_mask = (t >= t_focus) & event_mask
    sigma_env_model[post_mask] = sigma_env_peak * np.exp(-kappa_derived * (t[post_mask] - t_focus))

    delta_sigma_env = np.full(len(out), np.nan, dtype=float)
    delta_sigma_env[post_mask] = sigma_env[post_mask] - sigma_env_model[post_mask]

    ds_event_dt = np.gradient(S_event, t)
    propagation_proxy_event = np.abs(ds_event_dt)

    out["abs_h1"] = abs_h1
    out["abs_l1"] = abs_l1
    out["abs_v1"] = abs_v1
    out["A_gw"] = A_gw
    out["A_gw_safe"] = A_gw_safe
    out["D_gw"] = D_gw
    out["lambda_D"] = lambda_D
    out["A_base"] = A_base
    out["A_base_safe"] = A_base_safe
    out["sigma_gw_focus"] = sigma_gw_focus
    out["sigma_gw_focus_smooth"] = sigma_gw_focus_smooth
    out["P_em"] = P_em
    out["S_event"] = S_event
    out["E_event"] = E_event
    out["E_event_safe"] = E_event_safe
    out["sigma_env_obs"] = sigma_env
    out["sigma_env_model_postfocus"] = sigma_env_model
    out["delta_sigma_env_postfocus"] = delta_sigma_env
    out["propagation_proxy_event"] = propagation_proxy_event
    out["event_mask"] = event_mask.astype(int)
    out["time_from_focus_s"] = t - t_focus

    summary = {
        "row_count": len(out),
        "sample_rate_hz": sample_rate_hz,
        "duration_s": duration_s,
        "gps_start": gps_start,
        "baseline_window_s": baseline_window_s,
        "baseline_window_n": baseline_window_n,
        "focus_window_s": focus_window_s,
        "focus_window_n": focus_window_n,
        "lambda_D": lambda_D,
        "t_em": t_em,
        "tau_em": tau_em,
        "tau_focus": tau_focus,
        "event_half_window_s": event_half_window_s,
        "event_half_window_n": event_half_window_n,
        "envelope_window_s": envelope_window_s,
        "envelope_window_n": envelope_window_n,
        "A_floor": A_floor,
        "E_ref": E_ref,
        "E_floor": E_floor,
        "focus_idx": focus_idx,
        "t_focus": t_focus,
        "gps_focus": gps_focus,
        "s_event_peak": s_event_peak,
        "sigma_gw_focus_peak": sigma_gw_focus_peak,
        "focus_decay_status": focus_decay_status,
        "event_start_time_s": float(t[event_start_idx]),
        "event_end_time_s": float(t[event_end_idx]),
        "event_rows": int(np.sum(event_mask)),
        "tau_decay_env": tau_decay_env,
        "kappa_derived": kappa_derived,
        "decay_status_env": decay_status_env,
        "sigma_gw_focus_min": float(np.min(sigma_gw_focus)),
        "sigma_gw_focus_max": float(np.max(sigma_gw_focus)),
        "s_event_min": float(np.min(S_event)),
        "s_event_max": float(np.max(S_event)),
        "sigma_env_min": float(np.nanmin(sigma_env)),
        "sigma_env_max": float(np.nanmax(sigma_env)),
        "propagation_proxy_event_max": float(np.max(propagation_proxy_event)),
        "propagation_proxy_event_mean": float(np.mean(propagation_proxy_event)),
        "D_gw_mean": float(np.mean(D_gw)),
        "D_gw_max": float(np.max(D_gw)),
        "postfocus_rmse_sigma_env": float(np.sqrt(np.nanmean((delta_sigma_env[post_mask]) ** 2))),
        "postfocus_mae_sigma_env": float(np.nanmean(np.abs(delta_sigma_env[post_mask]))),
    }
    return out, summary


def build_summary_df(summary: dict) -> pd.DataFrame:
    order = [
        ("row_count", "Number of rows in GW170817 topological response table."),
        ("sample_rate_hz", "Sample rate copied from input summary."),
        ("duration_s", "Duration copied from input summary."),
        ("gps_start", "GPS start copied from input summary."),
        ("baseline_window_s", "Derived local baseline window in seconds."),
        ("baseline_window_n", "Derived local baseline window in samples."),
        ("focus_window_s", "Derived GW-focus smoothing window in seconds."),
        ("focus_window_n", "Derived GW-focus smoothing window in samples."),
        ("lambda_D", "Derived detector-mismatch penalty weight."),
        ("t_em", "EM-proxy center in the current centered-time axis."),
        ("tau_em", "EM-proxy tolerance width in seconds."),
        ("tau_focus", "Derived event-score e-fold timescale."),
        ("event_half_window_s", "Half-width of derived event window in seconds."),
        ("event_half_window_n", "Half-width of derived event window in samples."),
        ("envelope_window_s", "Derived event-envelope window in seconds."),
        ("envelope_window_n", "Derived event-envelope window in samples."),
        ("A_floor", "Minimum positive detector-common GW amplitude."),
        ("E_ref", "Reference amplitude from outside the event window."),
        ("E_floor", "Minimum positive event-envelope amplitude."),
        ("focus_idx", "Sample index of the event-score peak."),
        ("t_focus", "Time of the event-score peak."),
        ("gps_focus", "GPS time of the event-score peak."),
        ("s_event_peak", "Peak value of the EM-constrained event score."),
        ("sigma_gw_focus_peak", "Smoothed GW-focus value at the event-score peak."),
        ("focus_decay_status", "Status of the event-score e-fold detection."),
        ("event_start_time_s", "Start time of the derived event window."),
        ("event_end_time_s", "End time of the derived event window."),
        ("event_rows", "Number of rows inside the event window."),
        ("tau_decay_env", "Derived event-envelope e-fold decay time."),
        ("kappa_derived", "Derived relaxation rate 1/tau_decay_env."),
        ("decay_status_env", "Status of the event-envelope decay detection."),
        ("sigma_gw_focus_min", "Minimum raw GW-focus score."),
        ("sigma_gw_focus_max", "Maximum raw GW-focus score."),
        ("s_event_min", "Minimum EM-constrained event score."),
        ("s_event_max", "Maximum EM-constrained event score."),
        ("sigma_env_min", "Minimum event-envelope response value."),
        ("sigma_env_max", "Maximum event-envelope response value."),
        ("propagation_proxy_event_max", "Maximum absolute event-score propagation proxy."),
        ("propagation_proxy_event_mean", "Mean absolute event-score propagation proxy."),
        ("D_gw_mean", "Mean detector mismatch across the full window."),
        ("D_gw_max", "Maximum detector mismatch across the full window."),
        ("postfocus_rmse_sigma_env", "RMSE between observed and modeled event-envelope response after focus."),
        ("postfocus_mae_sigma_env", "MAE between observed and modeled event-envelope response after focus."),
    ]
    return pd.DataFrame([{"metric": k, "value": summary[k], "note": note} for k, note in order])


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> str:
    metric_map = {str(r["metric"]): r["value"] for _, r in summary_df.iterrows()}
    lines = []
    lines.append("GW170817 topological report (002)")
    lines.append("================================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append("This version constrains the detector-common GW event score with an EM-side event proxy before defining the event window and topological response.")
    lines.append("")
    lines.append("Derived event-focus quantities")
    lines.append("-----------------------------")
    for key in [
        "sample_rate_hz", "duration_s", "gps_start", "baseline_window_s", "focus_window_s", "lambda_D",
        "t_em", "tau_em", "tau_focus", "event_half_window_s", "envelope_window_s", "t_focus", "gps_focus",
        "s_event_peak", "sigma_gw_focus_peak", "focus_decay_status", "event_start_time_s", "event_end_time_s",
        "event_rows", "tau_decay_env", "kappa_derived", "decay_status_env",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Response summary")
    lines.append("----------------")
    for key in [
        "sigma_gw_focus_min", "sigma_gw_focus_max", "s_event_min", "s_event_max", "sigma_env_min", "sigma_env_max",
        "propagation_proxy_event_max", "propagation_proxy_event_mean", "D_gw_mean", "D_gw_max",
        "postfocus_rmse_sigma_env", "postfocus_mae_sigma_env",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append("This topological stage should be read as an EM-constrained event-window-centered response model. EM is used as an external event proxy, not as a direct strain-like waveform.")
    lines.append("")
    lines.append("Output directory")
    lines.append("----------------")
    lines.append(str(output_dir))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    input_dir = project_root / "data" / "derived" / "gravitational waves" / "GW170817" / "input"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "gravitational_wave" / "GW170817" / "output" / "topological" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    h1_path = input_dir / args.h1_input
    l1_path = input_dir / args.l1_input
    v1_path = input_dir / args.v1_input
    merged_path = input_dir / args.merged_input
    summary_path = input_dir / args.summary_input

    for p, label in [(h1_path, "H1 input"), (l1_path, "L1 input"), (v1_path, "V1 input"), (merged_path, "Merged input"), (summary_path, "Summary input")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    h1_df = pd.read_csv(h1_path)
    l1_df = pd.read_csv(l1_path)
    v1_df = pd.read_csv(v1_path)
    merged_df = pd.read_csv(merged_path)
    summary_input_df = pd.read_csv(summary_path)

    ensure_columns(h1_df, ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "detector", "strain_abs", "strain_sign", "strain_sq"], "h1_df")
    ensure_columns(l1_df, ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "detector", "strain_abs", "strain_sign", "strain_sq"], "l1_df")
    ensure_columns(v1_df, ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain", "detector", "strain_abs", "strain_sign", "strain_sq"], "v1_df")
    ensure_columns(merged_df, ["sample_index", "gps_time", "time_since_start_s", "time_centered_s", "strain_h1", "strain_l1", "strain_v1", "strain_mean_h1_l1", "strain_mean_h1_l1_v1", "strain_abs_mean_h1_l1", "strain_abs_mean_h1_l1_v1", "strain_abs_diff_h1_l1", "strain_abs_diff_h1_v1", "strain_abs_diff_l1_v1", "time_from_peak_s"], "merged_df")
    ensure_columns(summary_input_df, ["metric", "value", "note"], "summary_input_df")

    topo_df, summary = build_topological_response(merged_df, summary_input_df)
    summary_df = build_summary_df(summary)

    response_out = output_dir / "gw170817_topological_response_002.csv"
    summary_out = output_dir / "gw170817_topological_summary_002.csv"
    report_out = output_dir / "gw170817_topological_report_002.txt"

    topo_df.to_csv(response_out, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    report_out.write_text(build_report(summary_df, output_dir), encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"response_out: {response_out}")
    print(f"summary_out: {summary_out}")
    print(f"report_out: {report_out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
