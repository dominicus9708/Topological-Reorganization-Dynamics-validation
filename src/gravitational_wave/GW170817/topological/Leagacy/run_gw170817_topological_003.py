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
        tau = float(np.min(pos_dt)) if len(pos_dt) else 1.0
        return e_idx, tau, "dt_fallback"
    return e_idx, tau, "end_of_window_fallback"


def shift_series_by_lag(arr: np.ndarray, lag: int) -> np.ndarray:
    s = pd.Series(arr)
    shifted = s.shift(lag)
    return shifted.bfill().ffill().to_numpy(dtype=float)


def estimate_best_lag(reference: np.ndarray, target: np.ndarray, max_lag: int) -> int:
    best_lag = 0
    best_score = -np.inf
    ref = reference - np.mean(reference)
    tgt = target - np.mean(target)
    for lag in range(-max_lag, max_lag + 1):
        shifted = shift_series_by_lag(tgt, lag)
        score = float(np.dot(ref, shifted - np.mean(shifted)))
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def rolling_envelope_abs(arr: np.ndarray, window_n: int) -> np.ndarray:
    return pd.Series(np.abs(arr)).rolling(window=window_n, center=True, min_periods=1).max().to_numpy()


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

    # ----- Stage 1: event selection gate -----
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

    A_base = pd.Series(A_gw_safe).rolling(window=baseline_window_n, center=True, min_periods=1).median().to_numpy()
    A_base_safe = np.maximum(A_base, A_floor)

    sigma_gate = np.log(A_gw_safe / np.maximum(A_base_safe + lambda_D * D_gw, A_floor))
    sigma_gate_plus = sigma_gate - float(np.min(sigma_gate)) + 1e-12
    sigma_gate_plus_smooth = pd.Series(sigma_gate_plus).rolling(window=focus_window_n, center=True, min_periods=1).mean().to_numpy()

    t_em = float(GW170817_GPS - gps_start - duration_s / 2.0)
    tau_em = max(32.0, duration_s / 256.0)
    P_em = np.exp(-((t - t_em) ** 2) / (2.0 * tau_em ** 2))

    S_event = P_em * sigma_gate_plus_smooth
    focus_idx = int(np.argmax(S_event))
    t_focus = float(t[focus_idx])
    gps_focus = float(gps_time[focus_idx])
    s_event_peak = float(S_event[focus_idx])
    sigma_gate_peak = float(sigma_gate_plus_smooth[focus_idx])

    e_idx_focus, tau_focus, focus_decay_status = first_efold_time(t, S_event, focus_idx)
    if tau_focus <= 0:
        raise ValueError("Could not derive positive tau_focus.")

    min_event_half_window_s = max(16.0, duration_s / 128.0)
    event_half_window_s = max(5.0 * tau_focus, min_event_half_window_s)
    event_half_window_n = samples_from_seconds(sample_rate_hz, event_half_window_s, minimum=33)

    event_start_idx = max(0, focus_idx - event_half_window_n)
    event_end_idx = min(len(out) - 1, focus_idx + event_half_window_n)
    event_mask = np.zeros(len(out), dtype=bool)
    event_mask[event_start_idx:event_end_idx + 1] = True

    # ----- Stage 2: reconstruction layer -----
    h1_event = h1[event_mask]
    l1_event = l1[event_mask]
    v1_event = v1[event_mask]

    max_lag_s = 0.01
    max_lag_n = samples_from_seconds(sample_rate_hz, max_lag_s, minimum=1)
    lag_l1 = estimate_best_lag(h1_event, l1_event, max_lag_n)
    lag_v1 = estimate_best_lag(h1_event, v1_event, max_lag_n)

    h1_aligned = h1.copy()
    l1_aligned = shift_series_by_lag(l1, lag_l1)
    v1_aligned = shift_series_by_lag(v1, lag_v1)

    rms_h1 = float(np.sqrt(np.mean(h1_aligned[event_mask] ** 2)))
    rms_l1 = float(np.sqrt(np.mean(l1_aligned[event_mask] ** 2)))
    rms_v1 = float(np.sqrt(np.mean(v1_aligned[event_mask] ** 2)))

    h1_norm = h1_aligned / max(rms_h1, 1e-30)
    l1_norm = l1_aligned / max(rms_l1, 1e-30)
    v1_norm = v1_aligned / max(rms_v1, 1e-30)

    H_proxy = (h1_norm + l1_norm + v1_norm) / 3.0
    Delta_res = (np.abs(h1_norm - H_proxy) + np.abs(l1_norm - H_proxy) + np.abs(v1_norm - H_proxy)) / 3.0
    Sigma_reorg = np.log((np.abs(H_proxy) + 1e-12) / (Delta_res + 1e-12))

    # ----- Stage 3: response layer -----
    envelope_window_s = max(tau_focus, min_event_half_window_s / 4.0)
    envelope_window_n = samples_from_seconds(sample_rate_hz, envelope_window_s, minimum=9)

    E_proxy = rolling_envelope_abs(H_proxy, envelope_window_n)
    ref_mask = ~event_mask
    if not np.any(ref_mask):
        ref_mask = t < t_focus
    E_ref = float(np.median(E_proxy[ref_mask]))
    E_ref = max(E_ref, 1e-12)

    sigma_topo = np.full(len(out), np.nan, dtype=float)
    sigma_topo[event_mask] = np.log(np.maximum(E_proxy[event_mask], 1e-12) / E_ref)

    sigma_reorg_event = np.full(len(out), np.nan, dtype=float)
    sigma_reorg_event[event_mask] = Sigma_reorg[event_mask]

    e_idx_env, tau_decay_env, decay_status_env = first_efold_time(t, np.maximum(E_proxy, 1e-12), focus_idx)
    if tau_decay_env <= 0:
        raise ValueError("Could not derive positive tau_decay_env.")
    kappa_derived = 1.0 / tau_decay_env

    sigma_topo_peak = float(sigma_topo[focus_idx])
    sigma_topo_model = np.full(len(out), np.nan, dtype=float)
    post_mask = (t >= t_focus) & event_mask
    if not np.any(post_mask):
        post_mask[focus_idx:event_end_idx + 1] = True
    sigma_topo_model[post_mask] = sigma_topo_peak * np.exp(-kappa_derived * (t[post_mask] - t_focus))

    delta_sigma_topo = np.full(len(out), np.nan, dtype=float)
    delta_sigma_topo[post_mask] = sigma_topo[post_mask] - sigma_topo_model[post_mask]

    propagation_proxy_event = np.abs(np.gradient(S_event, t))

    out["A_gw"] = A_gw
    out["D_gw"] = D_gw
    out["A_base"] = A_base
    out["sigma_gate"] = sigma_gate
    out["sigma_gate_plus"] = sigma_gate_plus
    out["sigma_gate_plus_smooth"] = sigma_gate_plus_smooth
    out["P_em"] = P_em
    out["S_event"] = S_event
    out["event_mask"] = event_mask.astype(int)
    out["time_from_focus_s"] = t - t_focus

    out["lag_l1_samples"] = lag_l1
    out["lag_v1_samples"] = lag_v1
    out["lag_l1_s"] = lag_l1 / sample_rate_hz
    out["lag_v1_s"] = lag_v1 / sample_rate_hz

    out["h1_aligned"] = h1_aligned
    out["l1_aligned"] = l1_aligned
    out["v1_aligned"] = v1_aligned
    out["h1_norm"] = h1_norm
    out["l1_norm"] = l1_norm
    out["v1_norm"] = v1_norm
    out["H_proxy"] = H_proxy
    out["Delta_res"] = Delta_res
    out["Sigma_reorg"] = Sigma_reorg
    out["sigma_reorg_event"] = sigma_reorg_event

    out["E_proxy"] = E_proxy
    out["sigma_topo_obs"] = sigma_topo
    out["sigma_topo_model_postfocus"] = sigma_topo_model
    out["delta_sigma_topo_postfocus"] = delta_sigma_topo
    out["propagation_proxy_event"] = propagation_proxy_event

    valid_sigma = sigma_topo[event_mask]
    valid_sigma = valid_sigma[~np.isnan(valid_sigma)]
    valid_delta = delta_sigma_topo[post_mask]
    valid_delta = valid_delta[~np.isnan(valid_delta)]

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
        "min_event_half_window_s": min_event_half_window_s,
        "event_half_window_s": event_half_window_s,
        "event_half_window_n": event_half_window_n,
        "envelope_window_s": envelope_window_s,
        "envelope_window_n": envelope_window_n,
        "focus_idx": focus_idx,
        "t_focus": t_focus,
        "gps_focus": gps_focus,
        "s_event_peak": s_event_peak,
        "sigma_gate_peak": sigma_gate_peak,
        "focus_decay_status": focus_decay_status,
        "event_start_time_s": float(t[event_start_idx]),
        "event_end_time_s": float(t[event_end_idx]),
        "event_rows": int(np.sum(event_mask)),
        "lag_l1_samples": lag_l1,
        "lag_v1_samples": lag_v1,
        "lag_l1_s": lag_l1 / sample_rate_hz,
        "lag_v1_s": lag_v1 / sample_rate_hz,
        "rms_h1_event": rms_h1,
        "rms_l1_event": rms_l1,
        "rms_v1_event": rms_v1,
        "tau_decay_env": tau_decay_env,
        "kappa_derived": kappa_derived,
        "decay_status_env": decay_status_env,
        "sigma_gate_min": float(np.min(sigma_gate)),
        "sigma_gate_max": float(np.max(sigma_gate)),
        "s_event_min": float(np.min(S_event)),
        "s_event_max": float(np.max(S_event)),
        "sigma_reorg_min": float(np.nanmin(sigma_reorg_event)),
        "sigma_reorg_max": float(np.nanmax(sigma_reorg_event)),
        "sigma_topo_min": float(np.min(valid_sigma)) if len(valid_sigma) else np.nan,
        "sigma_topo_max": float(np.max(valid_sigma)) if len(valid_sigma) else np.nan,
        "propagation_proxy_event_max": float(np.max(propagation_proxy_event)),
        "propagation_proxy_event_mean": float(np.mean(propagation_proxy_event)),
        "D_gw_mean": float(np.mean(D_gw)),
        "D_gw_max": float(np.max(D_gw)),
        "postfocus_rmse_sigma_topo": float(np.sqrt(np.mean(valid_delta ** 2))) if len(valid_delta) else np.nan,
        "postfocus_mae_sigma_topo": float(np.mean(np.abs(valid_delta))) if len(valid_delta) else np.nan,
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
        ("focus_window_s", "Derived gate smoothing window in seconds."),
        ("focus_window_n", "Derived gate smoothing window in samples."),
        ("lambda_D", "Derived detector-mismatch penalty weight."),
        ("t_em", "EM-proxy center in the current centered-time axis."),
        ("tau_em", "EM-proxy tolerance width in seconds."),
        ("tau_focus", "Derived event-score e-fold timescale."),
        ("min_event_half_window_s", "Minimum enforced half-width of the event window."),
        ("event_half_window_s", "Half-width of derived event window in seconds."),
        ("event_half_window_n", "Half-width of derived event window in samples."),
        ("envelope_window_s", "Derived event-envelope window in seconds."),
        ("envelope_window_n", "Derived event-envelope window in samples."),
        ("focus_idx", "Sample index of the event-score peak."),
        ("t_focus", "Time of the event-score peak."),
        ("gps_focus", "GPS time of the event-score peak."),
        ("s_event_peak", "Peak value of the EM-constrained event score."),
        ("sigma_gate_peak", "Smoothed gate value at the event-score peak."),
        ("focus_decay_status", "Status of the event-score e-fold detection."),
        ("event_start_time_s", "Start time of the derived event window."),
        ("event_end_time_s", "End time of the derived event window."),
        ("event_rows", "Number of rows inside the event window."),
        ("lag_l1_samples", "Best lag applied to L1 relative to H1 in samples."),
        ("lag_v1_samples", "Best lag applied to V1 relative to H1 in samples."),
        ("lag_l1_s", "Best lag applied to L1 relative to H1 in seconds."),
        ("lag_v1_s", "Best lag applied to V1 relative to H1 in seconds."),
        ("rms_h1_event", "Event-window RMS for aligned H1."),
        ("rms_l1_event", "Event-window RMS for aligned L1."),
        ("rms_v1_event", "Event-window RMS for aligned V1."),
        ("tau_decay_env", "Derived event-envelope e-fold decay time."),
        ("kappa_derived", "Derived relaxation rate 1/tau_decay_env."),
        ("decay_status_env", "Status of the event-envelope decay detection."),
        ("sigma_gate_min", "Minimum gate score."),
        ("sigma_gate_max", "Maximum gate score."),
        ("s_event_min", "Minimum event score."),
        ("s_event_max", "Maximum event score."),
        ("sigma_reorg_min", "Minimum reconstruction response inside the event window."),
        ("sigma_reorg_max", "Maximum reconstruction response inside the event window."),
        ("sigma_topo_min", "Minimum topological response inside the event window."),
        ("sigma_topo_max", "Maximum topological response inside the event window."),
        ("propagation_proxy_event_max", "Maximum absolute event-score propagation proxy."),
        ("propagation_proxy_event_mean", "Mean absolute event-score propagation proxy."),
        ("D_gw_mean", "Mean detector mismatch across the full window."),
        ("D_gw_max", "Maximum detector mismatch across the full window."),
        ("postfocus_rmse_sigma_topo", "RMSE between observed and modeled topological response after focus."),
        ("postfocus_mae_sigma_topo", "MAE between observed and modeled topological response after focus."),
    ]
    return pd.DataFrame([{"metric": k, "value": summary[k], "note": note} for k, note in order])


def build_report(summary_df: pd.DataFrame, output_dir: Path) -> str:
    metric_map = {str(r["metric"]): r["value"] for _, r in summary_df.iterrows()}
    lines = []
    lines.append("GW170817 topological report (003)")
    lines.append("================================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append("This version separates event selection from structure reconstruction. The gate score selects the event window, then an aligned common waveform proxy H_proxy and residual Delta_res are used to build the reconstruction response.")
    lines.append("")
    lines.append("Event selection layer")
    lines.append("---------------------")
    for key in [
        "sample_rate_hz", "duration_s", "gps_start", "baseline_window_s", "focus_window_s", "lambda_D",
        "t_em", "tau_em", "tau_focus", "min_event_half_window_s", "event_half_window_s",
        "t_focus", "gps_focus", "s_event_peak", "sigma_gate_peak", "focus_decay_status",
        "event_start_time_s", "event_end_time_s", "event_rows",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Reconstruction layer")
    lines.append("--------------------")
    for key in [
        "lag_l1_samples", "lag_v1_samples", "lag_l1_s", "lag_v1_s",
        "rms_h1_event", "rms_l1_event", "rms_v1_event",
        "sigma_reorg_min", "sigma_reorg_max",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Response layer")
    lines.append("--------------")
    for key in [
        "tau_decay_env", "kappa_derived", "decay_status_env",
        "sigma_topo_min", "sigma_topo_max",
        "propagation_proxy_event_max", "propagation_proxy_event_mean",
        "D_gw_mean", "D_gw_max",
        "postfocus_rmse_sigma_topo", "postfocus_mae_sigma_topo",
    ]:
        if key in metric_map:
            lines.append(f"{key}: {metric_map[key]}")
    lines.append("")
    lines.append("Interpretation status")
    lines.append("---------------------")
    lines.append("This topological stage should be read as a two-layer model: event selection by gate score, then reconstruction from the aligned common waveform proxy rather than from a single ratio alone.")
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

    response_out = output_dir / "gw170817_topological_response_003.csv"
    summary_out = output_dir / "gw170817_topological_summary_003.csv"
    report_out = output_dir / "gw170817_topological_report_003.txt"

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
