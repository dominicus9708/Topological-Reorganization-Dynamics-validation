#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    parser.add_argument("--window-padding-s", type=float, default=60.0, help="Extra seconds around the topological event window.")
    parser.add_argument("--chunk-size", type=int, default=200000, help="CSV chunk size for selective loading.")
    parser.add_argument("--smooth-window-s", type=float, default=10.0, help="Rolling smoothing window in seconds for standard representative quantities.")
    parser.add_argument("--reorg-smooth-window-s", type=float, default=5.0, help="Rolling smoothing window in seconds for Sigma_reorg diagnostic display.")
    return parser.parse_args()


def resolve_result_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    direct_csv = list(base_dir.glob("*.csv"))
    direct_txt = list(base_dir.glob("*.txt"))
    if direct_csv or direct_txt:
        return base_dir
    dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if dirs:
        return sorted(dirs)[-1]
    return base_dir


def find_single_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched pattern '{pattern}' in {folder}")
    return matches[0]


def read_metric_csv(path: Path) -> dict[str, float | str]:
    df = pd.read_csv(path)
    if "metric" not in df.columns or "value" not in df.columns:
        raise ValueError(f"Summary file missing metric/value columns: {path}")
    out: dict[str, float | str] = {}
    for _, row in df.iterrows():
        key = str(row["metric"])
        value = row["value"]
        try:
            out[key] = float(value)
        except Exception:
            out[key] = str(value)
    return out


def parse_standard_report(path: Path) -> dict[str, float | str]:
    metric_map: dict[str, float | str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_]+):\s*(.+?)\s*$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        try:
            metric_map[key] = float(raw)
        except Exception:
            metric_map[key] = raw
    return metric_map


def normalize_series(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full_like(arr, np.nan)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if math.isclose(amax, amin):
        out = np.zeros_like(arr, dtype=float)
        out[~finite] = np.nan
        return out
    return (arr - amin) / (amax - amin)


def load_windowed_columns(csv_path: Path, usecols: list[str], time_col: str, t_min: float, t_max: float, chunk_size: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
        mask = (chunk[time_col].astype(float) >= t_min) & (chunk[time_col].astype(float) <= t_max)
        part = chunk.loc[mask].copy()
        if not part.empty:
            chunks.append(part)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(time_col).reset_index(drop=True)
    return out


def rolling_median(series: pd.Series, window_n: int) -> pd.Series:
    return series.rolling(window=window_n, center=True, min_periods=1).median()


def build_metric_comparison(standard_metrics: dict[str, float | str], topo_metrics: dict[str, float | str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    def add_row(metric: str, standard_key: str | None, topo_key: str | None, note: str) -> None:
        rows.append(
            {
                "metric": metric,
                "standard_value": standard_metrics.get(standard_key) if standard_key else "",
                "topological_value": topo_metrics.get(topo_key) if topo_key else "",
                "note": note,
            }
        )

    add_row("event_centered_time", "merged_peak_time_centered_s", "t_focus", "Standard merged peak time versus topological focus time.")
    add_row("event_gps_time", "standard_peak_gps_time", "gps_focus", "Standard peak GPS time versus topological focus GPS time.")
    add_row("event_window_half_width_s", None, "event_half_window_s", "Topological event-window half-width; no direct standard counterpart.")
    add_row("topological_decay_tau_s", None, "tau_decay_alpha", "Decay timescale of structural-dimension-gap response.")
    add_row("topological_decay_kappa", None, "kappa_alpha", "Decay rate of structural-dimension-gap response.")
    add_row("topological_postfocus_rmse", None, "postfocus_rmse_sigma_alpha", "Post-focus RMSE of structural-dimension-gap response.")
    add_row("topological_postfocus_mae", None, "postfocus_mae_sigma_alpha", "Post-focus MAE of structural-dimension-gap response.")
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    standard_base = project_root / "results" / "gravitational_wave" / "GW170817" / "output" / "standard"
    topo_base = project_root / "results" / "gravitational_wave" / "GW170817" / "output" / "topological"
    integration_base = project_root / "results" / "gravitational_wave" / "GW170817" / "output" / "Integration"

    standard_dir = resolve_result_dir(standard_base)
    topo_dir = resolve_result_dir(topo_base)

    standard_report = find_single_file(standard_dir, "gw170817_standard_report.txt")
    standard_merged = find_single_file(standard_dir, "gw170817_standard_merged_full.csv")
    topo_summary = find_single_file(topo_dir, "gw170817_topological_summary_*.csv")
    topo_response = find_single_file(topo_dir, "gw170817_topological_response_*.csv")

    standard_metrics = parse_standard_report(standard_report)
    topo_metrics = read_metric_csv(topo_summary)

    sample_rate_hz = float(topo_metrics.get("sample_rate_hz", 4096.0))
    t_min = float(topo_metrics["event_start_time_s"]) - float(args.window_padding_s)
    t_max = float(topo_metrics["event_end_time_s"]) + float(args.window_padding_s)
    focus_t = float(topo_metrics["t_focus"])

    standard_df = load_windowed_columns(
        csv_path=standard_merged,
        usecols=["time_centered_s", "strain_mean_h1_l1_v1"],
        time_col="time_centered_s",
        t_min=t_min,
        t_max=t_max,
        chunk_size=args.chunk_size,
    )
    topo_df = load_windowed_columns(
        csv_path=topo_response,
        usecols=["time_centered_s", "S_event", "Sigma_reorg", "sigma_alpha_obs", "sigma_alpha_model_postfocus", "event_mask"],
        time_col="time_centered_s",
        t_min=t_min,
        t_max=t_max,
        chunk_size=args.chunk_size,
    )

    smooth_n = max(5, int(round(sample_rate_hz * args.smooth_window_s)))
    reorg_smooth_n = max(5, int(round(sample_rate_hz * args.reorg_smooth_window_s)))

    standard_df["abs_mean_standard"] = np.abs(standard_df["strain_mean_h1_l1_v1"].astype(float))
    standard_df["abs_mean_standard_smooth"] = rolling_median(standard_df["abs_mean_standard"], smooth_n)

    topo_df["S_event_norm"] = normalize_series(topo_df["S_event"].astype(float).to_numpy())
    topo_df["sigma_alpha_norm"] = normalize_series(topo_df["sigma_alpha_obs"].astype(float).to_numpy())
    topo_df["sigma_alpha_model_norm"] = normalize_series(topo_df["sigma_alpha_model_postfocus"].astype(float).to_numpy())
    topo_df["Sigma_reorg_smooth"] = rolling_median(topo_df["Sigma_reorg"].astype(float), reorg_smooth_n)
    topo_df["Sigma_reorg_smooth_norm"] = normalize_series(topo_df["Sigma_reorg_smooth"].to_numpy())

    standard_norm = normalize_series(standard_df["abs_mean_standard_smooth"].to_numpy())
    topo_event_norm = topo_df["S_event_norm"].to_numpy()

    time_std = standard_df["time_centered_s"].to_numpy(dtype=float)
    time_topo = topo_df["time_centered_s"].to_numpy(dtype=float)

    interp_std_on_topo = np.interp(time_topo, time_std, standard_norm)
    delta_event = interp_std_on_topo - topo_event_norm

    delta_response = topo_df["sigma_alpha_norm"].to_numpy() - topo_df["sigma_alpha_model_norm"].to_numpy()

    event_mad = float(np.nanmean(np.abs(delta_event)))
    event_mse = float(np.nanmean(delta_event ** 2))
    event_corr = float(np.corrcoef(interp_std_on_topo, topo_event_norm)[0, 1]) if len(interp_std_on_topo) > 1 else np.nan

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = integration_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_df = build_metric_comparison(standard_metrics, topo_metrics)
    metric_df.to_csv(output_dir / "gw170817_metric_comparison.csv", index=False, encoding="utf-8-sig")

    diff_df = pd.DataFrame(
        {
            "time_centered_s": time_topo,
            "standard_smoothed_norm_on_topo_grid": interp_std_on_topo,
            "topological_s_event_norm": topo_event_norm,
            "delta_event_selection": delta_event,
            "topological_sigma_alpha_norm": topo_df["sigma_alpha_norm"].to_numpy(),
            "topological_sigma_alpha_model_norm": topo_df["sigma_alpha_model_norm"].to_numpy(),
            "delta_response_model": delta_response,
        }
    )
    diff_df.to_csv(output_dir / "gw170817_difference_curves.csv", index=False, encoding="utf-8-sig")

    summary_rows = [
        {"field": "standard_dir", "value": str(standard_dir)},
        {"field": "topological_dir", "value": str(topo_dir)},
        {"field": "integration_window_start_s", "value": t_min},
        {"field": "integration_window_end_s", "value": t_max},
        {"field": "smooth_window_s", "value": args.smooth_window_s},
        {"field": "reorg_smooth_window_s", "value": args.reorg_smooth_window_s},
        {"field": "event_selection_mad", "value": event_mad},
        {"field": "event_selection_mse", "value": event_mse},
        {"field": "event_selection_corr", "value": event_corr},
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "gw170817_integration_summary.csv", index=False, encoding="utf-8-sig")

    # Graph 1: event selection + difference
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=[2, 1])
    axes[0].plot(time_std, standard_norm, label="Standard smoothed |mean strain|")
    axes[0].plot(time_topo, topo_event_norm, label="Topological S_event")
    axes[0].axvline(focus_t, linestyle="--", label="Topological focus")
    axes[0].set_ylabel("normalized value")
    axes[0].set_title("GW170817 event-selection comparison and difference")
    axes[0].legend()
    axes[1].plot(time_topo, delta_event, label="standard - topological")
    axes[1].axhline(0.0, linestyle="--")
    axes[1].axvline(focus_t, linestyle="--")
    axes[1].set_xlabel("time_centered_s")
    axes[1].set_ylabel("difference")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gw170817_event_selection_difference.png", dpi=150)
    plt.close(fig)

    # Graph 2: reconstruction layer smoothed
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_topo, topo_df["Sigma_reorg_smooth_norm"], label="Topological Sigma_reorg smooth")
    ax.plot(time_topo, topo_df["sigma_alpha_norm"], label="Topological sigma_alpha")
    ax.axvline(focus_t, linestyle="--", label="Topological focus")
    ax.set_xlabel("time_centered_s")
    ax.set_ylabel("normalized value")
    ax.set_title("GW170817 reconstruction / structural-gap (smoothed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gw170817_reconstruction_comparison.png", dpi=150)
    plt.close(fig)

    # Graph 3: response model + residual
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=[2, 1])
    axes[0].plot(time_topo, topo_df["sigma_alpha_norm"], label="sigma_alpha_obs")
    axes[0].plot(time_topo, topo_df["sigma_alpha_model_norm"], label="sigma_alpha_model_postfocus")
    axes[0].axvline(focus_t, linestyle="--", label="Topological focus")
    axes[0].set_ylabel("normalized value")
    axes[0].set_title("GW170817 structural-gap response vs model")
    axes[0].legend()
    axes[1].plot(time_topo, delta_response, label="obs - model")
    axes[1].axhline(0.0, linestyle="--")
    axes[1].axvline(focus_t, linestyle="--")
    axes[1].set_xlabel("time_centered_s")
    axes[1].set_ylabel("difference")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gw170817_response_model_difference.png", dpi=150)
    plt.close(fig)

    # Graph 4: representative times only
    rep_labels = ["standard_peak_time_s", "topo_focus_time_s"]
    rep_values = [
        float(standard_metrics.get("merged_peak_time_centered_s", np.nan)),
        float(topo_metrics.get("t_focus", np.nan)),
    ]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(rep_labels, rep_values)
    ax.set_title("GW170817 representative times")
    fig.tight_layout()
    fig.savefig(output_dir / "gw170817_representative_times.png", dpi=150)
    plt.close(fig)

    # Graph 5: positive time scales only
    scale_labels = ["topo_event_half_window_s", "topo_tau_decay_alpha"]
    scale_values = [
        float(topo_metrics.get("event_half_window_s", np.nan)),
        float(topo_metrics.get("tau_decay_alpha", np.nan)),
    ]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(scale_labels, scale_values)
    ax.set_title("GW170817 topological time scales")
    fig.tight_layout()
    fig.savefig(output_dir / "gw170817_time_scales.png", dpi=150)
    plt.close(fig)

    # Graph 6: error bars
    error_labels = ["topo_rmse", "topo_mae", "event_mad", "event_mse"]
    error_values = [
        float(topo_metrics.get("postfocus_rmse_sigma_alpha", np.nan)),
        float(topo_metrics.get("postfocus_mae_sigma_alpha", np.nan)),
        event_mad,
        event_mse,
    ]
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(error_labels, error_values)
    ax.set_title("GW170817 error / difference metrics")
    fig.tight_layout()
    fig.savefig(output_dir / "gw170817_error_bars.png", dpi=150)
    plt.close(fig)

    report_lines = [
        "GW170817 standard-topological integration report (003)",
        "=====================================================",
        "",
        f"Standard folder: {standard_dir}",
        f"Topological folder: {topo_dir}",
        "",
        "Comparison window",
        "-----------------",
        f"window_start_s: {t_min}",
        f"window_end_s: {t_max}",
        f"smooth_window_s: {args.smooth_window_s}",
        f"reorg_smooth_window_s: {args.reorg_smooth_window_s}",
        "",
        "Key comparison notes",
        "--------------------",
        f"standard_peak_time_centered_s: {standard_metrics.get('merged_peak_time_centered_s', '')}",
        f"standard_peak_gps_time: {standard_metrics.get('standard_peak_gps_time', '')}",
        f"topological_t_focus: {topo_metrics.get('t_focus', '')}",
        f"topological_gps_focus: {topo_metrics.get('gps_focus', '')}",
        f"topological_event_half_window_s: {topo_metrics.get('event_half_window_s', '')}",
        f"topological_tau_decay_alpha: {topo_metrics.get('tau_decay_alpha', '')}",
        f"topological_postfocus_rmse_sigma_alpha: {topo_metrics.get('postfocus_rmse_sigma_alpha', '')}",
        f"topological_postfocus_mae_sigma_alpha: {topo_metrics.get('postfocus_mae_sigma_alpha', '')}",
        f"event_selection_mad: {event_mad}",
        f"event_selection_mse: {event_mse}",
        f"event_selection_corr: {event_corr}",
        "",
        "Generated files",
        "---------------",
        "gw170817_metric_comparison.csv",
        "gw170817_integration_summary.csv",
        "gw170817_difference_curves.csv",
        "gw170817_event_selection_difference.png",
        "gw170817_reconstruction_comparison.png",
        "gw170817_response_model_difference.png",
        "gw170817_representative_times.png",
        "gw170817_time_scales.png",
        "gw170817_error_bars.png",
    ]
    (output_dir / "gw170817_integration_report.txt").write_text("\n".join(map(str, report_lines)), encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"report_path: {output_dir / 'gw170817_integration_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
