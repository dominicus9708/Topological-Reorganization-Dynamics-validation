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
    return parser.parse_args()


def resolve_result_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    # Prefer the base folder itself if result files are placed directly there.
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


def load_windowed_columns(
    csv_path: Path,
    usecols: list[str],
    time_col: str,
    t_min: float,
    t_max: float,
    chunk_size: int,
) -> pd.DataFrame:
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
    add_row("mean_detector_difference_h1_l1", "mean_abs_diff_h1_l1", None, "Standard detector mean absolute difference |H1-L1|.")
    add_row("mean_detector_difference_h1_v1", "mean_abs_diff_h1_v1", None, "Standard detector mean absolute difference |H1-V1|.")
    add_row("mean_detector_difference_l1_v1", "mean_abs_diff_l1_v1", None, "Standard detector mean absolute difference |L1-V1|.")
    add_row("topological_mismatch_mean", None, "D_gw_mean", "Topological full-window detector mismatch mean.")
    add_row("topological_mismatch_max", None, "D_gw_max", "Topological full-window detector mismatch max.")
    add_row("topological_structure_gap_min", None, "sigma_alpha_min", "Minimum structural-dimension-gap response.")
    add_row("topological_structure_gap_max", None, "sigma_alpha_max", "Maximum structural-dimension-gap response.")
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

    if "event_start_time_s" not in topo_metrics or "event_end_time_s" not in topo_metrics:
        raise ValueError("Topological summary missing event_start_time_s or event_end_time_s")
    t_min = float(topo_metrics["event_start_time_s"]) - float(args.window_padding_s)
    t_max = float(topo_metrics["event_end_time_s"]) + float(args.window_padding_s)

    standard_df = load_windowed_columns(
        csv_path=standard_merged,
        usecols=[
            "time_centered_s",
            "strain_mean_h1_l1_v1",
            "strain_abs_mean_h1_l1_v1",
        ],
        time_col="time_centered_s",
        t_min=t_min,
        t_max=t_max,
        chunk_size=args.chunk_size,
    )

    topo_df = load_windowed_columns(
        csv_path=topo_response,
        usecols=[
            "time_centered_s",
            "S_event",
            "Sigma_reorg",
            "sigma_alpha_obs",
            "sigma_alpha_model_postfocus",
            "event_mask",
        ],
        time_col="time_centered_s",
        t_min=t_min,
        t_max=t_max,
        chunk_size=args.chunk_size,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = integration_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_df = build_metric_comparison(standard_metrics, topo_metrics)
    metric_path = output_dir / "gw170817_metric_comparison.csv"
    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")

    summary_rows = [
        {"field": "standard_dir", "value": str(standard_dir)},
        {"field": "topological_dir", "value": str(topo_dir)},
        {"field": "integration_window_start_s", "value": t_min},
        {"field": "integration_window_end_s", "value": t_max},
        {"field": "standard_window_rows", "value": len(standard_df)},
        {"field": "topological_window_rows", "value": len(topo_df)},
    ]
    summary_path = output_dir / "gw170817_integration_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    if not standard_df.empty:
        ax.plot(standard_df["time_centered_s"], normalize_series(standard_df["strain_abs_mean_h1_l1_v1"]), label="Standard |mean(H1,L1,V1)|")
    if not topo_df.empty:
        ax.plot(topo_df["time_centered_s"], normalize_series(topo_df["sigma_alpha_obs"]), label="Topological sigma_alpha")
    ax.axvline(float(topo_metrics["t_focus"]), linestyle="--", label="Topological focus")
    ax.set_xlabel("time_centered_s")
    ax.set_ylabel("normalized value")
    ax.set_title("GW170817 standard vs topological event-window overlay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gw170817_event_window_overlay.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    if not topo_df.empty:
        ax.plot(topo_df["time_centered_s"], normalize_series(topo_df["S_event"]), label="Topological S_event")
        ax.plot(topo_df["time_centered_s"], normalize_series(topo_df["Sigma_reorg"]), label="Topological Sigma_reorg")
        ax.plot(topo_df["time_centered_s"], normalize_series(topo_df["sigma_alpha_model_postfocus"]), label="Topological sigma_alpha model")
    if not standard_df.empty:
        ax.plot(standard_df["time_centered_s"], normalize_series(np.abs(standard_df["strain_mean_h1_l1_v1"])), label="Standard |mean strain|")
    ax.axvline(float(topo_metrics["t_focus"]), linestyle="--", label="Topological focus")
    ax.set_xlabel("time_centered_s")
    ax.set_ylabel("normalized value")
    ax.set_title("GW170817 focus / proxy comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gw170817_focus_proxy_overlay.png", dpi=150)
    plt.close(fig)

    bar_labels = [
        "standard_peak_time_s",
        "topo_focus_time_s",
        "topo_event_half_window_s",
        "topo_tau_decay_alpha",
        "topo_rmse",
        "topo_mae",
    ]
    bar_values = [
        float(standard_metrics.get("merged_peak_time_centered_s", np.nan)),
        float(topo_metrics.get("t_focus", np.nan)),
        float(topo_metrics.get("event_half_window_s", np.nan)),
        float(topo_metrics.get("tau_decay_alpha", np.nan)),
        float(topo_metrics.get("postfocus_rmse_sigma_alpha", np.nan)),
        float(topo_metrics.get("postfocus_mae_sigma_alpha", np.nan)),
    ]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(bar_labels, bar_values)
    ax.set_title("GW170817 summary comparison metrics")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "gw170817_summary_bars.png", dpi=150)
    plt.close(fig)

    report_path = output_dir / "gw170817_integration_report.txt"
    report_lines = [
        "GW170817 standard-topological integration report",
        "===============================================",
        "",
        f"Standard folder: {standard_dir}",
        f"Topological folder: {topo_dir}",
        "",
        "Comparison window",
        "-----------------",
        f"window_start_s: {t_min}",
        f"window_end_s: {t_max}",
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
        "",
        "Generated files",
        "---------------",
        "gw170817_metric_comparison.csv",
        "gw170817_integration_summary.csv",
        "gw170817_event_window_overlay.png",
        "gw170817_focus_proxy_overlay.png",
        "gw170817_summary_bars.png",
    ]
    report_path.write_text("\n".join(map(str, report_lines)), encoding="utf-8")

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"metric_path: {metric_path}")
    print(f"summary_path: {summary_path}")
    print(f"report_path: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
