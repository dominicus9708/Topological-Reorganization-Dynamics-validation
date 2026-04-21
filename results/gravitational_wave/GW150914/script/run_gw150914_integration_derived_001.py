#!/usr/bin/env python3
"""
run_gw150914_integration_derived_003_001.py

Purpose
-------
Integration pipeline for the GW150914 validation case using the fully derived
topological response (derived_003).

Placement
---------
results/gravitational_wave/GW150914/script/run_gw150914_integration_derived_003_001.py

Input
-----
Searches recursively under:
- results/gravitational_wave/GW150914/output/standard/
- results/gravitational_wave/GW150914/output/topological/

and selects the most recent run folders that contain the expected files.

Output
------
Writes timestamped integration results to:
  results/gravitational_wave/GW150914/output/Integration/YYYYMMDD_HHMMSS/

Generated files
---------------
- gw150914_integration_derived003_waveform_vs_response.png
- gw150914_integration_derived003_postpeak_relaxation.png
- gw150914_integration_derived003_propagation_mismatch.png
- gw150914_integration_derived003_summary.csv
- gw150914_integration_derived003_report.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    return parser.parse_args()


def find_most_recent_run_with_file(base_dir: Path, required_filename: str) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    candidate_dirs = [p.parent for p in base_dir.rglob(required_filename)]
    if not candidate_dirs:
        raise FileNotFoundError(
            f"Could not find any run folder under {base_dir} containing {required_filename}"
        )
    candidate_dirs = sorted(candidate_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidate_dirs[0]


def load_standard_run(standard_base: Path):
    run_dir = find_most_recent_run_with_file(standard_base, "gw150914_standard_merged_full.csv")
    h1_path = run_dir / "gw150914_standard_h1_full.csv"
    l1_path = run_dir / "gw150914_standard_l1_full.csv"
    merged_path = run_dir / "gw150914_standard_merged_full.csv"
    summary_path = run_dir / "gw150914_standard_baseline_summary.csv"
    report_path = run_dir / "gw150914_standard_report.txt"
    for p in [h1_path, l1_path, merged_path, summary_path, report_path]:
        if not p.exists():
            raise FileNotFoundError(p)
    return run_dir, pd.read_csv(h1_path), pd.read_csv(l1_path), pd.read_csv(merged_path), pd.read_csv(summary_path), report_path


def load_topological_run(topological_base: Path):
    run_dir = find_most_recent_run_with_file(topological_base, "gw150914_topological_response_derived_003.csv")
    response_path = run_dir / "gw150914_topological_response_derived_003.csv"
    summary_path = run_dir / "gw150914_topological_summary_derived_003.csv"
    report_path = run_dir / "gw150914_topological_report_derived_003.txt"
    for p in [response_path, summary_path, report_path]:
        if not p.exists():
            raise FileNotFoundError(p)
    return run_dir, pd.read_csv(response_path), pd.read_csv(summary_path), report_path


def get_summary_value(summary_df: pd.DataFrame, metric: str):
    if "metric" not in summary_df.columns or "value" not in summary_df.columns:
        return None
    mask = summary_df["metric"].astype(str) == metric
    if not mask.any():
        return None
    return summary_df.loc[mask, "value"].iloc[0]


def make_waveform_vs_response_plot(
    h1_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    standard_merged_df: pd.DataFrame,
    topo_df: pd.DataFrame,
    standard_summary_df: pd.DataFrame,
    topological_summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(11, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(h1_df["time_centered_s"], h1_df["strain"], label="strain_h1")
    ax1.plot(l1_df["time_centered_s"], l1_df["strain"], label="strain_l1")
    ax1.plot(standard_merged_df["time_centered_s"], standard_merged_df["strain_mean_h1_l1"], label="strain_mean_h1_l1")
    peak_t = get_summary_value(standard_summary_df, "merged_peak_time_centered_s")
    if peak_t is not None:
        ax1.axvline(float(peak_t), linestyle="--", label="merged_peak_time")
    ax1.set_xlabel("time_centered_s")
    ax1.set_ylabel("strain")
    ax1.set_title("GW150914 Standard Waveform Baseline")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(topo_df["time_centered_s"], topo_df["sigma_obs_local"], label="sigma_obs_local")
    ax2.plot(topo_df["time_centered_s"], topo_df["sigma_env_obs"], label="sigma_env_obs")
    ax2.plot(topo_df["time_centered_s"], topo_df["sigma_env_model_postpeak"], label="sigma_env_model_postpeak")
    ax2.set_xlabel("time_centered_s")
    ax2.set_ylabel("sigma")
    ax2.set_title("GW150914 Derived Local/Envelope Structural Response")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_postpeak_relaxation_plot(topo_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = topo_df[topo_df["time_from_peak_env_s"].notna()].copy()
    plot_df = plot_df[plot_df["time_from_peak_env_s"] >= 0].copy()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(plot_df["time_from_peak_env_s"], plot_df["sigma_env_obs"], label="sigma_env_obs")
    ax.plot(plot_df["time_from_peak_env_s"], plot_df["sigma_env_model_postpeak"], label="sigma_env_model_postpeak")
    ax.set_xlabel("time_from_peak_env_s")
    ax.set_ylabel("sigma_env")
    ax.set_title("GW150914 Post-Peak Envelope Relaxation Comparison")
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(plot_df["time_from_peak_env_s"], plot_df["delta_sigma_env_postpeak"], label="delta_sigma_env_postpeak")
    ax2.set_ylabel("delta_sigma_env_postpeak")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_propagation_mismatch_plot(topo_df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(topo_df["time_centered_s"], topo_df["propagation_proxy_env"], label="propagation_proxy_env")
    ax.set_xlabel("time_centered_s")
    ax.set_ylabel("propagation_proxy_env")
    ax.set_title("GW150914 Propagation Proxy and Detector Mismatch Stability")
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(topo_df["time_centered_s"], topo_df["detector_mismatch_env"], label="detector_mismatch_env")
    ax2.set_ylabel("detector_mismatch_env")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_integration_summary(
    standard_run_dir: Path,
    topological_run_dir: Path,
    standard_summary_df: pd.DataFrame,
    topological_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "standard_run_dir", "value": str(standard_run_dir), "note": "Standard run folder selected automatically."},
        {"metric": "topological_run_dir", "value": str(topological_run_dir), "note": "Topological run folder selected automatically."},
    ]

    std_metrics = [
        "h1_sample_rate_hz",
        "l1_sample_rate_hz",
        "h1_peak_time_centered_s",
        "l1_peak_time_centered_s",
        "merged_peak_time_centered_s",
        "h1_peak_abs_strain",
        "l1_peak_abs_strain",
        "merged_peak_abs_mean_strain",
        "h1_l1_peak_time_offset_s",
        "mean_abs_detector_difference",
        "max_abs_detector_difference",
    ]
    topo_metrics = [
        "t_peak_raw",
        "A_peak_raw",
        "tau_decay_raw",
        "baseline_window_s",
        "envelope_window_s",
        "postpeak_window_s",
        "t_peak_env",
        "E_peak",
        "tau_decay_env",
        "kappa_derived",
        "sigma_env_peak",
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
        "decay_status_env",
    ]

    for metric in std_metrics:
        val = get_summary_value(standard_summary_df, metric)
        if val is not None:
            rows.append({"metric": f"standard_{metric}", "value": val, "note": "Copied from standard baseline summary."})
    for metric in topo_metrics:
        val = get_summary_value(topological_summary_df, metric)
        if val is not None:
            rows.append({"metric": f"topological_{metric}", "value": val, "note": "Copied from topological summary."})

    return pd.DataFrame(rows)


def build_report(
    standard_run_dir: Path,
    topological_run_dir: Path,
    integration_summary_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    metric_map = {str(r["metric"]): r["value"] for _, r in integration_summary_df.iterrows()}

    lines = []
    lines.append("GW150914 integration report (derived_003 version)")
    lines.append("=================================================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This integration stage compares the standard detector-baseline waveform and the fully derived "
        "topological local/envelope structure-response output without forcing them into the same observational status."
    )
    lines.append("")
    lines.append("Selected run folders")
    lines.append("--------------------")
    lines.append(f"standard_run_dir: {standard_run_dir}")
    lines.append(f"topological_run_dir: {topological_run_dir}")
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append(
        "The waveform-vs-response plot shows the observed strain baseline in the upper panel and the "
        "derived local/envelope structural response in the lower panel."
    )
    lines.append(
        "The post-peak relaxation plot isolates the envelope-based relaxation model after the derived envelope peak."
    )
    lines.append(
        "The propagation-mismatch plot shows the envelope-based structural propagation proxy together with "
        "the envelope-normalized detector mismatch."
    )
    lines.append("")
    lines.append("Key copied metrics")
    lines.append("------------------")
    for k in [
        "standard_merged_peak_time_centered_s",
        "standard_merged_peak_abs_mean_strain",
        "standard_h1_l1_peak_time_offset_s",
        "topological_t_peak_env",
        "topological_E_peak",
        "topological_tau_decay_env",
        "topological_kappa_derived",
        "topological_baseline_window_s",
        "topological_envelope_window_s",
        "topological_postpeak_window_s",
        "topological_propagation_proxy_env_max",
        "topological_detector_mismatch_env_max",
        "topological_postpeak_rmse_sigma_env",
        "topological_postpeak_mae_sigma_env",
        "topological_decay_status_env",
    ]:
        if k in metric_map:
            lines.append(f"{k}: {metric_map[k]}")
    lines.append("")
    lines.append("Output directory")
    lines.append("----------------")
    lines.append(str(output_dir))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    standard_base = project_root / "results" / "gravitational_wave" / "GW150914" / "output" / "standard"
    topological_base = project_root / "results" / "gravitational_wave" / "GW150914" / "output" / "topological"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "gravitational_wave" / "GW150914" / "output" / "Integration" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_run_dir, h1_df, l1_df, standard_merged_df, standard_summary_df, _ = load_standard_run(standard_base)
    topological_run_dir, topo_df, topological_summary_df, _ = load_topological_run(topological_base)

    waveform_plot = output_dir / "gw150914_integration_derived003_waveform_vs_response.png"
    postpeak_plot = output_dir / "gw150914_integration_derived003_postpeak_relaxation.png"
    propagation_plot = output_dir / "gw150914_integration_derived003_propagation_mismatch.png"
    summary_csv = output_dir / "gw150914_integration_derived003_summary.csv"
    report_txt = output_dir / "gw150914_integration_derived003_report.txt"

    make_waveform_vs_response_plot(
        h1_df=h1_df,
        l1_df=l1_df,
        standard_merged_df=standard_merged_df,
        topo_df=topo_df,
        standard_summary_df=standard_summary_df,
        topological_summary_df=topological_summary_df,
        output_path=waveform_plot,
    )
    make_postpeak_relaxation_plot(topo_df=topo_df, output_path=postpeak_plot)
    make_propagation_mismatch_plot(topo_df=topo_df, output_path=propagation_plot)

    integration_summary_df = build_integration_summary(
        standard_run_dir=standard_run_dir,
        topological_run_dir=topological_run_dir,
        standard_summary_df=standard_summary_df,
        topological_summary_df=topological_summary_df,
    )
    integration_summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    report_txt.write_text(
        build_report(
            standard_run_dir=standard_run_dir,
            topological_run_dir=topological_run_dir,
            integration_summary_df=integration_summary_df,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"waveform_plot: {waveform_plot}")
    print(f"postpeak_plot: {postpeak_plot}")
    print(f"propagation_plot: {propagation_plot}")
    print(f"summary_csv: {summary_csv}")
    print(f"report_txt: {report_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
