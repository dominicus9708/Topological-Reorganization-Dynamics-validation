#!/usr/bin/env python3
"""
run_mercury_integration_001.py

Purpose
-------
Integration pipeline for the Mercury validation case.

Placement
---------
results/mercury/script/run_mercury_integration_001.py

Input
-----
Searches recursively under:
- results/mercury/output/standard/
- results/mercury/output/topological/

and selects the most recent run folders that contain the expected files.

Output
------
Writes timestamped integration results to:
  results/mercury/output/Integration/YYYYMMDD_HHMMSS/

Generated files
---------------
- mercury_integration_standard_topological_comparison.png
- mercury_integration_radius_response_phase.png
- mercury_integration_summary.csv
- mercury_integration_report.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True, help="Path to project root.")
    return parser.parse_args()


def find_most_recent_run_with_file(base_dir: Path, required_filename: str) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    candidate_dirs = []
    for path in base_dir.rglob(required_filename):
        candidate_dirs.append(path.parent)

    if not candidate_dirs:
        raise FileNotFoundError(
            f"Could not find any run folder under {base_dir} containing {required_filename}"
        )

    candidate_dirs = sorted(candidate_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidate_dirs[0]


def load_standard_run(standard_base: Path):
    run_dir = find_most_recent_run_with_file(standard_base, "mercury_standard_vectors_full.csv")
    vectors_path = run_dir / "mercury_standard_vectors_full.csv"
    summary_path = run_dir / "mercury_standard_baseline_summary.csv"
    report_path = run_dir / "mercury_standard_report.txt"

    if not vectors_path.exists():
        raise FileNotFoundError(vectors_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    vectors_df = pd.read_csv(vectors_path)
    summary_df = pd.read_csv(summary_path)
    return run_dir, vectors_df, summary_df, report_path


def load_topological_run(topological_base: Path):
    primary_vectors = "mercury_topological_vectors_response_derived_002.csv"
    fallback_vectors = "mercury_topological_vectors_response.csv"

    try:
        run_dir = find_most_recent_run_with_file(topological_base, primary_vectors)
        vectors_path = run_dir / "mercury_topological_vectors_response_derived_002.csv"
        summary_path = run_dir / "mercury_topological_summary_derived_002.csv"
        report_path = run_dir / "mercury_topological_report_derived_002.txt"
        mode = "derived_002"
    except FileNotFoundError:
        run_dir = find_most_recent_run_with_file(topological_base, fallback_vectors)
        vectors_path = run_dir / "mercury_topological_vectors_response.csv"
        summary_path = run_dir / "mercury_topological_summary.csv"
        report_path = run_dir / "mercury_topological_report.txt"
        mode = "trial_001"

    if not vectors_path.exists():
        raise FileNotFoundError(vectors_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    vectors_df = pd.read_csv(vectors_path)
    summary_df = pd.read_csv(summary_path)
    return run_dir, vectors_df, summary_df, report_path, mode


def get_summary_value(summary_df: pd.DataFrame, metric: str) -> Optional[float]:
    if "metric" not in summary_df.columns or "value" not in summary_df.columns:
        return None
    mask = summary_df["metric"].astype(str) == metric
    if not mask.any():
        return None
    value = summary_df.loc[mask, "value"].iloc[0]
    try:
        return float(value)
    except Exception:
        return None


def choose_topological_columns(topo_df: pd.DataFrame):
    sigma_col = "SIGMA_DERIVED" if "SIGMA_DERIVED" in topo_df.columns else "SIGMA_TRIAL"
    ratio_col = "TOPO_TO_NEWTON_RATIO_DERIVED" if "TOPO_TO_NEWTON_RATIO_DERIVED" in topo_df.columns else "TOPO_TO_NEWTON_RATIO"
    gtopo_col = "G_TOPOLOGICAL_DERIVED_AU_PER_DAY2" if "G_TOPOLOGICAL_DERIVED_AU_PER_DAY2" in topo_df.columns else "G_TOPOLOGICAL_TRIAL_AU_PER_DAY2"
    return sigma_col, ratio_col, gtopo_col


def make_time_comparison_plot(standard_vectors_df, topological_vectors_df, standard_summary_df, output_path: Path) -> None:
    sigma_col, ratio_col, _ = choose_topological_columns(topological_vectors_df)

    std_df = standard_vectors_df.copy().sort_values("JDTDB").reset_index(drop=True)
    topo_df = topological_vectors_df.copy().sort_values("JDTDB").reset_index(drop=True)

    mean_q = get_summary_value(standard_summary_df, "mean_periapsis_distance_qr_au")
    mean_ad = get_summary_value(standard_summary_df, "mean_apoapsis_distance_ad_au")
    mean_a = get_summary_value(standard_summary_df, "mean_semi_major_axis_au")

    fig = plt.figure(figsize=(11, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(std_df["JDTDB"], std_df["R_AU"])
    if mean_q is not None:
        ax1.axhline(mean_q, linestyle="--")
    if mean_ad is not None:
        ax1.axhline(mean_ad, linestyle="--")
    if mean_a is not None:
        ax1.axhline(mean_a, linestyle=":")
    ax1.set_xlabel("JDTDB")
    ax1.set_ylabel("R_AU")
    ax1.set_title("Mercury Standard Orbital Baseline")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(topo_df["JDTDB"], topo_df[sigma_col].astype(float))
    ax2.set_xlabel("JDTDB")
    ax2.set_ylabel(sigma_col)
    ax2.set_title("Mercury Topological Structural Response")

    ax2b = ax2.twinx()
    ax2b.plot(topo_df["JDTDB"], topo_df[ratio_col].astype(float))
    ax2b.set_ylabel(ratio_col)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_radius_response_phase_plot(topological_vectors_df, output_path: Path) -> None:
    sigma_col, ratio_col, _ = choose_topological_columns(topological_vectors_df)
    topo_df = topological_vectors_df.copy().sort_values("R_AU").reset_index(drop=True)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(
        topo_df["R_AU"].astype(float),
        topo_df[sigma_col].astype(float),
        c=topo_df[ratio_col].astype(float),
    )
    ax.set_xlabel("R_AU")
    ax.set_ylabel(sigma_col)
    ax.set_title("Mercury Radius–Response Phase Comparison")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(ratio_col)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_integration_summary(standard_run_dir, topological_run_dir, standard_summary_df, topological_summary_df, topo_mode: str) -> pd.DataFrame:
    rows = [
        {"metric": "standard_run_dir", "value": str(standard_run_dir), "note": "Standard run folder selected automatically."},
        {"metric": "topological_run_dir", "value": str(topological_run_dir), "note": "Topological run folder selected automatically."},
        {"metric": "topological_mode", "value": topo_mode, "note": "Topological output mode used for integration."},
    ]

    for metric in [
        "mean_eccentricity",
        "mean_semi_major_axis_au",
        "mean_sidereal_period_day",
        "mean_periapsis_distance_qr_au",
        "mean_apoapsis_distance_ad_au",
        "min_radius_au",
        "max_radius_au",
    ]:
        val = get_summary_value(standard_summary_df, metric)
        if val is not None:
            rows.append({"metric": f"standard_{metric}", "value": val, "note": "Copied from standard baseline summary."})

    for metric in [
        "sigma_derived_min",
        "sigma_derived_max",
        "topo_to_newton_ratio_max",
        "topo_to_newton_ratio_mean",
        "g_topological_abs_max",
        "sigma_trial_min",
        "sigma_trial_max",
    ]:
        val = get_summary_value(topological_summary_df, metric)
        if val is not None:
            rows.append({"metric": f"topological_{metric}", "value": val, "note": "Copied from topological summary."})

    return pd.DataFrame(rows)


def build_report(standard_run_dir, topological_run_dir, topo_mode: str, integration_summary_df: pd.DataFrame, output_dir: Path) -> str:
    metric_map = {str(r["metric"]): r["value"] for _, r in integration_summary_df.iterrows()}

    lines = []
    lines.append("Mercury integration report")
    lines.append("=========================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This integration stage compares the standard orbital baseline and the topological "
        "structure-response output without forcing them into the same observational status."
    )
    lines.append("")
    lines.append("Selected run folders")
    lines.append("--------------------")
    lines.append(f"standard_run_dir: {standard_run_dir}")
    lines.append(f"topological_run_dir: {topological_run_dir}")
    lines.append(f"topological_mode: {topo_mode}")
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append(
        "The upper-panel comparison plot shows the ordinary Mercury orbital radius baseline over time. "
        "The lower-panel comparison plot shows the topological structural response on the same time axis."
    )
    lines.append(
        "The phase plot shows how the structural response varies with orbital radius, with the color scale "
        "indicating the relative size of the topological term compared with the Newtonian baseline."
    )
    lines.append("")
    lines.append("Key copied metrics")
    lines.append("------------------")
    for k in [
        "standard_mean_eccentricity",
        "standard_mean_semi_major_axis_au",
        "standard_mean_sidereal_period_day",
        "standard_min_radius_au",
        "standard_max_radius_au",
        "topological_topo_to_newton_ratio_max",
        "topological_topo_to_newton_ratio_mean",
        "topological_g_topological_abs_max",
        "topological_sigma_derived_min",
        "topological_sigma_derived_max",
        "topological_sigma_trial_min",
        "topological_sigma_trial_max",
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

    standard_base = project_root / "results" / "mercury" / "output" / "standard"
    topological_base = project_root / "results" / "mercury" / "output" / "topological"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "mercury" / "output" / "Integration" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_run_dir, standard_vectors_df, standard_summary_df, _ = load_standard_run(standard_base)
    topological_run_dir, topological_vectors_df, topological_summary_df, _, topo_mode = load_topological_run(topological_base)

    time_plot = output_dir / "mercury_integration_standard_topological_comparison.png"
    phase_plot = output_dir / "mercury_integration_radius_response_phase.png"
    summary_csv = output_dir / "mercury_integration_summary.csv"
    report_txt = output_dir / "mercury_integration_report.txt"

    make_time_comparison_plot(
        standard_vectors_df=standard_vectors_df,
        topological_vectors_df=topological_vectors_df,
        standard_summary_df=standard_summary_df,
        output_path=time_plot,
    )
    make_radius_response_phase_plot(
        topological_vectors_df=topological_vectors_df,
        output_path=phase_plot,
    )

    integration_summary_df = build_integration_summary(
        standard_run_dir=standard_run_dir,
        topological_run_dir=topological_run_dir,
        standard_summary_df=standard_summary_df,
        topological_summary_df=topological_summary_df,
        topo_mode=topo_mode,
    )
    integration_summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    report_txt.write_text(
        build_report(
            standard_run_dir=standard_run_dir,
            topological_run_dir=topological_run_dir,
            topo_mode=topo_mode,
            integration_summary_df=integration_summary_df,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )

    print("Done.")
    print(f"output_dir: {output_dir}")
    print(f"time_plot: {time_plot}")
    print(f"phase_plot: {phase_plot}")
    print(f"summary_csv: {summary_csv}")
    print(f"report_txt: {report_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
