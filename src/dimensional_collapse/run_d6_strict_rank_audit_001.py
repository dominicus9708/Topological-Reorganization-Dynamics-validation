from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ((4, 5), (4, 6), (5, 6))
BASE_K_RHO = 3.502286898312362
BASE_K_ETA = 2.217


@dataclass(frozen=True)
class Params:
    name: str
    dt: float = 0.002
    t_end: float = 8.0
    k_rho_45: float = BASE_K_RHO
    k_rho_46: float = BASE_K_RHO
    k_rho_56: float = BASE_K_RHO
    gamma_rho: float = 0.01
    k_eta_4: float = BASE_K_ETA
    k_eta_5: float = BASE_K_ETA
    k_eta_6: float = BASE_K_ETA
    q4: float = 1.0
    q5: float = 1.0
    q6: float = 1.0
    rank_tolerance: float = 0.02


def drive(t: float) -> float:
    return (1.0 - math.exp(-t / 0.2)) * math.exp(-t / 12.0)


def closure(t: float) -> float:
    return 1.0 - math.exp(-t / 0.45)


def effective_gram(q: dict[int, float], eta: dict[int, float], rho: dict[tuple[int, int], float]) -> np.ndarray:
    amplitude = {a: q[a] * (1.0 - eta[a]) for a in (4, 5, 6)}
    g = np.zeros((3, 3), dtype=float)
    for i, a in enumerate((4, 5, 6)):
        g[i, i] = amplitude[a]
        for j in range(i + 1, 3):
            b = (4, 5, 6)[j]
            value = math.sqrt(amplitude[a] * amplitude[b]) * rho[(a, b)]
            g[i, j] = g[j, i] = value
    return g


def simulate(p: Params) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rho = {pair: 0.0 for pair in PAIRS}
    eta = {4: 0.0, 5: 0.0, 6: 0.0}
    q = {4: p.q4, 5: p.q5, 6: p.q6}
    k_rho = {(4, 5): p.k_rho_45, (4, 6): p.k_rho_46, (5, 6): p.k_rho_56}
    k_eta = {4: p.k_eta_4, 5: p.k_eta_5, 6: p.k_eta_6}
    rows, events = [], []
    previous_dimension = 6

    for t in np.arange(0.0, p.t_end + p.dt / 2.0, p.dt):
        g = effective_gram(q, eta, rho)
        eig = np.linalg.eigvalsh(g)
        extra_rank = int(np.sum(eig > p.rank_tolerance + 1e-12))
        dimension = 3 + extra_rank
        if dimension != previous_dimension:
            events.append({
                "scenario": p.name,
                "time_tau": float(t),
                "from_dimension": previous_dimension,
                "to_dimension": dimension,
                "event": f"D{previous_dimension}_to_D{dimension}_strict_rank",
                "eig_1": eig[0],
                "eig_2": eig[1],
                "eig_3": eig[2],
            })
            previous_dimension = dimension

        rows.append({
            "scenario": p.name,
            "t_tau": float(t),
            "dimension": dimension,
            "extra_rank": extra_rank,
            "eig_1": eig[0],
            "eig_2": eig[1],
            "eig_3": eig[2],
            "rho_45": rho[(4, 5)],
            "rho_46": rho[(4, 6)],
            "rho_56": rho[(5, 6)],
            "eta_4": eta[4],
            "eta_5": eta[5],
            "eta_6": eta[6],
            "N_prestructure_drive": drive(float(t)),
            "C3": closure(float(t)),
        })

        for pair in PAIRS:
            drho = k_rho[pair] * drive(float(t)) * closure(float(t)) * (1.0 - rho[pair]) - p.gamma_rho * rho[pair]
            rho[pair] = min(max(rho[pair] + p.dt * drho, 0.0), 1.0)
        for a in (4, 5, 6):
            deta = k_eta[a] * drive(float(t)) * closure(float(t)) * (1.0 - eta[a])
            eta[a] = min(max(eta[a] + p.dt * deta, 0.0), 1.0)

    event_df = pd.DataFrame(events)
    summary = {
        "scenario": p.name,
        "final_dimension": int(rows[-1]["dimension"]),
        "event_count": len(events),
        "event_sequence": [e["event"] for e in events],
        "first_event_time_tau": None if not events else events[0]["time_tau"],
    }
    return pd.DataFrame(rows), event_df, summary


def baselines(dt: float = 0.002) -> list[Params]:
    return [
        Params("symmetric_direct_rule_audit", dt=dt),
        Params("pair_dominant_45", dt=dt, k_rho_45=4.5, k_rho_46=2.7, k_rho_56=2.7),
        Params("single_dominant_6", dt=dt, k_rho_45=3.5, k_rho_46=2.4, k_rho_56=2.4, k_eta_6=5.2),
        Params("no_overlap_control", dt=dt, k_rho_45=0.0, k_rho_46=0.0, k_rho_56=0.0, k_eta_4=0.0, k_eta_5=0.0, k_eta_6=0.0),
    ]


def grid() -> pd.DataFrame:
    records = []
    for overlap, asymmetry, close in product((0.0, 0.5, 0.8, 1.0, 1.2), (0.0, 0.2, 0.4, 0.6, 0.8), (0.0, 0.4, 0.8, 1.0, 1.2)):
        side = max(0.05, 1.0 - asymmetry / 2.0)
        p = Params(
            f"grid_o{overlap:.1f}_a{asymmetry:.1f}_c{close:.1f}",
            k_rho_45=BASE_K_RHO * overlap * (1.0 + asymmetry),
            k_rho_46=BASE_K_RHO * overlap * side,
            k_rho_56=BASE_K_RHO * overlap * side,
            k_eta_4=BASE_K_ETA * close,
            k_eta_5=BASE_K_ETA * close,
            k_eta_6=BASE_K_ETA * close,
        )
        _, _, s = simulate(p)
        records.append({
            "overlap_scale": overlap,
            "pair_asymmetry": asymmetry,
            "closure_scale": close,
            "final_dimension": s["final_dimension"],
            "event_sequence": " -> ".join(s["event_sequence"]) or "none",
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summaries, event_frames = [], []
    for p in baselines():
        series, events, summary = simulate(p)
        series.to_csv(args.output / f"d6_strict_timeseries_{p.name}.csv", index=False)
        summaries.append(summary)
        if not events.empty:
            event_frames.append(events)

    pd.DataFrame(summaries).to_csv(args.output / "d6_strict_scenario_summary.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(args.output / "d6_strict_event_log.csv", index=False)
    g = grid()
    g.to_csv(args.output / "d6_strict_parameter_grid_125.csv", index=False)
    g.groupby(["event_sequence", "final_dimension"]).size().reset_index(name="cell_count").sort_values("cell_count", ascending=False).to_csv(args.output / "d6_strict_parameter_grid_summary.csv", index=False)


if __name__ == "__main__":
    main()
