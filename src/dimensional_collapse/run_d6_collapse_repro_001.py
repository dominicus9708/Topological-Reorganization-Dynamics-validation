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
    eps_pair: float = 0.02
    eps_volume: float = 0.035
    eps_single: float = 0.02
    c3_min: float = 0.98
    f_internal: float = 0.55
    f_prestructure: float = 0.30
    f_dissipation: float = 0.15
    q4: float = 1.0
    q5: float = 1.0
    q6: float = 1.0
    rank_tolerance: float = 0.02


def n_drive(t: float) -> float:
    """Coupling drive to an N-dimensional prestructural candidate/exchange regime."""
    return (1.0 - math.exp(-t / 0.2)) * math.exp(-t / 12.0)


def c3(t: float) -> float:
    return 1.0 - math.exp(-t / 0.45)


def pair_sigma(a: int, b: int, q: dict[int, float], rho: dict[tuple[int, int], float]) -> float:
    pair = tuple(sorted((a, b)))
    return math.sqrt(q[a] * q[b]) * (1.0 - rho[pair])


def gram(active: list[int], q: dict[int, float], rho: dict[tuple[int, int], float]) -> np.ndarray:
    g = np.zeros((len(active), len(active)))
    for i, a in enumerate(active):
        g[i, i] = q[a]
        for j in range(i + 1, len(active)):
            b = active[j]
            value = math.sqrt(q[a] * q[b]) * rho[tuple(sorted((a, b)))]
            g[i, j] = g[j, i] = value
    return g


def simulate(p: Params) -> tuple[pd.DataFrame, dict]:
    if not math.isclose(p.f_internal + p.f_prestructure + p.f_dissipation, 1.0, abs_tol=1e-12):
        raise ValueError("collapse allocation fractions must sum to 1")

    rho = {pair: 0.0 for pair in PAIRS}
    eta = {4: 0.0, 5: 0.0, 6: 0.0}
    q = {4: p.q4, 5: p.q5, 6: p.q6}
    k_pair = {(4, 5): p.k_rho_45, (4, 6): p.k_rho_46, (5, 6): p.k_rho_56}
    k_eta = {4: p.k_eta_4, 5: p.k_eta_5, 6: p.k_eta_6}
    active = [4, 5, 6]
    ledger = {"internal": 0.0, "prestructure": 0.0, "dissipation": 0.0}
    events: list[dict] = []

    for t in np.arange(0.0, p.t_end + p.dt / 2.0, p.dt):
        active_before = active.copy()
        sigmas = {pair: pair_sigma(*pair, q, rho) for pair in PAIRS}
        g = gram(active_before, q, rho)
        triple = math.nan
        if set(active_before) == {4, 5, 6}:
            triple = math.sqrt(max(float(np.linalg.det(g)), 0.0))

        event = ""
        removed: list[int] = []
        if set(active_before) == {4, 5, 6} and max(sigmas.values()) <= p.eps_pair and triple <= p.eps_volume and c3(t) >= p.c3_min:
            event, removed = "D6_to_D3_direct_triple", [4, 5, 6]

        if not event and len(active_before) >= 2:
            candidates = [(sigmas[pair], pair) for pair in PAIRS if set(pair).issubset(active_before) and sigmas[pair] <= p.eps_pair and c3(t) >= p.c3_min]
            if candidates:
                _, pair = min(candidates)
                old = 3 + len(active_before)
                event, removed = f"D{old}_to_D{old-2}_pair_{pair[0]}{pair[1]}", list(pair)

        if not event:
            candidates = [(q[a] * (1.0 - eta[a]), a) for a in active_before if q[a] * (1.0 - eta[a]) <= p.eps_single and c3(t) >= p.c3_min]
            if candidates:
                _, axis = min(candidates)
                old = 3 + len(active_before)
                event, removed = f"D{old}_to_D{old-1}_single_{axis}", [axis]

        if event:
            released = sum(q[a] for a in removed)
            ledger["internal"] += p.f_internal * released
            ledger["prestructure"] += p.f_prestructure * released
            ledger["dissipation"] += p.f_dissipation * released
            active = [a for a in active_before if a not in removed]
            events.append({
                "scenario": p.name,
                "time_tau": float(t),
                "event": event,
                "removed_axes": ",".join(map(str, removed)),
                "released_normalized": released,
                "internal_retained_cumulative": ledger["internal"],
                "prestructure_outflow_cumulative": ledger["prestructure"],
                "dissipation_cumulative": ledger["dissipation"],
            })

        for pair in PAIRS:
            drho = k_pair[pair] * n_drive(t) * c3(t) * (1.0 - rho[pair]) - p.gamma_rho * rho[pair]
            rho[pair] = min(max(rho[pair] + p.dt * drho, 0.0), 1.0)
        for axis in (4, 5, 6):
            deta = k_eta[axis] * n_drive(t) * c3(t) * (1.0 - eta[axis])
            eta[axis] = min(max(eta[axis] + p.dt * deta, 0.0), 1.0)

    event_df = pd.DataFrame(events)
    ledger_total = sum(ledger.values()) + sum(q[a] for a in active)
    summary = {
        "scenario": p.name,
        "final_dimension": 3 + len(active),
        "event_count": len(events),
        "event_sequence": [row["event"] for row in events],
        "first_event_time_tau": None if not events else events[0]["time_tau"],
        "direct_D6_to_D3": bool(events and events[0]["event"] == "D6_to_D3_direct_triple"),
        "max_ledger_error": abs(ledger_total - 3.0),
    }
    return event_df, summary


def parameter_grid() -> pd.DataFrame:
    records = []
    for overlap, asymmetry, closure in product((0.0, 0.5, 0.8, 1.0, 1.2), (0.0, 0.2, 0.4, 0.6, 0.8), (0.0, 0.4, 0.8, 1.0, 1.2)):
        side = max(0.05, 1.0 - asymmetry / 2.0)
        p = Params(
            name=f"grid_o{overlap:.1f}_a{asymmetry:.1f}_c{closure:.1f}",
            k_rho_45=BASE_K_RHO * overlap * (1.0 + asymmetry),
            k_rho_46=BASE_K_RHO * overlap * side,
            k_rho_56=BASE_K_RHO * overlap * side,
            k_eta_4=BASE_K_ETA * closure,
            k_eta_5=BASE_K_ETA * closure,
            k_eta_6=BASE_K_ETA * closure,
        )
        _, s = simulate(p)
        records.append({
            "overlap_scale": overlap,
            "pair_asymmetry": asymmetry,
            "closure_scale": closure,
            "final_dimension": s["final_dimension"],
            "event_count": s["event_count"],
            "direct_D6_to_D3": s["direct_D6_to_D3"],
            "first_event_time_tau": s["first_event_time_tau"],
            "event_sequence": " -> ".join(s["event_sequence"]) or "none",
            "max_ledger_error": s["max_ledger_error"],
        })
    return pd.DataFrame(records)


def convergence() -> pd.DataFrame:
    records = []
    for dt in (0.008, 0.004, 0.002, 0.001, 0.0005):
        _, s = simulate(Params(name=f"dt_{dt}", dt=dt, t_end=4.0))
        records.append({"dt": dt, "direct_D6_to_D3": s["direct_D6_to_D3"], "event_time_tau": s["first_event_time_tau"], "max_ledger_error": s["max_ledger_error"]})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = pd.read_csv(args.input_config)
    summaries, event_tables = [], []
    for row in config.to_dict(orient="records"):
        events, summary = simulate(Params(**row))
        summaries.append(summary)
        if not events.empty:
            event_tables.append(events)

    pd.DataFrame(summaries).to_csv(args.output / "d6_scenario_summary.csv", index=False)
    (pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()).to_csv(args.output / "d6_event_log.csv", index=False)
    parameter_grid().to_csv(args.output / "d6_parameter_grid_125.csv", index=False)
    convergence().to_csv(args.output / "d6_convergence.csv", index=False)


if __name__ == "__main__":
    main()
