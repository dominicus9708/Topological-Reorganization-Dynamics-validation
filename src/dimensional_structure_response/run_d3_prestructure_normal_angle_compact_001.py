from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = (4, 5, 6)
AXIS = {4: 0, 5: 1, 6: 2}


@dataclass(frozen=True)
class P:
    name: str
    theta4: float = 54.735610317
    theta5: float = 54.735610317
    theta6: float = 54.735610317
    dt: float = 0.004
    t_end: float = 18.0
    e0: float = 12.0
    cap: float = 5.0
    kin: float = 0.62
    kout: float = 0.11
    kout_relax: float = 0.060
    out_scale: float = 1.0
    equalize: float = 1.0
    inflow_on: float = 0.8
    inflow_off: float = 6.0
    outflow_on: float = 4.2
    outflow_off: float = 13.0


def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 700.0), -700.0)))


def win(t: float, on: float, off: float, width: float) -> float:
    return sig((t - on) / width) * (1.0 - sig((t - off) / width))


def angle(p: P, q: int) -> float:
    return {4: p.theta4, 5: p.theta5, 6: p.theta6}[q]


def geom(theta_deg: float) -> tuple[float, float, float]:
    th = math.radians(theta_deg)
    crossing = max(math.cos(th), 0.0)
    coupling = max(math.sin(th), 0.0) ** 2
    return crossing, coupling, 1.0 - coupling


def iso(r: float, n: float = 4.0) -> float:
    x = max(r, 0.0) ** n
    return x / (1.0 + x)


def simulate(p: P) -> dict:
    pre = p.e0
    cross = np.zeros(3)
    relax = np.zeros(3)
    tension = np.ones(3)
    memory = np.zeros(3)
    a = np.ones(3)
    h = np.zeros(3)
    dissipated = 0.0

    gross = passed = internal = withdrawn = 0.0
    peak_h = min_h = 0.0
    peak_tension_dev = peak_shear = 0.0
    first_convergence = math.nan

    nsteps = int(round(p.t_end / p.dt)) + 1
    for k in range(nsteps):
        t = k * p.dt
        ig = win(t, p.inflow_on, p.inflow_off, 0.35)
        og = win(t, p.outflow_on, p.outflow_off, 0.50)
        volume = float(np.prod(a))
        cap_gate = max(1.0 - float(np.sum(cross)) / p.cap, 0.0)

        # Gross crossing is cos(theta); the D3-defined share is sin^2(theta).
        wanted = []
        for q in CHANNELS:
            i = AXIS[q]
            crossing, _, _ = geom(angle(p, q))
            gate = 1.0 - iso(max(h[i], 0.0) * a[i] * (1.75 + 0.35 * i) / 0.86)
            wanted.append(p.kin * ig * pre * cap_gate * crossing * gate / 3.0 * p.dt)
        scale = min(1.0, pre / max(sum(wanted), 1e-30))
        for q, want in zip(CHANNELS, wanted):
            i = AXIS[q]
            _, coupling, pass_fraction = geom(angle(p, q))
            amount = want * scale
            d3 = amount * coupling
            direct = amount * pass_fraction
            pre -= amount
            pre += direct
            cross[i] += d3
            gross += amount
            passed += direct
            internal += d3

        # Coupled high-rank energy transfers to D3 axial relaxation.
        transfer = np.minimum(cross, 0.78 * cross * p.dt)
        cross -= transfer
        relax += transfer

        # Only withdrawal of already-coupled energy drives convergence.
        out = np.zeros(3)
        for q in CHANNELS:
            i = AXIS[q]
            crossing, _, _ = geom(angle(p, q))
            from_cross = min(cross[i], p.kout * p.out_scale * crossing * og * cross[i] * p.dt)
            from_relax = min(relax[i], p.kout_relax * p.out_scale * crossing * og * relax[i] * p.dt)
            cross[i] -= from_cross
            relax[i] -= from_relax
            out[i] = from_cross + from_relax
            pre += out[i]
            withdrawn += out[i]

        # Return/decay bookkeeping.
        ret = np.minimum(cross, 0.014 * cross * p.dt)
        cross -= ret
        pre += 0.92 * float(np.sum(ret))
        dissipated += 0.08 * float(np.sum(ret))
        dec = np.minimum(relax, 0.050 * relax * p.dt)
        relax -= dec
        pre += 0.20 * float(np.sum(dec))
        dissipated += 0.80 * float(np.sum(dec))

        mean_t = float(np.mean(tension))
        td = (-0.25 * transfer / p.dt + 0.33 * out / p.dt
              + 0.52 * p.equalize * (mean_t - tension)
              + 0.075 * (1.0 - tension))
        tension += p.dt * td
        memory += p.dt * (0.84 * p.out_scale * out / p.dt - 0.32 * memory)

        relax_drive = relax / (volume + 0.85) + 0.44 * np.maximum(1.0 - tension, 0.0)
        conv_drive = memory + 0.48 * np.maximum(tension - 1.0, 0.0)
        mean_h = float(np.mean(h))
        dh = (0.96 * relax_drive - 0.90 * conv_drive - 0.43 * h
              + 0.28 * p.equalize * (mean_h - h))
        h += p.dt * dh
        a *= np.exp(h * p.dt)
        a = np.maximum(a, 1e-8)

        mh = float(np.mean(h))
        shear = float(np.std(h))
        tdev = float(np.std(tension))
        peak_h = max(peak_h, mh)
        min_h = min(min_h, mh)
        peak_shear = max(peak_shear, shear)
        peak_tension_dev = max(peak_tension_dev, tdev)
        if math.isnan(first_convergence) and mh <= -0.02:
            first_convergence = t

    mean_a = float(np.prod(a) ** (1.0 / 3.0))
    final_h = float(np.mean(h))
    if mean_a <= 1e-6:
        state = "scale_floor_contracted_d3"
    elif final_h <= -0.02:
        state = "converging_d3"
    elif peak_shear >= 0.02:
        state = "anisotropic_expansion_d3"
    else:
        state = "expanding_d3"

    ledger_error = pre + float(np.sum(cross)) + float(np.sum(relax)) + dissipated - p.e0
    return {
        "scenario": p.name,
        "theta_deg_4": p.theta4,
        "theta_deg_5": p.theta5,
        "theta_deg_6": p.theta6,
        "total_gross_crossing": gross,
        "total_pass_through": passed,
        "total_internal_coupling": internal,
        "pass_through_fraction": passed / max(gross, 1e-30),
        "total_withdrawal_outflow": withdrawn,
        "mean_e_fold": math.log(mean_a),
        "peak_mean_H": peak_h,
        "minimum_mean_H": min_h,
        "final_mean_H": final_h,
        "peak_axial_tension_deviation": peak_tension_dev,
        "peak_directional_H_shear": peak_shear,
        "first_convergence_time_tau": first_convergence,
        "terminal_state": state,
        "energy_ledger_error": ledger_error,
    }


def read_scenarios(path: Path) -> list[P]:
    defaults = P("scenario")
    rows = []
    for raw in pd.read_csv(path).to_dict(orient="records"):
        values = {k: raw[k] for k in defaults.__dataclass_fields__ if k in raw and not pd.isna(raw[k])}
        rows.append(P(**values))
    if not rows:
        raise ValueError("input config contains no scenarios")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--write-grid", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    configured = read_scenarios(args.input_config)
    base = configured[1] if len(configured) > 1 else configured[0]
    pd.DataFrame([simulate(p) for p in configured]).to_csv(
        args.output / "d3_normal_angle_compact_scenario_summary.csv", index=False)

    scan = []
    for deg in range(0, 91, 5):
        row = simulate(replace(base, name=f"angle_{deg}", theta4=deg, theta5=deg, theta6=deg))
        row["analytic_kernel"] = math.cos(math.radians(deg)) * math.sin(math.radians(deg)) ** 2
        scan.append(row)
    pd.DataFrame(scan).to_csv(args.output / "d3_normal_angle_compact_scan.csv", index=False)

    if args.write_grid:
        rows = []
        for deg, out_scale, eq in product(
            (5.0, 25.0, 45.0, 55.0, 75.0),
            (0.5, 1.0, 1.8, 2.8, 4.0),
            (0.25, 0.5, 1.0, 1.75, 3.0),
        ):
            p = replace(base, name=f"g_{deg}_{out_scale}_{eq}", theta4=deg,
                        theta5=deg, theta6=deg, out_scale=out_scale,
                        equalize=eq, dt=0.04, t_end=16.0)
            row = simulate(p)
            row.update(common_angle_deg=deg, outflow_scale=out_scale,
                       equalization_scale=eq)
            rows.append(row)
        grid = pd.DataFrame(rows)
        grid.to_csv(args.output / "d3_normal_angle_compact_grid_125.csv", index=False)
        grid.groupby("terminal_state").size().rename("count").to_csv(
            args.output / "d3_normal_angle_compact_grid_summary.csv")


if __name__ == "__main__":
    main()
