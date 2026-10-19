"""Matched-baseline pressure-loss study: isothermal water at 20 C.

Run from repository root:
    python studies/pressure_loss_comparison.py --reynolds 20 40 60
Outputs report-ready CSVs and figures. The excess pressure drop is not
automatically an irreversible (total-pressure) loss.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fluid_sim.analysis.engineering import mean_pressure_in_x_band
from fluid_sim.analysis.visualization import flow_rate_metrics
from fluid_sim.config import load_case
from fluid_sim.core.grid import MACGrid
from fluid_sim.numerics.diffusion import diffusion_timestep_limit
from fluid_sim.numerics.solver import divergence_metrics
from fluid_sim.simulation import run_case

# Every sampling band is outside obstacle x in [0.8, 1.0] m.
PAIRS = {"near": (0.65, 1.25), "reference": (0.60, 1.40),
         "far": (0.50, 1.50)}

def csv_write(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def pressures(grid, band):
    return {label: (mean_pressure_in_x_band(grid, up, band),
                    mean_pressure_in_x_band(grid, down, band))
            for label, (up, down) in PAIRS.items()}

def drops(grid, band):
    return {key: values[0] - values[1]
            for key, values in pressures(grid, band).items()}

def pressure_profile(grid):
    x = (np.arange(grid.nx) + 0.5) * grid.dx
    pressure = np.array([
        np.mean(grid.p[~grid.solid[:, i], i])
        if np.any(~grid.solid[:, i]) else np.nan for i in range(grid.nx)
    ])
    return x, pressure

def late_variation(values):
    if len(values) < 5:
        return float("nan")
    last = np.asarray(values[-max(3, math.ceil(len(values) * 0.25)):])
    return float(np.ptp(last) / max(np.mean(np.abs(last)), 1e-30))

def chart(path, x, series, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, y in series.items():
        ax.plot(x, y, "o-", label=name)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def make_case(base, reynolds, obstacle, args, directory):
    velocity = reynolds * base.fluid.dynamic_viscosity / (
        base.fluid.density * base.solver.characteristic_length)
    return replace(
        base,
        name=f"water20c_Re{reynolds:g}_{'obstacle' if obstacle else 'clear'}",
        inlet=replace(base.inlet, profile="uniform", mean_velocity=velocity),
        domain=replace(base.domain, nx=args.nx, ny=args.ny),
        time=replace(base.time, dt=args.dt, steps=round(args.duration / args.dt)),
        validation=replace(base.validation, enabled=False),
        engineering=replace(base.engineering, enabled=True),
        output=replace(base.output, directory=str(directory), save_plots=False,
                       save_animation=False, show_final_plot=False,
                       diagnostics_interval=max(1, round(args.duration / args.dt) // 8)),
        obstacles=list(base.obstacles) if obstacle else [])

def validate_study(base, args):
    base.validate()
    if not np.isclose(base.fluid.density, 998.2) or not np.isclose(
            base.fluid.dynamic_viscosity, 0.001002):
        raise ValueError("This study requires water at 20 C: rho=998.2, mu=0.001002.")
    if not base.obstacles:
        raise ValueError("Use cases/water20c_obstacle.json for the obstacle geometry.")
    if args.dt <= 0 or args.duration <= 0 or args.sample_width <= 0:
        raise ValueError("dt, duration and sample width must be positive.")
    if args.sample_every < 1 or min(args.nx, args.ny) < 8:
        raise ValueError("sample-every must be >=1 and grid dimensions >=8.")
    if args.reynolds != list(dict.fromkeys(args.reynolds)) or any(r <= 0 for r in args.reynolds):
        raise ValueError("Reynolds numbers must be distinct and positive.")
    steps = round(args.duration / args.dt)
    if steps < 5 or not np.isclose(steps * args.dt, args.duration, rtol=1e-12):
        raise ValueError("duration must be an integer multiple of dt (at least 5 steps).")
    grid = MACGrid(nx=args.nx, ny=args.ny, width=base.domain.width, height=base.domain.height)
    limit = diffusion_timestep_limit(grid, base.fluid.kinematic_viscosity)
    if args.dt > limit:
        raise ValueError(f"dt={args.dt} exceeds diffusion limit {limit:g} seconds.")
    for obstacle in base.obstacles:
        for pos, delta in ((obstacle.x_min, grid.dx), (obstacle.x_max, grid.dx),
                           (obstacle.y_min, grid.dy), (obstacle.y_max, grid.dy)):
            if not np.isclose(pos / delta, round(pos / delta), rtol=0, atol=1e-8):
                raise ValueError("Obstacle boundaries must coincide with grid faces.")
        for label, (up, down) in PAIRS.items():
            if not (args.sample_width / 2 < up < down <
                    base.domain.width - args.sample_width / 2):
                raise ValueError(f"Invalid station pair: {label}")
            if up + args.sample_width / 2 >= obstacle.x_min or (
                    down - args.sample_width / 2 <= obstacle.x_max):
                raise ValueError(f"Station pair {label} overlaps obstacle.")
    return steps, limit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="cases/water20c_obstacle.json")
    parser.add_argument("--reynolds", type=float, nargs="+", default=[20, 40, 60])
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--dt", type=float, default=25.0)
    parser.add_argument("--duration", type=float, default=20000.0)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--sample-width", type=float, default=0.1)
    parser.add_argument("--steady-tolerance", type=float, default=0.02)
    parser.add_argument("--output", default="results/studies/water20c_pressure_loss")
    args = parser.parse_args()
    base = load_case(args.case)
    steps, limit = validate_study(base, args)
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    rows, stations, profiles, history = [], [], [], []
    for reynolds in args.reynolds:
        runs = {}
        for label, obstacle in (("clear", False), ("obstacle", True)):
            case = make_case(base, reynolds, obstacle, args,
                             dest / "runs" / f"Re{reynolds:g}_{label}")
            samples = []
            def callback(step, t, grid):
                if step == 1 or step % args.sample_every == 0 or step == steps:
                    samples.append({"step": step, "time_s": t, **drops(grid, args.sample_width)})
            print(f"Running Re={reynolds:g} {label} ({args.nx}x{args.ny})", flush=True)
            start = perf_counter()
            grid = run_case(case, step_callback=callback)
            seconds = perf_counter() - start
            runs[label] = (case, grid, samples, seconds)
            x, p = pressure_profile(grid)
            profiles.extend({"Re": reynolds, "geometry": label, "x_m": float(a),
                             "mean_pressure_pa": float(b)} for a, b in zip(x, p))
            history.extend({"Re": reynolds, "geometry": label, "step": rec["step"],
                            "time_s": rec["time_s"],
                            **{f"delta_p_{key}_pa": rec[key] for key in PAIRS}}
                           for rec in samples)
            csv_write(dest / "pressure_profiles.csv", profiles)
            csv_write(dest / "pressure_history.csv", history)

        clear_case, clear, clear_hist, tc = runs["clear"]
        obs_case, obs, obs_hist, to = runs["obstacle"]
        dp_clear, dp_obs = drops(clear, args.sample_width), drops(obs, args.sample_width)
        pressure_c, pressure_o = pressures(clear, args.sample_width), pressures(obs, args.sample_width)
        q = 0.5 * obs_case.fluid.density * obs_case.inlet.mean_velocity ** 2
        flow_c, flow_o = flow_rate_metrics(clear), flow_rate_metrics(obs)
        max_c, rms_c = divergence_metrics(clear)
        max_o, rms_o = divergence_metrics(obs)
        if [v["step"] for v in clear_hist] != [v["step"] for v in obs_hist]:
            raise RuntimeError("Pressure-history sampling steps differ between runs.")
        excess_hist = np.array([o["reference"] - c["reference"]
                                for o, c in zip(obs_hist, clear_hist)])
        variation = late_variation(excess_hist)
        for pair, (up, down) in PAIRS.items():
            excess = dp_obs[pair] - dp_clear[pair]
            stations.append({
                "Re": reynolds, "pair": pair, "upstream_x_m": up,
                "downstream_x_m": down, "sample_width_m": args.sample_width,
                "clear_upstream_pressure_pa": pressure_c[pair][0],
                "clear_downstream_pressure_pa": pressure_c[pair][1],
                "obstacle_upstream_pressure_pa": pressure_o[pair][0],
                "obstacle_downstream_pressure_pa": pressure_o[pair][1],
                "clear_delta_p_pa": dp_clear[pair],
                "obstacle_delta_p_pa": dp_obs[pair],
                "excess_delta_p_pa": excess,
                "excess_coefficient": excess / q,
                "excess_pumping_power_per_m_depth_w_per_m": excess * flow_o["inlet"]})
        excess = dp_obs["reference"] - dp_clear["reference"]
        rows.append({
            "Re": reynolds, "fluid": "water at 20 C",
            "density_kg_m3": obs_case.fluid.density,
            "dynamic_viscosity_pa_s": obs_case.fluid.dynamic_viscosity,
            "inlet_velocity_m_s": obs_case.inlet.mean_velocity,
            "nx": args.nx, "ny": args.ny, "dt_s": args.dt, "steps": steps,
            "physical_duration_s": args.duration,
            "dynamic_pressure_pa": q,
            "clear_delta_p_pa": dp_clear["reference"],
            "obstacle_delta_p_pa": dp_obs["reference"],
            "excess_delta_p_pa": excess,
            "excess_coefficient": excess / q,
            "clear_coefficient": dp_clear["reference"] / q,
            "obstacle_coefficient": dp_obs["reference"] / q,
            "volume_flow_per_m_depth_m2_per_s": flow_o["inlet"],
            "excess_pumping_power_per_m_depth_w_per_m": excess * flow_o["inlet"],
            "clear_flow_imbalance_m2_per_s": flow_c["difference"],
            "obstacle_flow_imbalance_m2_per_s": flow_o["difference"],
            "clear_rms_divergence_per_s": rms_c, "obstacle_rms_divergence_per_s": rms_o,
            "clear_max_divergence_per_s": max_c, "obstacle_max_divergence_per_s": max_o,
            "clear_runtime_s": tc, "obstacle_runtime_s": to,
            "excess_late_variation_fraction": variation,
            "quasi_steady_indicator": ("PASS" if np.isfinite(variation) and
                variation < args.steady_tolerance else "NOT_ESTABLISHED"),
        })
        csv_write(dest / "summary.csv", rows)
        csv_write(dest / "station_comparisons.csv", stations)
        x, pc = pressure_profile(clear)
        _, po = pressure_profile(obs)
        chart(dest / f"pressure_profile_Re{reynolds:g}.png", x,
              {"Clear": pc * 1e6, "Obstacle": po * 1e6}, "x (m)",
              "Mean static pressure (microPa)", f"Water 20 C: pressure profile, Re={reynolds:g}")
        chart(dest / f"pressure_history_Re{reynolds:g}.png",
              [s["time_s"] for s in clear_hist],
              {"Clear": [s["reference"]*1e6 for s in clear_hist],
               "Obstacle": [s["reference"]*1e6 for s in obs_hist],
               "Excess": excess_hist*1e6},
              "Physical simulation time (s)", "Pressure drop (microPa)",
              f"Pressure drop over time, Re={reynolds:g}")
        print(f"Re {reynolds:g}: clear={dp_clear['reference']*1e6:.3f}, "
              f"obstacle={dp_obs['reference']*1e6:.3f}, "
              f"excess={excess*1e6:.3f} microPa; "
              f"late variation={variation:.3%}")

    chart(dest / "pressure_loss_vs_reynolds.png", [v["Re"] for v in rows],
          {"Clear": [v["clear_delta_p_pa"]*1e6 for v in rows],
           "Obstacle": [v["obstacle_delta_p_pa"]*1e6 for v in rows],
           "Excess": [v["excess_delta_p_pa"]*1e6 for v in rows]},
          "Obstacle-height Reynolds number", "Static pressure drop (microPa)",
          "Matched clear-channel and obstacle pressure comparison")
    chart(dest / "excess_coefficient_vs_reynolds.png", [v["Re"] for v in rows],
          {"Excess coefficient": [v["excess_coefficient"] for v in rows]},
          "Obstacle-height Reynolds number", "Excess pressure coefficient (-)",
          "Incremental pressure-drop coefficient")
    (dest / "study_metadata.json").write_text(json.dumps({
        "fluid": "water at 20 C", "density_kg_m3": base.fluid.density,
        "dynamic_viscosity_pa_s": base.fluid.dynamic_viscosity,
        "inlet_profile": "uniform for both matched configurations",
        "reynolds": args.reynolds, "grid": [args.nx, args.ny],
        "dt_s": args.dt, "duration_s": args.duration,
        "station_pairs_m": PAIRS, "sample_width_m": args.sample_width,
        "explicit_diffusion_dt_limit_s": limit,
        "interpretation": "Static-pressure differences between matched runs; do not "
            "label irreversible total-pressure loss without additional analysis.",
    }, indent=2)+"\n", encoding="utf-8")
    print("Study complete:", dest, flush=True)

if __name__ == "__main__":
    main()
