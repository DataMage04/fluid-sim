"""Grid refinement and performance measurements for water at 20 C.

Run from repository root:
 python studies/grid_performance.py --case cases/water20c_poiseuille.json
 python studies/grid_performance.py --case cases/water20c_obstacle.json
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import replace
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fluid_sim.config import load_case
from fluid_sim.simulation import run_case
from fluid_sim.analysis.engineering import compute_engineering_results, analytical_poiseuille_pressure_gradient
from fluid_sim.analysis.validation import analytical_poiseuille_profile, numerical_velocity_profile, profile_error
from fluid_sim.analysis.visualization import flow_rate_metrics
from fluid_sim.numerics.diffusion import diffusion_timestep_limit
from fluid_sim.numerics.solver import divergence_metrics
from fluid_sim.core.grid import MACGrid


def mesh_size(value):
    try:
        nx, ny = map(int, value.lower().split("x"))
        if min(nx, ny) < 4:
            raise ValueError()
        return nx, ny
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Mesh must be Nx x Ny, e.g. 80x40") from exc


def csv_out(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(case, nx, ny, dt, duration, repeat, folder):
    steps = round(duration / dt)
    if abs(steps * dt - duration) > 1e-8 * max(1, duration):
        raise ValueError("duration must be an integer multiple of dt")
    output = replace(case.output, directory=str(folder / f"{nx}x{ny}_repeat{repeat}"),
                     save_plots=False, save_animation=False, show_final_plot=False)
    trial = replace(case, domain=replace(case.domain, nx=nx, ny=ny),
                    time=replace(case.time, dt=dt, steps=steps), output=output)
    trial.validate()
    nu = trial.fluid.kinematic_viscosity
    limit = diffusion_timestep_limit(MACGrid(nx, ny, trial.domain.width, trial.domain.height), nu)
    if dt > limit:
        raise ValueError(f"Unstable diffusion dt={dt:g}; limit={limit:g} for {nx}x{ny}")
    # The obstacle edges must be aligned with mesh faces to keep geometry unchanged.
    for ob in trial.obstacles:
        for position, spacing in ((ob.x_min, trial.domain.width / nx),
                                  (ob.x_max, trial.domain.width / nx),
                                  (ob.y_min, trial.domain.height / ny),
                                  (ob.y_max, trial.domain.height / ny)):
            if abs(position / spacing - round(position / spacing)) > 1e-8:
                raise ValueError(f"Obstacle edge {position:g} not aligned on {nx}x{ny}")
    start = perf_counter()
    grid = run_case(trial)
    elapsed = perf_counter() - start
    flow = flow_rate_metrics(grid)
    max_div, rms_div = divergence_metrics(grid)
    u = (grid.u[:, :-1] + grid.u[:, 1:]) / 2
    v = (grid.v[:-1, :] + grid.v[1:, :]) / 2
    speed = np.hypot(u, v)[~grid.solid]
    gradient = None
    if trial.inlet.profile == "poiseuille" and not trial.obstacles:
        gradient = analytical_poiseuille_pressure_gradient(
            trial.fluid.dynamic_viscosity, trial.inlet.mean_velocity, trial.domain.height)
    eng = compute_engineering_results(
        grid, density=trial.fluid.density, reference_velocity=trial.inlet.mean_velocity,
        upstream_x=trial.engineering.upstream_x, downstream_x=trial.engineering.downstream_x,
        sample_width=trial.engineering.sample_width,
        reference_pressure=trial.engineering.reference_pressure,
        analytical_pressure_gradient=gradient)
    rmse = float("nan")
    if gradient is not None:
        y, numeric = numerical_velocity_profile(grid, trial.validation.x_location)
        analytic = analytical_poiseuille_profile(y, grid.height, trial.inlet.mean_velocity)
        rmse = profile_error(numeric, analytic)["relative_rmse"]
    return dict(mesh=f"{nx}x{ny}", nx=nx, ny=ny, cells=nx*ny, dx=grid.dx, dy=grid.dy,
                dt_s=dt, steps=steps, physical_time_s=steps*dt, repeat=repeat,
                runtime_s=elapsed, seconds_per_step=elapsed/steps,
                pressure_drop_pa=eng.pressure_drop, max_speed_m_s=float(np.max(speed)),
                velocity_relative_rmse=rmse,
                analytical_pressure_drop_pa=eng.analytical_pressure_drop,
                pressure_relative_error=eng.pressure_drop_relative_error,
                inlet_flow_m2_s=flow["inlet"], outlet_flow_m2_s=flow["outlet"],
                flow_imbalance_m2_s=flow["difference"], max_divergence_1_s=max_div,
                rms_divergence_1_s=rms_div, diffusion_dt_limit_s=limit)


def make_plots(summary, output):
    for key, label, name in [
        ("runtime_s", "Median runtime (s)", "runtime_vs_cells.png"),
        ("pressure_drop_pa", "Pressure drop (Pa)", "pressure_drop_vs_cells.png"),
        ("max_speed_m_s", "Maximum velocity (m/s)", "velocity_vs_cells.png"),
        ("velocity_relative_rmse", "Relative velocity RMSE", "poiseuille_error_vs_cells.png")]:
        values = [(r["cells"], r[key]) for r in summary
                  if r[key] is not None and np.isfinite(float(r[key]))]
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([v[0] for v in values], [v[1] for v in values], "o-")
        ax.set(xlabel="Grid cells", ylabel=label)
        ax.grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(output / name, dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--meshes", nargs="+", type=mesh_size, default=[(40,20),(80,40),(160,80)])
    parser.add_argument("--dt", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", default="results/studies/water20c")
    args = parser.parse_args()
    case = load_case(args.case)
    dt = args.dt if args.dt is not None else case.time.dt
    duration = args.duration if args.duration is not None else case.time.dt * case.time.steps
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw = []
    for nx, ny in args.meshes:
        for repeat in range(1, args.repeats+1):
            print(f"Running {case.name}: {nx}x{ny}, repetition {repeat}/{args.repeats}", flush=True)
            row = measure(case, nx, ny, dt, duration, repeat, output / "runs")
            raw.append(row)
            csv_out(output / "raw_runs.csv", raw)
    summary = []
    for nx, ny in args.meshes:
        group = [r for r in raw if r["nx"] == nx and r["ny"] == ny]
        row = dict(group[0])
        row.pop("repeat")
        for key in ("runtime_s", "seconds_per_step", "pressure_drop_pa",
                    "max_speed_m_s", "velocity_relative_rmse", "flow_imbalance_m2_s",
                    "rms_divergence_1_s", "pressure_relative_error"):
            vals = [float(r[key]) for r in group if r[key] is not None and np.isfinite(float(r[key]))]
            row[key] = median(vals) if vals else None
        row["repeats"] = len(group)
        summary.append(row)
    csv_out(output / "summary.csv", summary)
    make_plots(summary, output)
    (output / "metadata.json").write_text(json.dumps(dict(
        case=args.case, fluid="water at 20 C", density_kg_m3=case.fluid.density,
        viscosity_pa_s=case.fluid.dynamic_viscosity, dt_s=dt, duration_s=duration,
        python=sys.version, platform=platform.platform(), repeats=args.repeats,
        note="Numerical simulation; runtime includes run_case reporting and output writing."
    ), indent=2), encoding="utf-8")
    print(f"Complete. Send back {output / 'summary.csv'} and the generated plots.")


if __name__ == "__main__":
    main()
