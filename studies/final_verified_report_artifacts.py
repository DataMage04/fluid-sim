from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def read_csv(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def to_float(value, default=float('nan')):
    try:
        if value is None or value == '':
            return default
        return float(value)
    except Exception:
        return default


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_grid_independence(poiseuille_rows, obstacle_rows, outdir: Path):
    # pressure error vs cell count (Poiseuille)
    cells = [to_float(r['cells']) for r in poiseuille_rows]
    p_err = [100.0 * to_float(r.get('pressure_relative_error')) for r in poiseuille_rows]
    v_err = [100.0 * to_float(r.get('velocity_relative_rmse')) for r in poiseuille_rows]
    runtime = [to_float(r.get('runtime_s')) for r in poiseuille_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(cells, p_err, 'o-', label='Pressure-drop relative error')
    ax.plot(cells, v_err, 's-', label='Velocity-profile relative RMSE')
    ax.set_xlabel('Grid cells')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Analytical Poiseuille verification under grid refinement')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, outdir / 'poiseuille_grid_convergence.png')

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(cells, runtime, 'o-')
    ax.set_xlabel('Grid cells')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Poiseuille runtime versus grid size')
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / 'poiseuille_runtime_vs_grid.png')

    # Obstacle pressure drop and runtime
    ocells = [to_float(r['cells']) for r in obstacle_rows]
    odp = [1e6 * to_float(r.get('pressure_drop_pa')) for r in obstacle_rows]
    ort = [to_float(r.get('runtime_s')) for r in obstacle_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(ocells, odp, 'o-')
    ax.set_xlabel('Grid cells')
    ax.set_ylabel('Pressure drop (µPa)')
    ax.set_title('Obstacle-case pressure drop versus grid size')
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / 'obstacle_pressure_drop_vs_grid.png')

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(ocells, ort, 'o-')
    ax.set_xlabel('Grid cells')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Obstacle-case runtime versus grid size')
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / 'obstacle_runtime_vs_grid.png')


def plot_pressure_loss_sweep(summary_rows, station_rows, outdir: Path):
    re_vals = [to_float(r['Re']) for r in summary_rows]
    clear_dp = [1e6 * to_float(r['clear_delta_p_pa']) for r in summary_rows]
    obstacle_dp = [1e6 * to_float(r['obstacle_delta_p_pa']) for r in summary_rows]
    excess_dp = [1e6 * to_float(r['excess_delta_p_pa']) for r in summary_rows]
    excess_k = [to_float(r['excess_coefficient']) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(re_vals, clear_dp, 'o-', label='Clear channel')
    ax.plot(re_vals, obstacle_dp, 's-', label='Obstacle channel')
    ax.plot(re_vals, excess_dp, '^-', label='Additional drop due to obstacle')
    ax.set_xlabel('Obstacle-height Reynolds number')
    ax.set_ylabel('Static pressure drop (µPa)')
    ax.set_title('Quantitative pressure-loss comparison for water at 20°C')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, outdir / 'final_pressure_loss_vs_reynolds.png')

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(re_vals, excess_k, 'o-')
    ax.set_xlabel('Obstacle-height Reynolds number')
    ax.set_ylabel('Excess pressure-drop coefficient (-)')
    ax.set_title('Additional resistance coefficient versus Reynolds number')
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / 'final_excess_coefficient_vs_reynolds.png')

    pairs = ['near', 'reference', 'far']
    for re_target in re_vals:
        rows = [r for r in station_rows if abs(to_float(r['Re']) - re_target) < 1e-9]
        values = [1e6 * to_float(next(r['excess_delta_p_pa'] for r in rows if r['pair'] == p)) for p in pairs]
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.bar(pairs, values)
        ax.set_xlabel('Pressure sampling station pair')
        ax.set_ylabel('Additional pressure drop (µPa)')
        ax.set_title(f'Station sensitivity of additional pressure drop at Re = {int(re_target)}')
        ax.grid(True, axis='y', alpha=0.3)
        save_plot(fig, outdir / f'station_sensitivity_Re{int(re_target)}.png')


def plot_benchmark_results(poi_rows, cyl_rows, outdir: Path):
    # Poiseuille benchmark summary could reuse same plot but separate verified plot
    cells = [to_float(r['cells']) for r in poi_rows]
    v_err = [100.0 * to_float(r['velocity_relative_rmse']) for r in poi_rows]
    p_err = [100.0 * to_float(r['pressure_relative_error']) for r in poi_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(cells, v_err, 'o-', label='Velocity-profile RMSE')
    ax.plot(cells, p_err, 's-', label='Pressure-drop error')
    ax.set_xlabel('Grid cells')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Verified Poiseuille benchmark error summary')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, outdir / 'verified_poiseuille_benchmark_errors.png')

    cyl_cells = [to_float(r['cells']) for r in cyl_rows]
    computed = [to_float(r['numerical_dp_coefficient']) for r in cyl_rows]
    reference = [to_float(r['published_reference_dp_coefficient']) for r in cyl_rows]
    deviation = [to_float(r['coefficient_error_percent']) for r in cyl_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(cyl_cells, computed, 'o-', label='Computed')
    ax.plot(cyl_cells, reference, 's--', label='Published reference')
    ax.set_xlabel('Grid cells')
    ax.set_ylabel('Front-to-rear pressure coefficient (-)')
    ax.set_title('DFG-inspired cylinder benchmark comparison at Re = 20')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, outdir / 'verified_cylinder_benchmark_comparison.png')

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(cyl_cells, deviation, 'o-')
    ax.set_xlabel('Grid cells')
    ax.set_ylabel('Deviation from published reference (%)')
    ax.set_title('Cylinder benchmark deviation versus grid size')
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / 'verified_cylinder_benchmark_error.png')


def plot_final_runtime_summary(poi_grid, obs_grid, pressure_rows, poi_bench, cyl_bench, outdir: Path):
    labels = []
    times = []

    def pick_mid(rows, name):
        if not rows:
            return
        row = rows[min(len(rows) // 2, len(rows) - 1)]
        labels.append(name)
        times.append(to_float(row.get('runtime_s', row.get('obstacle_runtime_s', float('nan')))))

    pick_mid(poi_grid, 'Poiseuille grid study')
    pick_mid(obs_grid, 'Obstacle grid study')
    pick_mid(pressure_rows, 'Pressure-loss sweep')
    pick_mid(poi_bench, 'Poiseuille benchmark')
    pick_mid(cyl_bench, 'Cylinder benchmark')

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(labels, times)
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Representative runtimes for verified study components')
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=18)
    save_plot(fig, outdir / 'runtime_summary_bar_chart.png')


def draw_obstacle_domain_schematic(outdir: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    domain = Rectangle((0, 0), 2.0, 1.0, fill=False, linewidth=2)
    obstacle = Rectangle((0.8, 0.3), 0.2, 0.4, facecolor='0.75', edgecolor='black')
    ax.add_patch(domain)
    ax.add_patch(obstacle)

    # pressure sampling bands
    band_w = 0.1
    x_pairs = [('near', 0.65, 1.25), ('reference', 0.60, 1.40), ('far', 0.50, 1.50)]
    for label, xu, xd in x_pairs:
        ax.add_patch(Rectangle((xu - band_w/2, 0), band_w, 1.0, fill=False, linestyle='--', linewidth=1.1))
        ax.add_patch(Rectangle((xd - band_w/2, 0), band_w, 1.0, fill=False, linestyle='--', linewidth=1.1))
        ax.text(xu, 1.03, f'{label}\nupstream', ha='center', va='bottom', fontsize=9)
        ax.text(xd, 1.03, f'{label}\ndownstream', ha='center', va='bottom', fontsize=9)

    # inlet / outlet arrows
    ax.annotate('', xy=(0.12, 0.5), xytext=(-0.18, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(-0.2, 0.62, 'Uniform inlet\nwater at 20°C', ha='left', va='center', fontsize=10)
    ax.annotate('', xy=(2.18, 0.5), xytext=(2.02, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(2.04, 0.62, 'Outlet\nreference pressure', ha='left', va='center', fontsize=10)

    # dimensions
    ax.annotate('', xy=(0, -0.08), xytext=(2.0, -0.08), arrowprops=dict(arrowstyle='<->', lw=1.2))
    ax.text(1.0, -0.12, '2.0 m', ha='center', va='top')
    ax.annotate('', xy=(-0.08, 0), xytext=(-0.08, 1.0), arrowprops=dict(arrowstyle='<->', lw=1.2))
    ax.text(-0.11, 0.5, '1.0 m', rotation=90, ha='right', va='center')

    ax.annotate('', xy=(0.8, 0.27), xytext=(1.0, 0.27), arrowprops=dict(arrowstyle='<->', lw=1.1))
    ax.text(0.9, 0.24, '0.2 m', ha='center', va='top', fontsize=9)
    ax.annotate('', xy=(1.03, 0.3), xytext=(1.03, 0.7), arrowprops=dict(arrowstyle='<->', lw=1.1))
    ax.text(1.06, 0.5, '0.4 m', rotation=90, ha='left', va='center', fontsize=9)

    ax.text(0.9, 0.5, 'Rectangular\nobstruction', ha='center', va='center', fontsize=10)
    ax.set_xlim(-0.25, 2.28)
    ax.set_ylim(-0.18, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Computational domain schematic for the engineering obstacle case')
    save_plot(fig, outdir / 'computational_domain_schematic_obstacle.png')


def draw_cylinder_domain_schematic(outdir: Path):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    domain = Rectangle((0, 0), 2.2, 0.41, fill=False, linewidth=2)
    cyl = Circle((0.2, 0.2), 0.05, facecolor='0.75', edgecolor='black')
    ax.add_patch(domain)
    ax.add_patch(cyl)

    ax.annotate('', xy=(0.12, 0.205), xytext=(-0.18, 0.205), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(-0.2, 0.30, 'Parabolic inlet\nbenchmark case', ha='left', va='center', fontsize=10)
    ax.annotate('', xy=(2.32, 0.205), xytext=(2.2, 0.205), arrowprops=dict(arrowstyle='->', lw=2))

    ax.annotate('', xy=(0, -0.05), xytext=(2.2, -0.05), arrowprops=dict(arrowstyle='<->', lw=1.2))
    ax.text(1.1, -0.07, '2.2 m', ha='center', va='top')
    ax.annotate('', xy=(-0.05, 0), xytext=(-0.05, 0.41), arrowprops=dict(arrowstyle='<->', lw=1.2))
    ax.text(-0.08, 0.205, '0.41 m', rotation=90, ha='right', va='center')

    ax.annotate('', xy=(0.15, 0.2), xytext=(0.25, 0.2), arrowprops=dict(arrowstyle='<->', lw=1.1))
    ax.text(0.2, 0.215, 'D = 0.1 m', ha='center', va='bottom', fontsize=9)
    ax.text(0.2, 0.2, 'Cylinder', ha='center', va='center', fontsize=10)
    ax.text(0.2, 0.11, '(0.2, 0.2)', ha='center', va='top', fontsize=9)

    ax.set_xlim(-0.25, 2.35)
    ax.set_ylim(-0.10, 0.47)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Computational domain schematic for the DFG-inspired cylinder benchmark')
    save_plot(fig, outdir / 'computational_domain_schematic_cylinder.png')


def write_verified_summary(outdir: Path, obstacle_grid_rows, pressure_rows, poi_bench_rows, cyl_bench_rows):
    outpath = outdir / 'verified_engineering_summary.csv'
    headers = [
        'category', 'selected_case', 'key_metric', 'value', 'units', 'notes'
    ]
    rows = []
    if obstacle_grid_rows:
        final_obs = obstacle_grid_rows[-1]
        rows.append({
            'category': 'Grid refinement',
            'selected_case': final_obs.get('mesh', 'n/a'),
            'key_metric': 'Obstacle pressure drop',
            'value': to_float(final_obs.get('pressure_drop_pa')) * 1e6,
            'units': 'µPa',
            'notes': 'Highest-resolution engineering obstacle case from grid study'
        })
    for r in pressure_rows:
        rows.append({
            'category': 'Pressure-loss sweep',
            'selected_case': f"Re {int(to_float(r['Re']))}",
            'key_metric': 'Additional pressure drop due to obstacle',
            'value': to_float(r.get('excess_delta_p_pa')) * 1e6,
            'units': 'µPa',
            'notes': f"Excess coefficient = {to_float(r.get('excess_coefficient')):.4f}"
        })
    if poi_bench_rows:
        best_poi = poi_bench_rows[-1]
        rows.append({
            'category': 'Benchmark verification',
            'selected_case': best_poi.get('mesh', 'n/a'),
            'key_metric': 'Poiseuille pressure-drop relative error',
            'value': 100.0 * to_float(best_poi.get('pressure_relative_error')),
            'units': '%',
            'notes': 'Analytical benchmark'
        })
    if cyl_bench_rows:
        best_cyl = cyl_bench_rows[-1]
        rows.append({
            'category': 'Benchmark comparison',
            'selected_case': best_cyl.get('mesh', 'n/a'),
            'key_metric': 'Cylinder pressure-coefficient deviation',
            'value': to_float(best_cyl.get('coefficient_error_percent')),
            'units': '%',
            'notes': 'Relative to published DFG-inspired reference'
        })

    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Generate final verified engineering plots and computational-domain schematics from completed study CSV files.')
    parser.add_argument('--root', default='results/studies', help='Root directory containing completed study folders.')
    parser.add_argument('--output', default='results/final_verified_artifacts', help='Output directory for final plots and schematics.')
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    poiseuille_grid = read_csv(root / 'water20c_poiseuille' / 'summary.csv')
    obstacle_grid = read_csv(root / 'water20c_obstacle' / 'summary.csv')
    pressure_summary = read_csv(root / 'pressure_loss_sweep' / 'summary.csv')
    station_comparisons = read_csv(root / 'pressure_loss_sweep' / 'station_comparisons.csv')
    poiseuille_bench = read_csv(root / 'benchmark_poiseuille' / 'poiseuille_summary.csv')
    cylinder_bench = read_csv(root / 'benchmark_cylinder' / 'cylinder_summary.csv')

    plot_grid_independence(poiseuille_grid, obstacle_grid, outdir)
    plot_pressure_loss_sweep(pressure_summary, station_comparisons, outdir)
    plot_benchmark_results(poiseuille_bench, cylinder_bench, outdir)
    plot_final_runtime_summary(poiseuille_grid, obstacle_grid, pressure_summary, poiseuille_bench, cylinder_bench, outdir)
    draw_obstacle_domain_schematic(outdir)
    draw_cylinder_domain_schematic(outdir)
    write_verified_summary(outdir, obstacle_grid, pressure_summary, poiseuille_bench, cylinder_bench)

    print(f'Final verified engineering plots and schematics saved to: {outdir}')


if __name__ == '__main__':
    main()
