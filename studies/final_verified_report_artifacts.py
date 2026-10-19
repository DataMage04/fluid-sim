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
    poiseuille_rows = sorted(poiseuille_rows, key=lambda r: to_float(r['cells']))
    obstacle_rows = sorted(obstacle_rows, key=lambda r: to_float(r['cells']))
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
    summary_rows = sorted(summary_rows, key=lambda r: to_float(r['Re']))
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
    ax.set_title('Excess static-pressure-drop coefficient versus Reynolds number')
    ax.grid(True, alpha=0.3)
    save_plot(fig, outdir / 'final_excess_coefficient_vs_reynolds.png')

    pairs = ['near', 'reference', 'far']
    for re_target in re_vals:
        rows = [r for r in station_rows if abs(to_float(r['Re']) - re_target) < 1e-9]
        values = [1e6 * to_float(next(r['excess_delta_p_pa'] for r in rows if r['pair'] == p)) for p in pairs]
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.bar(['Near\n0.65–1.25 m', 'Reference\n0.60–1.40 m',
                'Far\n0.50–1.50 m'], values)
        ax.set_xlabel('Pressure sampling station pair')
        ax.set_ylabel('Additional pressure drop (µPa)')
        ax.set_title(f'Station sensitivity of additional pressure drop at Re = {int(re_target)}')
        ax.grid(True, axis='y', alpha=0.3)
        save_plot(fig, outdir / f'station_sensitivity_Re{int(re_target)}.png')


def plot_benchmark_results(poi_rows, cyl_rows, outdir: Path):
    poi_rows = sorted(poi_rows, key=lambda r: to_float(r['cells']))
    cyl_rows = sorted(cyl_rows, key=lambda r: to_float(r['cells']))
    cells = [to_float(r['cells']) for r in poi_rows]
    v_err = [100.0 * to_float(r['velocity_relative_rmse']) for r in poi_rows]
    p_err = [100.0 * to_float(r['pressure_relative_error']) for r in poi_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(cells, v_err, 'o-', label='Velocity-profile RMSE')
    ax.plot(cells, p_err, 's-', label='Pressure-drop error')
    ax.set_xlabel('Grid cells')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Analytical Poiseuille benchmark errors (water at 20°C)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, outdir / 'verified_poiseuille_benchmark_errors.png')

    computed = [to_float(r['numerical_dp_coefficient']) for r in cyl_rows]
    reference = [to_float(r['published_reference_dp_coefficient']) for r in cyl_rows]
    deviation = [to_float(r['coefficient_error_percent']) for r in cyl_rows]
    grids = [str(r['mesh']) for r in cyl_rows]
    if len(cyl_rows) == 1:
        # A single point cannot demonstrate grid convergence: use paired bars.
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.bar(['Numerical: '+grids[0], 'Published DFG reference'],
               [computed[0], reference[0]], width=0.65)
        ax.set_ylabel('Front-to-rear pressure coefficient (-)')
        ax.set_title('DFG-inspired Re = 20 pressure comparison (single grid)')
        ax.grid(True, axis='y', alpha=0.3)
        for i, val in enumerate([computed[0], reference[0]]):
            ax.annotate(f'{val:.4f}', (i, val), xytext=(0, 6),
                        textcoords='offset points', ha='center', va='bottom')
        ax.set_ylim(0, max(computed[0], reference[0]) * 1.2)
        save_plot(fig, outdir / 'verified_cylinder_benchmark_comparison.png')

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.bar([grids[0]], [deviation[0]], width=0.45)
        ax.set_ylabel('Absolute deviation from published reference (%)')
        ax.set_xlabel('Computed mesh')
        ax.set_title('DFG-inspired Re = 20 deviation (no grid study yet)')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, deviation[0] * 1.25 if deviation[0] else 1.0)
        ax.annotate(f'{deviation[0]:.2f}%', (0, deviation[0]),
                    xytext=(0, 6), textcoords='offset points', ha='center')
        save_plot(fig, outdir / 'verified_cylinder_benchmark_error.png')
    else:
        cells = [to_float(r['cells']) for r in cyl_rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(cells, computed, 'o-', label='Computed')
        ax.plot(cells, reference, 's--', label='Published reference')
        ax.set(xlabel='Grid cells', ylabel='Front-to-rear pressure coefficient (-)',
               title='DFG-inspired cylinder comparison at Re = 20')
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_plot(fig, outdir / 'verified_cylinder_benchmark_comparison.png')

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(cells, deviation, 'o-')
        ax.set(xlabel='Grid cells',
               ylabel='Absolute deviation from published reference (%)',
               title='DFG-inspired benchmark deviation by grid size')
        ax.grid(True, alpha=0.3)
        save_plot(fig, outdir / 'verified_cylinder_benchmark_error.png')

def plot_final_runtime_summary(poi_grid, obs_grid, pressure_rows, poi_bench, cyl_bench, outdir: Path):
    """Observed runtimes with both paired pressure-loss CFD runs counted."""
    def representative(rows, key='cells'):
        sorted_rows = sorted(rows, key=lambda r: to_float(r.get(key)))
        return sorted_rows[(len(sorted_rows) - 1) // 2]

    selected_poi = representative(poi_grid)
    selected_obs = representative(obs_grid)
    selected_bench = representative(poi_bench)
    selected_cyl = representative(cyl_bench)
    selected_pair = min(pressure_rows, key=lambda r: abs(to_float(r['Re']) - 40))
    paired_runtime = (to_float(selected_pair['clear_runtime_s']) +
                      to_float(selected_pair['obstacle_runtime_s']))
    datasets = [
        (f"Poiseuille grid {selected_poi['mesh']} (800 steps)",
         to_float(selected_poi['runtime_s'])),
        (f"Obstacle grid {selected_obs['mesh']} (800 steps)",
         to_float(selected_obs['runtime_s'])),
        (f"Paired pressure runs Re={to_float(selected_pair['Re']):g} "
         f"({selected_pair['nx']}×{selected_pair['ny']}; 2 × {selected_pair['steps']} steps)",
         paired_runtime),
        (f"Poiseuille benchmark {selected_bench['mesh']} "
         f"({to_float(selected_bench['physical_duration_s'])/to_float(selected_bench['dt_s']):.0f} steps)",
         to_float(selected_bench['runtime_s'])),
        (f"Cylinder benchmark {selected_cyl['mesh']} "
         f"({to_float(selected_cyl['physical_duration_s'])/to_float(selected_cyl['dt_s']):.0f} steps)",
         to_float(selected_cyl['runtime_s'])),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    labels = [d[0] for d in datasets]
    durations = [d[1] / 60 for d in datasets]
    ax.barh(labels[::-1], durations[::-1], height=0.65)
    ax.set_xlabel('Observed wall-clock runtime (minutes)')
    ax.set_title('Recorded computational costs (different grids and step counts)')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, max(durations) * 1.18)
    for i, val in enumerate(durations[::-1]):
        ax.annotate(f'{val:.1f} min', (val, i),
                    xytext=(5, 0), textcoords='offset points', va='center')
    fig.text(0.02, 0.015, 'Runs have different grids, timesteps and durations; '
             'do not use this chart to infer speedup or computational scaling.',
             fontsize=8, color='0.35')
    save_plot(fig, outdir / 'runtime_summary_bar_chart.png')

def draw_obstacle_domain_schematic(outdir: Path):
    """Dimensioned engineering channel schematic with readable station labels.

    Draw only the reference sampling pair: three overlapping pairs cannot be
    legibly labelled at this scale. Other station coordinates appear below.
    """
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.add_patch(Rectangle((0, 0), 2.0, 1.0,
                           facecolor='0.98', edgecolor='black', linewidth=1.7))
    ax.add_patch(Rectangle((0.8, 0.3), 0.2, 0.4,
                           facecolor='0.65', edgecolor='black', linewidth=1.3))
    for x, label in ((0.60, 'Upstream\nx = 0.60 m'),
                     (1.40, 'Downstream\nx = 1.40 m')):
        ax.add_patch(Rectangle((x - 0.05, 0), 0.10, 1.0,
                               fill=False, linestyle='--', linewidth=1.8,
                               edgecolor='tab:blue'))
        ax.annotate(label, xy=(x, 1.00), xytext=(x, 1.12),
                    ha='center', va='bottom', fontsize=10,
                    arrowprops=dict(arrowstyle='-', lw=0.9, color='0.35'))
    ax.text(0.90, 0.52, 'Solid', ha='center', va='center',
            fontsize=9, rotation=90)
    ax.annotate('', xy=(0, 0.5), xytext=(-0.40, 0.5),
                arrowprops=dict(arrowstyle='->', lw=1.8))
    ax.text(-0.43, 0.69, 'Uniform inlet\nWater at 20°C', fontsize=10,
            ha='left', va='bottom')
    ax.annotate('', xy=(2.40, 0.5), xytext=(2.0, 0.5),
                arrowprops=dict(arrowstyle='->', lw=1.8))
    ax.text(2.04, 0.69, 'Outlet', fontsize=10, ha='left', va='bottom')
    ax.annotate('', xy=(0, -0.09), xytext=(2.0, -0.09),
                arrowprops=dict(arrowstyle='<->', lw=1.2))
    ax.text(1.0, -0.12, 'Channel length: 2.00 m', fontsize=10,
            ha='center', va='top')
    ax.annotate('', xy=(-0.10, 0.0), xytext=(-0.10, 1.0),
                arrowprops=dict(arrowstyle='<->', lw=1.2))
    ax.text(-0.14, 0.5, '1.00 m', fontsize=10, rotation=90,
            ha='right', va='center')
    ax.text(1.0, -0.28,
            'Other sampling pairs (upstream → downstream): near 0.65 → 1.25 m; '
            'far 0.50 → 1.50 m',
            fontsize=9.5, ha='center', va='top')
    ax.text(1.0, -0.36, 'Solid block: x = 0.80–1.00 m, y = 0.30–0.70 m | '
            'Sampling-band width = 0.10 m',
            fontsize=9, ha='center', va='top', color='0.35')
    ax.set(xlim=(-0.48, 2.48), ylim=(-0.46, 1.34))
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_title('Computational domain: rectangular-obstacle pressure-loss study',
                 fontsize=13, pad=18)
    save_plot(fig, outdir / 'computational_domain_schematic_obstacle.png')


def draw_cylinder_domain_schematic(outdir: Path):
    """Full DFG-inspired channel at scale plus a separate enlarged cylinder."""
    fig, (ax, detail) = plt.subplots(
        1, 2, figsize=(14, 5.3),
        gridspec_kw={'width_ratios': [3.9, 1.5]})
    ax.add_patch(Rectangle((0, 0), 2.2, 0.41,
                           facecolor='0.98', edgecolor='black', linewidth=1.6))
    ax.add_patch(Circle((0.2, 0.2), 0.05,
                        facecolor='0.65', edgecolor='black', linewidth=1.2))
    ax.add_patch(Rectangle((0.09, 0.09), 0.22, 0.22, fill=False,
                           edgecolor='tab:blue', linestyle='--', linewidth=1.3))
    ax.annotate('', xy=(0.0, 0.205), xytext=(-0.25, 0.205),
                arrowprops=dict(arrowstyle='->', lw=1.8))
    ax.text(0.02, 0.62, 'Parabolic inlet, water at 20°C',
            ha='left', va='bottom', fontsize=10)
    ax.annotate('', xy=(2.42, 0.205), xytext=(2.2, 0.205),
                arrowprops=dict(arrowstyle='->', lw=1.8))
    ax.text(2.21, 0.32, 'Outlet', ha='left', va='bottom', fontsize=10)
    ax.annotate('', xy=(0.0, -0.065), xytext=(2.2, -0.065),
                arrowprops=dict(arrowstyle='<->', lw=1.2))
    ax.text(1.1, -0.09, 'Channel length: 2.20 m', ha='center', va='top')
    ax.annotate('', xy=(-0.07, 0), xytext=(-0.07, 0.41),
                arrowprops=dict(arrowstyle='<->', lw=1.2))
    ax.text(-0.11, 0.205, '0.41 m', rotation=90, ha='right', va='center')
    ax.text(1.1, 0.47, 'Circle centred at (0.20, 0.20) m; see enlargement →',
            fontsize=10, ha='center')
    ax.set(xlim=(-0.31, 2.49), ylim=(-0.15, 0.72))
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_title('Full computational domain', fontsize=12, pad=15)

    detail.add_patch(Circle((0.2, 0.2), 0.05,
                            facecolor='0.65', edgecolor='black', linewidth=1.5))
    detail.scatter([0.15, 0.25], [0.2, 0.2], s=30,
                   marker='o', zorder=4, color='tab:blue')
    detail.annotate('Front pressure', xy=(0.15, 0.2),
                    xytext=(0.105, 0.295), ha='left', va='bottom', fontsize=9,
                    arrowprops=dict(arrowstyle='->', lw=1.1))
    detail.annotate('Rear pressure', xy=(0.25, 0.2),
                    xytext=(0.285, 0.13), ha='center', va='top', fontsize=9,
                    arrowprops=dict(arrowstyle='->', lw=1.1))
    detail.text(0.2, 0.105, 'D = 0.10 m', ha='center', va='center', fontsize=10)
    detail.set(xlim=(0.085, 0.32), ylim=(0.085, 0.34))
    detail.set_aspect('equal', adjustable='box')
    detail.set_axis_off()
    detail.set_title('Cylinder and pressure probes (enlarged)', fontsize=12, pad=15)
    fig.suptitle('DFG 2D-1-inspired benchmark geometry, Re = 20',
                 fontsize=14, y=0.98)
    save_plot(fig, outdir / 'computational_domain_schematic_cylinder.png')


def write_verified_summary(outdir: Path, obstacle_grid_rows, pressure_rows, poi_bench_rows, cyl_bench_rows):
    outpath = outdir / 'verified_engineering_summary.csv'
    headers = [
        'category', 'selected_case', 'key_metric', 'value', 'units', 'notes'
    ]
    rows = []
    if obstacle_grid_rows:
        final_obs = max(obstacle_grid_rows, key=lambda r: to_float(r['cells']))
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
        best_poi = max(poi_bench_rows, key=lambda r: to_float(r['cells']))
        rows.append({
            'category': 'Benchmark verification',
            'selected_case': best_poi.get('mesh', 'n/a'),
            'key_metric': 'Poiseuille pressure-drop relative error',
            'value': 100.0 * to_float(best_poi.get('pressure_relative_error')),
            'units': '%',
            'notes': 'Analytical benchmark'
        })
    if cyl_bench_rows:
        best_cyl = max(cyl_bench_rows, key=lambda r: to_float(r['cells']))
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


def verify_study_tables(grid_poiseuille, grid_obstacle, sweep, stations,
                        benchmark_poiseuille, benchmark_cylinder):
    """Fail fast if report graphics would misrepresent inconsistent CSV data."""
    groups = {
        'grid Poiseuille': grid_poiseuille,
        'grid obstacle': grid_obstacle,
        'pressure sweep': sweep,
        'station comparison': stations,
        'Poiseuille benchmark': benchmark_poiseuille,
        'cylinder benchmark': benchmark_cylinder,
    }
    for name, rows in groups.items():
        if not rows:
            raise ValueError(f'{name} CSV is empty; cannot generate verified figures.')

    def require_close(name, actual, expected, *, rel_tol=2e-7):
        if not (abs(actual - expected) <= rel_tol * max(abs(expected), 1e-14)):
            raise ValueError(f'{name} discrepancy: CSV has {actual:g}, '
                             f'but direct calculation gives {expected:g}.')

    for row in sweep:
        re = to_float(row['Re'])
        label = f'pressure sweep Re={re:g}'
        dp_clear = to_float(row['clear_delta_p_pa'])
        dp_obstacle = to_float(row['obstacle_delta_p_pa'])
        excess = to_float(row['excess_delta_p_pa'])
        require_close(label + ' excess drop', excess, dp_obstacle - dp_clear)
        if 'dynamic_pressure_pa' in row:
            q = to_float(row['dynamic_pressure_pa'])
            require_close(label + ' pressure coefficient',
                          to_float(row['excess_coefficient']), excess / q)
        if 'fluid' in row and row['fluid'] != 'water at 20 C':
            raise ValueError(f'{label} has an unexpected fluid: {row["fluid"]}')
        if 'density_kg_m3' in row:
            require_close(label + ' water density', to_float(row['density_kg_m3']), 998.2)
        if 'dynamic_viscosity_pa_s' in row:
            require_close(label + ' water viscosity',
                          to_float(row['dynamic_viscosity_pa_s']), 0.001002)
        matching = [station for station in stations
                    if abs(to_float(station['Re']) - re) < 1e-9
                    and station['pair'] == 'reference']
        if len(matching) != 1:
            raise ValueError(f'{label} needs exactly one reference station row.')
        require_close(label + ' reference station',
                      to_float(matching[0]['excess_delta_p_pa']), excess)

    for row in benchmark_poiseuille:
        if 'numerical_delta_p_pa' in row:
            num = to_float(row['numerical_delta_p_pa'])
            reference = to_float(row['analytical_delta_p_pa'])
            require_close('analytical Poiseuille pressure error',
                          to_float(row['pressure_relative_error']),
                          abs(num - reference) / abs(reference))
    for row in benchmark_cylinder:
        numerical = to_float(row['numerical_dp_coefficient'])
        published = to_float(row['published_reference_dp_coefficient'])
        require_close('published cylinder reference deviation',
                      to_float(row['coefficient_error_percent']),
                      100 * abs(numerical / published - 1))
    for row in grid_poiseuille:
        if 'analytical_pressure_drop_pa' in row:
            num = to_float(row['pressure_drop_pa'])
            reference = to_float(row['analytical_pressure_drop_pa'])
            require_close('grid Poiseuille pressure error',
                          to_float(row['pressure_relative_error']),
                          abs(num - reference) / abs(reference))
    print('Verified numerical consistency of pressure drops, coefficients '
          'and published-reference deviations against input CSV files.')


def write_figure_review_notes(outdir, grid_obstacle, sweep, cyl_rows):
    obs_rows = sorted(grid_obstacle, key=lambda r: to_float(r['cells']))
    grid_pct = float('nan')
    if len(obs_rows) >= 2:
        coarse = to_float(obs_rows[-2]['pressure_drop_pa'])
        fine = to_float(obs_rows[-1]['pressure_drop_pa'])
        grid_pct = 100 * abs(fine - coarse) / abs(coarse)
    cylinder = max(cyl_rows, key=lambda r: to_float(r['cells']))
    lines = [
        'FINAL FIGURE REVIEW — NUMERICAL RESULTS AND LIMITATIONS',
        '',
        'All charts in this folder were regenerated from existing study CSV files.',
        'This script does NOT rerun the CFD solver or independently validate its physics.',
        'Working fluid: liquid water at 20°C as specified in the input studies.',
        '',
        f'Obstacle pressure-drop grid change, two finest meshes: {grid_pct:.3f}%.',
        'This is a grid sensitivity indicator, not proof of grid independence.',
        f'Published DFG-inspired cylinder comparison: '
        f'{to_float(cylinder["coefficient_error_percent"]):.3f}% deviation.',
        f'Cylinder benchmark mesh count: {len(cyl_rows)}.',
        'One cylinder mesh cannot demonstrate grid convergence.',
        'The cylinder geometry is stair-stepped and uses a different outlet treatment.',
        'Pressure-loss sweep measures excess STATIC pressure drop, not total-pressure loss.',
        'The pressure-sweep settling PASS flags are screening indicators only.',
        'The runtime chart reports observed times at different grids/step counts;',
        'pressure-sweep timing includes BOTH clear and obstructed runs.',
        '',
        'For publication, confirm dimensions, legend styles and captions '
        'against the original input configurations.',
    ]
    (outdir / 'FIGURE_REVIEW_NOTES.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


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

    verify_study_tables(poiseuille_grid, obstacle_grid, pressure_summary,
                        station_comparisons, poiseuille_bench, cylinder_bench)
    plot_grid_independence(poiseuille_grid, obstacle_grid, outdir)
    plot_pressure_loss_sweep(pressure_summary, station_comparisons, outdir)
    plot_benchmark_results(poiseuille_bench, cylinder_bench, outdir)
    plot_final_runtime_summary(poiseuille_grid, obstacle_grid, pressure_summary, poiseuille_bench, cylinder_bench, outdir)
    draw_obstacle_domain_schematic(outdir)
    draw_cylinder_domain_schematic(outdir)
    write_verified_summary(outdir, obstacle_grid, pressure_summary, poiseuille_bench, cylinder_bench)
    write_figure_review_notes(outdir, obstacle_grid, pressure_summary, cylinder_bench)

    print(f'Final verified engineering plots and schematics saved to: {outdir}')


if __name__ == '__main__':
    main()
