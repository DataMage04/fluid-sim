"""Fast regression coverage for report schematics and source-data QA."""
from pathlib import Path

import pytest

from studies.final_verified_report_artifacts import (
    draw_cylinder_domain_schematic,
    draw_obstacle_domain_schematic,
    verify_study_tables,
    write_figure_review_notes,
)


def _rows():
    grid_poi = [{'cells': '800', 'pressure_drop_pa': '1.8e-6',
                 'analytical_pressure_drop_pa': '2e-6',
                 'pressure_relative_error': '0.1'}]
    grid_obs = [
        {'cells': '800', 'pressure_drop_pa': '2.8e-5'},
        {'cells': '3200', 'pressure_drop_pa': '2.7e-5'},
    ]
    sweep = [
        {'Re': '40', 'clear_delta_p_pa': '1e-6', 'obstacle_delta_p_pa': '5e-6',
         'excess_delta_p_pa': '4e-6', 'dynamic_pressure_pa': '2e-6',
         'excess_coefficient': '2', 'fluid': 'water at 20 C',
         'density_kg_m3': '998.2', 'dynamic_viscosity_pa_s': '0.001002'}
    ]
    stations = [{'Re': '40', 'pair': 'reference', 'excess_delta_p_pa': '4e-6'}]
    bench_poi = [{'numerical_delta_p_pa': '1.8e-6',
                  'analytical_delta_p_pa': '2e-6',
                  'pressure_relative_error': '0.1'}]
    bench_cyl = [{'cells': '4510', 'mesh': '110x41',
                  'numerical_dp_coefficient': '5.0',
                  'published_reference_dp_coefficient': '5.876',
                  'coefficient_error_percent': str(100 * abs(5.0 / 5.876 - 1))}]
    return grid_poi, grid_obs, sweep, stations, bench_poi, bench_cyl


def test_input_consistency_accepts_matched_pressure_data():
    verify_study_tables(*_rows())


def test_input_consistency_rejects_bad_excess_pressure():
    groups = _rows()
    groups[2][0]['excess_delta_p_pa'] = '6e-6'
    with pytest.raises(ValueError, match='excess drop'):
        verify_study_tables(*groups)


def test_figures_have_content_and_review_notes(tmp_path: Path):
    draw_obstacle_domain_schematic(tmp_path)
    draw_cylinder_domain_schematic(tmp_path)
    for filename in ('computational_domain_schematic_obstacle.png',
                     'computational_domain_schematic_cylinder.png'):
        path = tmp_path / filename
        assert path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
        assert path.stat().st_size > 10_000
    rows = _rows()
    write_figure_review_notes(tmp_path, rows[1], rows[2], rows[5])
    text = (tmp_path / 'FIGURE_REVIEW_NOTES.txt').read_text(encoding='utf-8')
    assert 'Cylinder benchmark mesh count: 1' in text
    assert 'does NOT rerun the CFD solver' in text
