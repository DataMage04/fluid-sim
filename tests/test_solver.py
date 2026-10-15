import numpy as np
import pytest

from grid import MACGrid
from solver import (
    compute_divergence,
    divergence_metrics,
    project_velocity,
    solve_pressure,
)


def test_uniform_horizontal_flow_has_zero_divergence():
    grid = MACGrid(nx=12, ny=8, width=2.0, height=1.0)
    grid.u[:] = 1.0
    grid.v[:] = 0.0

    divergence = compute_divergence(grid)

    assert np.allclose(divergence, 0.0)


def test_divergence_is_zeroed_inside_solid_cells():
    grid = MACGrid(nx=12, ny=8, width=2.0, height=1.0)
    grid.add_rectangle(0.8, 1.0, 0.25, 0.75)
    grid.u[:] = np.arange(grid.nx + 1)

    divergence = compute_divergence(grid)

    assert np.allclose(divergence[grid.solid], 0.0)


def test_divergence_metrics_match_manual_values():
    grid = MACGrid(nx=2, ny=2, width=2.0, height=2.0)
    grid.u[:] = 0.0
    grid.v[:] = 0.0
    grid.u[:, 1] = 1.0

    divergence = compute_divergence(grid)
    fluid = divergence[~grid.solid]

    expected_max = np.max(np.abs(fluid))
    expected_rms = np.sqrt(np.mean(fluid**2))

    max_div, rms_div = divergence_metrics(grid)

    assert np.isclose(max_div, expected_max)
    assert np.isclose(rms_div, expected_rms)


def test_pressure_projection_reduces_divergence():
    grid = MACGrid(nx=12, ny=8, width=2.0, height=1.0)

    # Deliberately construct a divergent horizontal velocity field.
    for i in range(grid.nx + 1):
        grid.u[:, i] = i / grid.nx
    grid.v[:] = 0.0

    _, rms_before = divergence_metrics(grid)

    solve_pressure(grid, dt=0.01, rho=1.0)
    project_velocity(grid, dt=0.01, rho=1.0)

    _, rms_after = divergence_metrics(grid)

    assert rms_before > 0.0
    assert rms_after < rms_before


def test_pressure_solve_keeps_solid_pressure_zero():
    grid = MACGrid(nx=12, ny=8, width=2.0, height=1.0)
    grid.add_rectangle(0.8, 1.0, 0.25, 0.75)
    grid.u[:] = 1.0
    grid.v[:] = 0.0

    solve_pressure(grid, dt=0.01, rho=1.0)

    assert np.allclose(grid.p[grid.solid], 0.0)


def test_pressure_solve_rejects_all_solid_grid():
    grid = MACGrid(nx=4, ny=4)
    grid.solid[:] = True

    with pytest.raises(ValueError, match="no fluid cells"):
        solve_pressure(grid, dt=0.01, rho=1.0)


def test_projection_preserves_impermeable_top_and_bottom_walls():
    grid = MACGrid(nx=8, ny=6)
    grid.p[:] = np.linspace(0.0, 1.0, grid.ny)[:, None]
    grid.v[:] = 2.0

    project_velocity(grid, dt=0.01, rho=1.0)

    assert np.allclose(grid.v[0, :], 0.0)
    assert np.allclose(grid.v[-1, :], 0.0)
