import numpy as np
import pytest

from fluid_sim.numerics.boundary_conditions import (
    apply_boundary_conditions,
    apply_poiseuille_inlet,
)
from fluid_sim.core.grid import MACGrid


def test_poiseuille_inlet_matches_analytical_profile():
    grid = MACGrid(nx=12, ny=8, width=2.0, height=1.0)
    mean_velocity = 1.25

    apply_poiseuille_inlet(grid, mean_velocity)

    y = (np.arange(grid.ny) + 0.5) * grid.dy
    eta = y / grid.height
    expected = 6.0 * mean_velocity * eta * (1.0 - eta)

    assert np.allclose(grid.u[:, 0], expected)


def test_boundary_conditions_enforce_top_and_bottom_impermeability():
    grid = MACGrid(nx=12, ny=8)
    grid.v[:] = 3.0

    apply_boundary_conditions(grid, inlet_velocity=1.0)

    assert np.allclose(grid.v[0, :], 0.0)
    assert np.allclose(grid.v[-1, :], 0.0)


def test_boundary_conditions_zero_faces_blocked_by_solid_cells():
    grid = MACGrid(nx=20, ny=10, width=2.0, height=1.0)
    grid.add_rectangle(0.8, 1.0, 0.2, 0.6)
    grid.u[:] = 2.0
    grid.v[:] = 2.0

    apply_boundary_conditions(grid, inlet_velocity=1.0)

    u_blocked = np.zeros_like(grid.u, dtype=bool)
    u_blocked[:, 1:-1] = grid.solid[:, :-1] | grid.solid[:, 1:]

    v_blocked = np.zeros_like(grid.v, dtype=bool)
    v_blocked[1:-1, :] = grid.solid[:-1, :] | grid.solid[1:, :]

    assert np.allclose(grid.u[u_blocked], 0.0)
    assert np.allclose(grid.v[v_blocked], 0.0)


def test_boundary_conditions_clear_pressure_inside_solids():
    grid = MACGrid(nx=20, ny=10, width=2.0, height=1.0)
    grid.add_rectangle(0.8, 1.0, 0.2, 0.6)
    grid.p[:] = 5.0

    apply_boundary_conditions(grid, inlet_velocity=1.0)

    assert np.allclose(grid.p[grid.solid], 0.0)


@pytest.mark.xfail(
    reason=(
        "Known Milestone 7 issue: apply_boundary_conditions currently "
        "hardcodes mean_velocity=1.0 instead of using inlet_velocity."
    ),
    strict=True,
)
def test_boundary_conditions_honour_inlet_velocity_argument():
    grid = MACGrid(nx=12, ny=8, width=2.0, height=1.0)
    requested_velocity = 0.4

    apply_boundary_conditions(grid, inlet_velocity=requested_velocity)

    y = (np.arange(grid.ny) + 0.5) * grid.dy
    eta = y / grid.height
    expected = 6.0 * requested_velocity * eta * (1.0 - eta)

    assert np.allclose(grid.u[:, 0], expected)
