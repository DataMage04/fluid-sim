import numpy as np
import matplotlib.pyplot as plt

from fluid_sim.core.grid import MACGrid
from fluid_sim.analysis.visualization import (
    _symmetric_pressure_limit,
    get_cell_center_velocity,
    measure_flow_rates,
    reconstuct_cell_centred_velocities,
)


def test_get_cell_center_velocity_reconstructs_mac_faces():
    grid = MACGrid(nx=4, ny=3, width=2.0, height=1.5)

    grid.u[:] = 2.0
    grid.v[:] = -0.5

    u_center, v_center = get_cell_center_velocity(grid)

    assert u_center.shape == (grid.ny, grid.nx)
    assert v_center.shape == (grid.ny, grid.nx)
    assert np.allclose(u_center, 2.0)
    assert np.allclose(v_center, -0.5)


def test_get_cell_center_velocity_masks_solid_cells_with_nan():
    grid = MACGrid(nx=4, ny=3, width=2.0, height=1.5)
    grid.solid[1, 2] = True
    grid.u[:] = 1.0
    grid.v[:] = 1.0

    u_center, v_center = get_cell_center_velocity(grid)

    assert np.isnan(u_center[1, 2])
    assert np.isnan(v_center[1, 2])


def test_measure_flow_rates_reports_matching_uniform_flow():
    grid = MACGrid(nx=8, ny=4, width=2.0, height=1.0)
    grid.u[:] = 1.0

    report = measure_flow_rates(grid)

    assert len(report) == 3
    assert "Inlet flow rate" in report[0]
    assert "Outlet flow rate" in report[1]
    assert "0.000000e+00" in report[2]


def test_pressure_limit_uses_only_fluid_cells():
    grid = MACGrid(nx=4, ny=3, width=2.0, height=1.5)
    grid.solid[1, 2] = True

    pressure = np.array(
        [
            [0.0, -2.0, 1.0, 0.5],
            [1.5, 3.0, 1000.0, -1.0],
            [0.0, 2.0, -0.5, 1.0],
        ]
    )

    assert np.isclose(
        _symmetric_pressure_limit(pressure, grid.solid),
        3.0,
    )


def test_pressure_limit_falls_back_for_zero_field():
    grid = MACGrid(nx=4, ny=3, width=2.0, height=1.5)
    pressure = np.zeros((grid.ny, grid.nx))

    assert np.isclose(
        _symmetric_pressure_limit(pressure, grid.solid),
        1.0,
    )


def test_cell_centred_visualization_smoke(monkeypatch):
    """Exercise Matplotlib's contour stack without running a simulation."""
    grid = MACGrid(nx=6, ny=4, width=2.0, height=1.0)
    grid.u[:] = 1.0
    grid.v[:] = 0.0
    grid.solid[1:3, 2:4] = True

    # Avoid opening an interactive window during automated tests.
    monkeypatch.setattr(plt, "show", lambda: None)

    reconstuct_cell_centred_velocities(grid)
    plt.close("all")
