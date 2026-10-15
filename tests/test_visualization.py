import numpy as np

from fluid_sim.core.grid import MACGrid
from fluid_sim.analysis.visualization import get_cell_center_velocity, measure_flow_rates


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
