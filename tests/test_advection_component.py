import numpy as np

from advection import advect_velocity
from grid import MACGrid


def test_constant_velocity_field_is_preserved_by_advection():
    grid = MACGrid(nx=10, ny=6, width=2.0, height=1.0)
    grid.u[:] = 1.0
    grid.v[:] = 0.0

    advect_velocity(grid, dt=0.01)

    assert np.allclose(grid.u, 1.0)
    assert np.allclose(grid.v, 0.0)
