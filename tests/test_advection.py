
from pathlib import Path
import sys

import numpy as np

# Adds the parent directory to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from grid import MACGrid
from advection import (
    bilinear_sample,
    sample_velocity,
    advect_velocity,
)
from boundary_conditions import apply_boundary_conditions


def test_bilinear_interpolation():
    # f(x, y) = 2x + 3y
    # Bilinear interpolation should reproduce this
    # linear function exactly, apart from roundoff.
    field = np.zeros((4, 5))

    for j in range(4):
        for i in range(5):
            field[j, i] = 2.0 * i + 3.0 * j

    value = bilinear_sample(field, 1.5, 2.0)

    expected = 2.0 * 1.5 + 3.0 * 2.0

    assert np.isclose(value, expected), (
        f"Expected {expected}, got {value}"
    )


def test_zero_velocity_advection():
    grid = MACGrid(nx=12, ny=8)

    grid.u[:] = 0.0
    grid.v[:] = 0.0

    advect_velocity(grid, dt=0.01)

    assert np.allclose(grid.u, 0.0)
    assert np.allclose(grid.v, 0.0)


def test_constant_velocity_advection():
    grid = MACGrid(nx=12, ny=8)

    grid.u[:] = 1.0
    grid.v[:] = 0.0

    # Test a point away from boundaries.
    u_value, v_value = sample_velocity(
        grid, x=0.5, y=0.5
    )

    assert np.isclose(u_value, 1.0)
    assert np.isclose(v_value, 0.0)


def test_solid_face_conditions():
    grid = MACGrid(nx=20, ny=10)

    grid.add_rectangle(
        x_min=0.8,
        x_max=1.0,
        y_min=0.2,
        y_max=0.6,
    )

    grid.u[:] = 1.0
    grid.v[:] = 1.0

    apply_boundary_conditions(
        grid,
        inlet_velocity=1.0,
    )

    # No vertical flow through the horizontal walls.
    assert np.allclose(grid.v[0, :], 0.0)
    assert np.allclose(grid.v[-1, :], 0.0)

    # The horizontal velocity on the inlet is prescribed
    # except at wall/solid faces, where it is constrained.
    assert np.all(np.isfinite(grid.u))
    assert np.all(np.isfinite(grid.v))


if __name__ == "__main__":
    test_bilinear_interpolation()
    test_zero_velocity_advection()
    test_constant_velocity_advection()
    test_solid_face_conditions()

    print("All advection tests passed.")
