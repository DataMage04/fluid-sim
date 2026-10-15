import numpy as np
import pytest

from fluid_sim.core.grid import MACGrid


def test_mac_grid_array_shapes_and_spacing():
    grid = MACGrid(nx=8, ny=4, width=2.0, height=1.0)

    assert grid.p.shape == (4, 8)
    assert grid.u.shape == (4, 9)
    assert grid.v.shape == (5, 8)
    assert grid.solid.shape == (4, 8)
    assert np.isclose(grid.dx, 0.25)
    assert np.isclose(grid.dy, 0.25)


def test_cell_center_coordinates_are_correct():
    grid = MACGrid(nx=4, ny=2, width=2.0, height=1.0)
    x, y = grid.cell_centres()

    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    assert np.allclose(x[0], [0.25, 0.75, 1.25, 1.75])
    assert np.allclose(y[:, 0], [0.25, 0.75])


def test_staggered_velocity_coordinates_match_array_shapes():
    grid = MACGrid(nx=4, ny=2, width=2.0, height=1.0)

    xu, yu = grid.u_coordinates()
    xv, yv = grid.v_coordinates()

    assert xu.shape == grid.u.shape
    assert yu.shape == grid.u.shape
    assert xv.shape == grid.v.shape
    assert yv.shape == grid.v.shape


def test_add_rectangle_marks_expected_cells():
    # Use exactly representable binary fractions so this test checks geometry
    # logic rather than floating-point division edge cases.
    grid = MACGrid(nx=4, ny=4, width=1.0, height=1.0)

    grid.add_rectangle(
        x_min=0.25,
        x_max=0.75,
        y_min=0.25,
        y_max=0.75,
    )

    expected = np.zeros((4, 4), dtype=bool)
    expected[1:3, 1:3] = True

    assert np.array_equal(grid.solid, expected)


def test_add_rectangle_rejects_non_overlapping_geometry():
    grid = MACGrid(nx=10, ny=10, width=1.0, height=1.0)

    with pytest.raises(ValueError):
        grid.add_rectangle(
            x_min=2.0,
            x_max=3.0,
            y_min=2.0,
            y_max=3.0,
        )
