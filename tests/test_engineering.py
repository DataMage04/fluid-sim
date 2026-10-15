import numpy as np

from fluid_sim.analysis.engineering import (
    analytical_poiseuille_pressure_drop,
    analytical_poiseuille_pressure_gradient,
    compute_engineering_results,
    dynamic_pressure,
    maximum_velocity_location,
    mean_pressure_in_x_band,
    pressure_coefficient,
    pressure_drop,
    pressure_force_on_solids,
    reverse_flow_metrics,
    solid_reference_geometry,
    streamwise_pressure_profile,
    wake_extent,
)
from fluid_sim.core.grid import MACGrid


def test_mean_pressure_in_x_band_averages_fluid_cells():
    grid = MACGrid(nx=4, ny=2, width=2.0, height=1.0)
    grid.p[:, :] = np.array([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])

    result = mean_pressure_in_x_band(grid, x_location=0.75, width=0.5)

    assert np.isclose(result, 3.0)


def test_pressure_drop_is_upstream_minus_downstream():
    grid = MACGrid(nx=4, ny=2, width=2.0, height=1.0)
    grid.p[:, :] = np.array([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])

    upstream, downstream, delta_p = pressure_drop(
        grid,
        upstream_x=0.25,
        downstream_x=1.75,
        sample_width=0.5,
    )

    assert np.isclose(upstream, 4.0)
    assert np.isclose(downstream, 1.0)
    assert np.isclose(delta_p, 3.0)


def test_dynamic_pressure_and_pressure_coefficient():
    assert np.isclose(dynamic_pressure(1.2, 2.0), 2.4)
    assert np.isclose(
        pressure_coefficient(
            pressure=3.4,
            reference_pressure=1.0,
            density=1.2,
            reference_velocity=2.0,
        ),
        1.0,
    )


def test_streamwise_pressure_profile_ignores_solid_cells():
    grid = MACGrid(nx=3, ny=2, width=3.0, height=2.0)
    grid.p[:, :] = np.array([[3.0, 100.0, 1.0], [3.0, 2.0, 1.0]])
    grid.solid[0, 1] = True

    x, pressure, cp = streamwise_pressure_profile(
        grid,
        density=1.0,
        reference_velocity=2.0,
        reference_pressure=0.0,
    )

    assert np.allclose(x, [0.5, 1.5, 2.5])
    assert np.allclose(pressure, [3.0, 2.0, 1.0])
    assert np.allclose(cp, [1.5, 1.0, 0.5])


def test_maximum_velocity_location_uses_cell_centres():
    grid = MACGrid(nx=3, ny=2, width=3.0, height=2.0)
    grid.u[:] = 0.0
    grid.v[:] = 0.0
    grid.u[1, 1:3] = 4.0

    speed, x, y = maximum_velocity_location(grid)

    assert np.isclose(speed, 4.0)
    assert np.isclose(x, 1.5)
    assert np.isclose(y, 1.5)


def test_reverse_flow_metrics_detect_negative_streamwise_velocity():
    grid = MACGrid(nx=3, ny=2, width=3.0, height=2.0)
    grid.u[:] = 1.0
    grid.v[:] = 0.0

    grid.u[0, 1] = -2.0
    grid.u[0, 2] = -2.0

    min_u, x, y, fraction = reverse_flow_metrics(grid)

    assert np.isclose(min_u, -2.0)
    assert np.isclose(x, 1.5)
    assert np.isclose(y, 0.5)
    assert np.isclose(fraction, 0.5)


def test_pressure_force_on_solid_uses_adjacent_fluid_pressure():
    grid = MACGrid(nx=3, ny=3, width=3.0, height=3.0)
    grid.solid[1, 1] = True
    grid.p[1, 0] = 4.0
    grid.p[1, 2] = 1.0
    grid.p[0, 1] = 2.0
    grid.p[2, 1] = 2.0

    drag, lift = pressure_force_on_solids(grid)

    assert np.isclose(drag, 3.0)
    assert np.isclose(lift, 0.0)


def test_solid_reference_geometry_and_wake_extent():
    grid = MACGrid(nx=3, ny=3, width=3.0, height=3.0)
    grid.solid[1, 1] = True
    grid.u[:] = 1.0
    grid.v[:] = 0.0
    grid.u[1, 2] = -1.0
    grid.u[1, 3] = -1.0

    reference_height, rear_x = solid_reference_geometry(grid)
    wake_length, wake_end_x = wake_extent(
        grid,
        wake_start_x=rear_x,
        velocity_tolerance=1.0e-3,
    )

    assert np.isclose(reference_height, 1.0)
    assert np.isclose(rear_x, 2.0)
    assert np.isclose(wake_end_x, 3.0)
    assert np.isclose(wake_length, 1.0)


def test_poiseuille_pressure_drop_formula():
    gradient = analytical_poiseuille_pressure_gradient(
        dynamic_viscosity=0.01,
        mean_velocity=1.0,
        channel_height=1.0,
    )
    delta_p = analytical_poiseuille_pressure_drop(
        dynamic_viscosity=0.01,
        mean_velocity=1.0,
        channel_height=1.0,
        distance=1.5,
    )

    assert np.isclose(gradient, 0.12)
    assert np.isclose(delta_p, 0.18)


def test_compute_engineering_results_compares_pressure_drop():
    grid = MACGrid(nx=4, ny=2, width=2.0, height=1.0)
    grid.p[:, :] = np.array([[0.24, 0.18, 0.12, 0.06], [0.24, 0.18, 0.12, 0.06]])
    grid.u[:] = 1.0
    grid.v[:] = 0.0

    results = compute_engineering_results(
        grid,
        density=1.0,
        reference_velocity=1.0,
        upstream_x=0.25,
        downstream_x=1.75,
        sample_width=0.5,
        analytical_pressure_gradient=0.12,
    )

    assert np.isclose(results.pressure_drop, 0.18)
    assert np.isclose(results.analytical_pressure_drop, 0.18)
    assert np.isclose(results.pressure_drop_relative_error, 0.0)
    assert np.isclose(results.pressure_drop_coefficient, 0.36)
    assert np.isclose(results.reverse_flow_fraction, 0.0)
    assert results.pressure_drag_coefficient is None
    assert results.wake_length is None


def test_compute_engineering_results_adds_solid_force_metrics():
    grid = MACGrid(nx=3, ny=3, width=3.0, height=3.0)
    grid.solid[1, 1] = True
    grid.p[:, :] = 0.0
    grid.p[1, 0] = 4.0
    grid.p[1, 2] = 1.0
    grid.u[:] = 1.0
    grid.v[:] = 0.0
    grid.u[1, 2] = -1.0
    grid.u[1, 3] = -1.0

    results = compute_engineering_results(
        grid,
        density=1.0,
        reference_velocity=2.0,
        upstream_x=0.5,
        downstream_x=2.5,
        sample_width=1.0,
    )

    assert np.isclose(results.pressure_drag_per_unit_depth, 3.0)
    assert np.isclose(results.pressure_drag_coefficient, 1.5)
    assert np.isclose(results.pressure_lift_per_unit_depth, 0.0)
    assert np.isclose(results.pressure_lift_coefficient, 0.0)
    assert np.isclose(results.wake_length, 1.0)
    assert np.isclose(results.wake_end_x, 3.0)
