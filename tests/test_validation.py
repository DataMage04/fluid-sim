import numpy as np

from fluid_sim.core.grid import MACGrid
from fluid_sim.analysis.validation import (
    analytical_poiseuille_profile,
    numerical_velocity_profile,
    profile_error,
    validate_poiseuille,
)


def test_analytical_poiseuille_profile_has_expected_shape():
    y = np.array([0.0, 0.5, 1.0])

    u = analytical_poiseuille_profile(
        y,
        height=1.0,
        mean_velocity=1.0,
    )

    assert np.allclose(u, [0.0, 1.5, 0.0])


def test_profile_error_is_zero_for_identical_profiles():
    profile = np.array([0.2, 0.8, 1.2, 0.8, 0.2])

    errors = profile_error(profile, profile)

    assert np.isclose(errors["rmse"], 0.0)
    assert np.isclose(errors["max_error"], 0.0)
    assert np.isclose(errors["relative_rmse"], 0.0)


def test_numerical_velocity_profile_reconstructs_face_velocity():
    grid = MACGrid(nx=6, ny=4, width=3.0, height=1.0)
    row_values = np.array([0.2, 0.6, 1.0, 0.4])

    for j, value in enumerate(row_values):
        grid.u[j, :] = value

    y, u = numerical_velocity_profile(grid, x_location=1.5)

    assert y.shape == (grid.ny,)
    assert np.allclose(u, row_values)


def test_validate_poiseuille_accepts_exact_discrete_profile(tmp_path):
    grid = MACGrid(nx=12, ny=8, width=2.0, height=1.0)
    mean_velocity = 1.0

    y = (np.arange(grid.ny) + 0.5) * grid.dy
    profile = analytical_poiseuille_profile(
        y,
        height=grid.height,
        mean_velocity=mean_velocity,
    )

    for j, value in enumerate(profile):
        grid.u[j, :] = value

    output_file = tmp_path / "poiseuille_validation.png"
    errors, report = validate_poiseuille(
        grid,
        mean_velocity=mean_velocity,
        x_location=1.0,
        filename=str(output_file),
    )

    assert np.isclose(errors["rmse"], 0.0)
    assert np.isclose(errors["max_error"], 0.0)
    assert output_file.exists()
    assert any("POISEUILLE VALIDATION" in line for line in report)
