import numpy as np
import pytest

from diffusion import diffuse_velocity, diffusion_timestep_limit
from grid import MACGrid


def test_diffusion_timestep_limit_matches_formula():
    grid = MACGrid(nx=20, ny=10, width=2.0, height=1.0)
    nu = 0.01

    expected = 0.5 / (nu * (1.0 / grid.dx**2 + 1.0 / grid.dy**2))

    assert np.isclose(diffusion_timestep_limit(grid, nu), expected)


def test_diffusion_timestep_limit_is_infinite_for_zero_viscosity():
    grid = MACGrid(nx=20, ny=10)

    assert np.isinf(diffusion_timestep_limit(grid, 0.0))


def test_zero_viscosity_leaves_velocity_unchanged():
    grid = MACGrid(nx=8, ny=6)
    rng = np.random.default_rng(1234)
    grid.u[:] = rng.normal(size=grid.u.shape)
    grid.v[:] = rng.normal(size=grid.v.shape)

    u_before = grid.u.copy()
    v_before = grid.v.copy()

    diffuse_velocity(grid, dt=0.01, nu=0.0)

    assert np.array_equal(grid.u, u_before)
    assert np.array_equal(grid.v, v_before)


def test_negative_viscosity_is_rejected():
    grid = MACGrid(nx=8, ny=6)

    with pytest.raises(ValueError, match="cannot be negative"):
        diffuse_velocity(grid, dt=0.01, nu=-0.01)


def test_unstable_diffusion_timestep_is_rejected():
    grid = MACGrid(nx=20, ny=10, width=2.0, height=1.0)
    nu = 0.01
    dt_limit = diffusion_timestep_limit(grid, nu)

    with pytest.raises(ValueError, match="unstable"):
        diffuse_velocity(grid, dt=1.01 * dt_limit, nu=nu)


def test_zero_velocity_remains_zero_after_diffusion():
    grid = MACGrid(nx=12, ny=8)

    diffuse_velocity(grid, dt=0.001, nu=0.01)

    assert np.allclose(grid.u, 0.0)
    assert np.allclose(grid.v, 0.0)


def test_diffusion_keeps_solid_normal_faces_impermeable():
    grid = MACGrid(nx=20, ny=10, width=2.0, height=1.0)
    grid.add_rectangle(0.8, 1.0, 0.2, 0.6)
    grid.u[:] = 1.0
    grid.v[:] = 1.0

    diffuse_velocity(grid, dt=0.001, nu=0.01)

    u_blocked = np.zeros_like(grid.u, dtype=bool)
    u_blocked[:, 1:-1] = grid.solid[:, :-1] | grid.solid[:, 1:]

    v_blocked = np.zeros_like(grid.v, dtype=bool)
    v_blocked[1:-1, :] = grid.solid[:-1, :] | grid.solid[1:, :]

    assert np.allclose(grid.u[u_blocked], 0.0)
    assert np.allclose(grid.v[v_blocked], 0.0)
    assert np.allclose(grid.v[0, :], 0.0)
    assert np.allclose(grid.v[-1, :], 0.0)
