"""Regression tests for pressure-loss postprocessing (fast, no long runs)."""
import numpy as np
import pytest

from fluid_sim.config import CaseConfig
from fluid_sim.core.grid import MACGrid
from studies.pressure_loss_comparison import (
    PAIRS, drops, late_variation, make_case, pressure_profile,
)

def test_linear_pressure_drop_matches_station_spacing():
    grid = MACGrid(nx=80, ny=40, width=2, height=1)
    x = (np.arange(grid.nx)+0.5)*grid.dx
    grid.p[:] = 5.0 - 2.0*x[None, :]
    measured = drops(grid, 0.1)
    assert measured["reference"] == pytest.approx(2.0 * 0.8, rel=1e-10)

def test_pressure_profile_excludes_solid_values():
    grid = MACGrid(nx=4, ny=2, width=2, height=1)
    grid.p[:] = 2
    grid.solid[0, 1] = True
    grid.p[0, 1] = 900
    _, profile = pressure_profile(grid)
    assert profile[1] == pytest.approx(2.0)

def test_settling_indicator():
    assert late_variation([1.0]*8) == pytest.approx(0)
    assert late_variation([1, 1, 1, 1, 1, 1.4, 1.7, 2]) > 0.02

def test_cases_match_flow_and_fluid():
    from argparse import Namespace
    base = CaseConfig.from_dict({
        "fluid": {"density": 998.2, "dynamic_viscosity": 0.001002},
        "obstacles": [{"x_min": 0.8, "x_max": 1.0, "y_min": 0.3, "y_max": 0.7}],
    })
    settings = Namespace(nx=80, ny=40, dt=25, duration=200)
    clear = make_case(base, 40, False, settings, "results/test-clear")
    obstacle = make_case(base, 40, True, settings, "results/test-obstacle")
    assert clear.inlet.profile == obstacle.inlet.profile == "uniform"
    assert clear.inlet.mean_velocity == obstacle.inlet.mean_velocity
    assert not clear.obstacles and len(obstacle.obstacles) == 1
    assert obstacle.reynolds_number == pytest.approx(40)
