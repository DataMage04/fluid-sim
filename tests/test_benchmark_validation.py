"""Fast numerical tests for the analytical/published benchmark driver."""
import numpy as np
import pytest
from fluid_sim.core.grid import MACGrid
from studies.benchmark_validation import (
    DFG_CDP, DFG_DP, NU, MU, RHO, cylinder_probe, surface_pressure
)

def test_reference_coefficient_scaling():
    assert DFG_CDP == pytest.approx(DFG_DP / (0.5 * 1.0 * 0.2**2))
    water_speed = 20 * NU / 0.1
    assert water_speed == pytest.approx(0.0002007613704668403)
    assert DFG_CDP * 0.5 * RHO * water_speed**2 > 0.0

def test_cylinder_is_rasterized_inside_domain():
    grid = MACGrid(nx=110, ny=41, width=2.2, height=0.41)
    grid.add_circle(0.2, 0.2, 0.05)
    assert np.count_nonzero(grid.solid) > 0
    assert not np.any(grid.solid[:,0])
    assert not np.any(grid.solid[:,-1])

def test_surface_extrapolation_of_linear_pressure_field():
    grid = MACGrid(nx=110, ny=41, width=2.2, height=0.41)
    grid.add_circle(0.2, 0.2, 0.05)
    x = (np.arange(grid.nx)+0.5)*grid.dx
    grid.p[:,:] = 2.0 - 3.0 * x[None,:]
    front = surface_pressure(grid, 'front')
    rear = surface_pressure(grid, 'rear')
    assert front == pytest.approx(2-3*0.15)
    assert rear == pytest.approx(2-3*0.25)
    calculated = cylinder_probe(grid, 20*NU/0.1)
    assert calculated['dp_pa'] == pytest.approx(0.3)
