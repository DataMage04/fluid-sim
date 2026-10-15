import json

import numpy as np

from fluid_sim.config import CaseConfig
from fluid_sim.simulation import (
    build_grid,
    initialize_velocity,
    run_case,
)


def test_build_grid_applies_configured_obstacle():
    case = CaseConfig.from_dict(
        {
            "domain": {
                "nx": 20,
                "ny": 10,
                "width": 2.0,
                "height": 1.0,
            },
            "obstacles": [
                {
                    "x_min": 0.8,
                    "x_max": 1.0,
                    "y_min": 0.2,
                    "y_max": 0.6,
                }
            ],
        }
    )

    grid = build_grid(case)

    assert np.count_nonzero(grid.solid) > 0


def test_initialize_uniform_velocity_uses_case_value():
    case = CaseConfig.from_dict(
        {
            "inlet": {
                "profile": "uniform",
                "mean_velocity": 0.75,
            }
        }
    )
    grid = build_grid(case)

    initialize_velocity(grid, case)

    assert np.allclose(grid.u, 0.75)
    assert np.allclose(grid.v, 0.0)


def test_initialize_poiseuille_velocity_uses_case_value():
    case = CaseConfig.from_dict(
        {
            "domain": {
                "nx": 8,
                "ny": 4,
                "width": 2.0,
                "height": 1.0,
            },
            "inlet": {
                "profile": "poiseuille",
                "mean_velocity": 0.5,
            },
        }
    )
    grid = build_grid(case)

    initialize_velocity(grid, case)

    y = (np.arange(grid.ny) + 0.5) * grid.dy
    eta = y / grid.height
    expected = 6.0 * 0.5 * eta * (1.0 - eta)

    assert np.allclose(grid.u, expected[:, None])
    assert np.allclose(grid.v, 0.0)


def test_small_case_runs_without_main_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    case = CaseConfig.from_dict(
        {
            "name": "smoke",
            "domain": {
                "nx": 8,
                "ny": 4,
                "width": 2.0,
                "height": 1.0,
            },
            "inlet": {
                "profile": "uniform",
                "mean_velocity": 0.5,
            },
            "time": {
                "dt": 0.001,
                "steps": 1,
            },
            "validation": {
                "enabled": False,
            },
            "output": {
                "directory": "results/smoke",
                "save_plots": False,
                "save_animation": False,
                "show_final_plot": False,
                "frame_interval": 1,
                "animation_fps": 20,
                "diagnostics_interval": 1,
            },
        }
    )

    grid = run_case(case)

    assert np.all(np.isfinite(grid.u))
    assert np.all(np.isfinite(grid.v))
    assert np.all(np.isfinite(grid.p))

    result_directory = tmp_path / "results" / "smoke"
    output = result_directory / "output.txt"
    case_snapshot = result_directory / "resolved_case.json"

    assert output.exists()
    assert case_snapshot.exists()

    report = output.read_text(encoding="utf-8")
    assert "2D INCOMPRESSIBLE FLOW SIMULATION REPORT" in report
    assert "RUN SUMMARY" in report
    assert "MASS CONSERVATION" in report
    assert "PRESSURE PROJECTION / CONTINUITY" in report
    assert "No analytical validation configured" in report

    saved_case = json.loads(case_snapshot.read_text(encoding="utf-8"))
    assert saved_case["name"] == "smoke"
    assert saved_case["inlet"]["mean_velocity"] == 0.5
