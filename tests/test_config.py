import json

import numpy as np
import pytest

from fluid_sim.config import CaseConfig, load_case


def test_case_config_computes_kinematic_viscosity_and_reynolds_number():
    case = CaseConfig.from_dict(
        {
            "fluid": {
                "density": 2.0,
                "dynamic_viscosity": 0.04,
            },
            "inlet": {
                "mean_velocity": 3.0,
            },
            "solver": {
                "characteristic_length": 0.5,
            },
        }
    )

    assert np.isclose(case.fluid.kinematic_viscosity, 0.02)
    assert np.isclose(case.reynolds_number, 75.0)


def test_zero_viscosity_reports_infinite_reynolds_number():
    case = CaseConfig.from_dict(
        {
            "fluid": {
                "density": 1.0,
                "dynamic_viscosity": 0.0,
            }
        }
    )

    assert np.isinf(case.reynolds_number)


def test_load_case_reads_json(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "name": "small-test",
                "domain": {"nx": 8, "ny": 4},
                "time": {"dt": 0.01, "steps": 5},
            }
        ),
        encoding="utf-8",
    )

    case = load_case(path)

    assert case.name == "small-test"
    assert case.domain.nx == 8
    assert case.domain.ny == 4
    assert case.time.steps == 5


def test_case_rejects_unknown_inlet_profile():
    with pytest.raises(ValueError, match="Inlet profile"):
        CaseConfig.from_dict(
            {
                "inlet": {
                    "profile": "turbulent",
                }
            }
        )


def test_case_rejects_invalid_grid_size():
    with pytest.raises(ValueError, match="Grid dimensions"):
        CaseConfig.from_dict(
            {
                "domain": {
                    "nx": 0,
                    "ny": 10,
                }
            }
        )


def test_case_rejects_validation_location_outside_domain():
    with pytest.raises(ValueError, match="x_location"):
        CaseConfig.from_dict(
            {
                "domain": {
                    "width": 1.0,
                },
                "validation": {
                    "x_location": 1.5,
                },
            }
        )


def test_case_rejects_obstacle_outside_domain():
    with pytest.raises(ValueError, match="obstacle"):
        CaseConfig.from_dict(
            {
                "obstacles": [
                    {
                        "x_min": 3.0,
                        "x_max": 4.0,
                        "y_min": 0.2,
                        "y_max": 0.6,
                    }
                ]
            }
        )
