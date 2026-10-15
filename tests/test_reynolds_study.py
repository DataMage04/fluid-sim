from pathlib import Path

import numpy as np

from fluid_sim.config import load_case
from fluid_sim.studies.config import load_study
from fluid_sim.studies.runner import build_study_case


REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY_PATH = REPO_ROOT / "cases" / "studies" / "obstacle_reynolds.json"


def test_obstacle_reynolds_study_generates_expected_reynolds_numbers(tmp_path):
    study = load_study(STUDY_PATH)
    base_case_path = STUDY_PATH.parent / study.base_case
    base_case = load_case(base_case_path.resolve())

    expected = [10.0, 20.0, 40.0, 80.0]
    actual = []

    for run in study.runs:
        case = build_study_case(base_case, study, run, tmp_path / "study")
        actual.append(case.reynolds_number)

        assert case.domain.nx == 60
        assert case.domain.ny == 30
        assert np.isclose(case.inlet.mean_velocity, 1.0)
        assert case.output.save_plots is False
        assert case.output.save_animation is False

    assert np.allclose(actual, expected)


def test_re10_case_remains_within_explicit_diffusion_limit():
    study = load_study(STUDY_PATH)
    base_case = load_case((STUDY_PATH.parent / study.base_case).resolve())
    re10 = build_study_case(base_case, study, study.runs[0], "results/test")

    dx = re10.domain.width / re10.domain.nx
    dy = re10.domain.height / re10.domain.ny
    nu = re10.fluid.kinematic_viscosity
    diffusion_limit = 0.5 / (nu * (1.0 / dx**2 + 1.0 / dy**2))

    assert re10.time.dt < diffusion_limit
