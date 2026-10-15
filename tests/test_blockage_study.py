from pathlib import Path

import numpy as np

from fluid_sim.config import load_case
from fluid_sim.numerics.diffusion import diffusion_timestep_limit
from fluid_sim.simulation import build_grid
from fluid_sim.studies.blockage import obstacle_geometry_metrics
from fluid_sim.studies.config import load_study
from fluid_sim.studies.runner import build_study_case


def test_blockage_study_keeps_reynolds_number_constant():
    root = Path(__file__).resolve().parents[1]
    study_path = root / "cases" / "studies" / "obstacle_blockage.json"
    study = load_study(study_path)
    base_case = load_case(study_path.parent / study.base_case)

    expected_blockage = [0.2, 0.3, 0.4, 0.5, 0.6]

    for run, expected_beta in zip(study.runs, expected_blockage, strict=True):
        case = build_study_case(base_case, study, run, root / "tmp-study")
        obstacle_height, blockage_ratio = obstacle_geometry_metrics(case)
        obstacle = case.obstacles[0]

        assert np.isclose(blockage_ratio, expected_beta)
        assert np.isclose(obstacle_height, expected_beta * case.domain.height)
        assert np.isclose(obstacle.y_min + obstacle.y_max, case.domain.height)
        assert np.isclose(case.solver.characteristic_length, obstacle_height)
        assert np.isclose(case.reynolds_number, 40.0)

        grid = build_grid(case)
        dt_limit = diffusion_timestep_limit(
            grid,
            case.fluid.kinematic_viscosity,
        )
        assert case.time.dt <= dt_limit
