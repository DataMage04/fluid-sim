import csv
import json
from pathlib import Path

import numpy as np

from fluid_sim.config import CaseConfig
from fluid_sim.studies.config import StudyConfig, StudyRunConfig
from fluid_sim.studies.runner import (
    _add_grid_convergence_orders,
    _set_nested_value,
    build_study_case,
    run_study,
)


def test_set_nested_value_supports_dicts_and_list_indices():
    data = {
        "domain": {"nx": 20},
        "obstacles": [{"y_max": 0.7}],
    }

    _set_nested_value(data, "domain.nx", 40)
    _set_nested_value(data, "obstacles.0.y_max", 0.8)

    assert data["domain"]["nx"] == 40
    assert np.isclose(data["obstacles"][0]["y_max"], 0.8)


def test_build_study_case_applies_overrides_and_disables_heavy_outputs(tmp_path):
    base_case = CaseConfig.from_dict(
        {
            "domain": {"nx": 20, "ny": 10, "width": 2.0, "height": 1.0},
            "validation": {"enabled": False},
        }
    )
    study = StudyConfig.from_dict(
        {
            "name": "grid",
            "base_case": "base.json",
            "output_directory": str(tmp_path / "study"),
            "runs": [
                {"label": "coarse", "overrides": {"domain.nx": 40}},
                {"label": "fine", "overrides": {"domain.nx": 80}},
            ],
        }
    )

    case = build_study_case(
        base_case,
        study,
        StudyRunConfig(label="coarse", overrides={"domain.nx": 40}),
        tmp_path / "study",
    )

    assert case.domain.nx == 40
    assert case.output.save_plots is False
    assert case.output.save_animation is False
    assert case.output.show_final_plot is False
    assert case.output.directory.endswith("runs/coarse")


def test_grid_convergence_observed_order_is_computed_pairwise():
    rows = [
        {"h_m": 0.1, "validation_relative_rmse": 0.04, "pressure_drop_relative_error": 0.08},
        {"h_m": 0.05, "validation_relative_rmse": 0.01, "pressure_drop_relative_error": 0.02},
        {"h_m": 0.025, "validation_relative_rmse": 0.0025, "pressure_drop_relative_error": 0.005},
    ]

    _add_grid_convergence_orders(rows)

    assert rows[0]["velocity_observed_order"] is None
    assert np.isclose(rows[1]["velocity_observed_order"], 2.0)
    assert np.isclose(rows[2]["velocity_observed_order"], 2.0)
    assert np.isclose(rows[1]["pressure_drop_observed_order"], 2.0)


def test_tiny_parameter_study_runs_and_writes_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    base_case_path = cases_dir / "base.json"
    base_case_path.write_text(
        json.dumps(
            {
                "name": "tiny",
                "domain": {"nx": 8, "ny": 4, "width": 2.0, "height": 1.0},
                "inlet": {"profile": "uniform", "mean_velocity": 0.5},
                "time": {"dt": 0.001, "steps": 1},
                "validation": {"enabled": False},
                "engineering": {"enabled": False},
                "output": {
                    "directory": "results/base",
                    "save_plots": False,
                    "save_animation": False,
                    "show_final_plot": False,
                    "frame_interval": 1,
                    "animation_fps": 20,
                    "diagnostics_interval": 1
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    study_path = cases_dir / "study.json"
    study_path.write_text(
        json.dumps(
            {
                "name": "tiny_sweep",
                "kind": "parameter_sweep",
                "base_case": "base.json",
                "output_directory": "results/studies/tiny_sweep",
                "runs": [
                    {"label": "u05", "overrides": {"inlet.mean_velocity": 0.5}},
                    {"label": "u075", "overrides": {"inlet.mean_velocity": 0.75}}
                ],
                "plots": [
                    {
                        "x": "mean_inlet_velocity_m_s",
                        "y": "max_speed_m_s",
                        "filename": "speed_sweep.png",
                        "title": "Speed sweep",
                        "x_label": "Inlet velocity (m/s)",
                        "y_label": "Maximum speed (m/s)"
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = run_study(study_path)

    output_dir = tmp_path / "results" / "studies" / "tiny_sweep"
    assert len(rows) == 2
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "study_report.txt").exists()
    assert (output_dir / "speed_sweep.png").exists()
    assert (output_dir / "runs" / "u05" / "output.txt").exists()
    assert (output_dir / "runs" / "u075" / "output.txt").exists()

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))

    assert len(records) == 2
    assert records[0]["label"] == "u05"
    assert records[1]["label"] == "u075"
