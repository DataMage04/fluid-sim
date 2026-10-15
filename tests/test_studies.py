import csv
import json
from pathlib import Path

import numpy as np

from fluid_sim.config import CaseConfig
from fluid_sim.studies.config import StudyConfig, StudyRunConfig
from fluid_sim.studies.runner import (
    _add_grid_convergence_orders,
    _grid_convergence_assessment,
    _set_nested_value,
    build_study_case,
    reprocess_study,
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
    assert Path(case.output.directory).parts[-2:] == ("runs", "coarse")


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


def test_grid_convergence_assessment_recovers_second_order_limit():
    # Both scalar quantities follow phi(h) = phi_exact + C h^2 and the
    # three values are close enough to the limiting solution for the
    # relative-GCI asymptotic ratio to be close to one.
    rows = [
        {
            "label": "coarse",
            "h_m": 0.1,
            "pressure_drop": 0.17,
            "max_speed_m_s": 1.46,
        },
        {
            "label": "medium",
            "h_m": 0.05,
            "pressure_drop": 0.1775,
            "max_speed_m_s": 1.49,
        },
        {
            "label": "fine",
            "h_m": 0.025,
            "pressure_drop": 0.179375,
            "max_speed_m_s": 1.4975,
        },
    ]

    assessment = _grid_convergence_assessment(rows)
    by_metric = {item["metric"]: item for item in assessment}

    pressure = by_metric["pressure_drop"]
    speed = by_metric["max_speed_m_s"]

    assert np.isclose(pressure["observed_order"], 2.0)
    assert np.isclose(pressure["richardson_extrapolated"], 0.18)
    assert np.isclose(speed["observed_order"], 2.0)
    assert np.isclose(speed["richardson_extrapolated"], 1.5)
    assert abs(float(pressure["asymptotic_ratio"]) - 1.0) < 0.02
    assert abs(float(speed["asymptotic_ratio"]) - 1.0) < 0.02


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


def test_reprocess_study_uses_existing_summary_without_running_cases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    study_path = cases_dir / "study.json"
    output_dir = tmp_path / "results" / "studies" / "reuse"
    output_dir.mkdir(parents=True)

    study_path.write_text(
        json.dumps(
            {
                "name": "reuse",
                "kind": "grid_convergence",
                "base_case": "unused.json",
                "output_directory": str(output_dir),
                "runs": [
                    {"label": "coarse"},
                    {"label": "medium"},
                    {"label": "fine"}
                ],
                "plots": []
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = [
        {
            "label": "coarse",
            "nx": 20,
            "ny": 10,
            "h_m": 0.1,
            "runtime_s": 1.0,
            "validation_relative_rmse": 0.04,
            "validation_relative_rmse_percent": 4.0,
            "pressure_drop_relative_error": 0.08,
            "pressure_drop_relative_error_percent": 8.0,
            "pressure_drop": 0.17,
            "max_speed_m_s": 1.46,
        },
        {
            "label": "medium",
            "nx": 40,
            "ny": 20,
            "h_m": 0.05,
            "runtime_s": 2.0,
            "validation_relative_rmse": 0.01,
            "validation_relative_rmse_percent": 1.0,
            "pressure_drop_relative_error": 0.02,
            "pressure_drop_relative_error_percent": 2.0,
            "pressure_drop": 0.1775,
            "max_speed_m_s": 1.49,
        },
        {
            "label": "fine",
            "nx": 80,
            "ny": 40,
            "h_m": 0.025,
            "runtime_s": 4.0,
            "validation_relative_rmse": 0.0025,
            "validation_relative_rmse_percent": 0.25,
            "pressure_drop_relative_error": 0.005,
            "pressure_drop_relative_error_percent": 0.5,
            "pressure_drop": 0.179375,
            "max_speed_m_s": 1.4975,
        },
    ]

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    reprocessed = reprocess_study(study_path)

    assert len(reprocessed) == 3
    assert (output_dir / "convergence_summary.csv").exists()
    report = (output_dir / "study_report.txt").read_text(encoding="utf-8")
    assert "GRID CONVERGENCE INDEX" in report
