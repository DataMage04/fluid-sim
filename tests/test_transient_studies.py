import json

import numpy as np

from fluid_sim.studies.transient import (
    TransientAnalysisConfig,
    summarize_history,
)


def _history(values):
    rows = []
    for index, value in enumerate(values, start=1):
        rows.append(
            {
                "step": index,
                "time_s": float(index),
                "reynolds_number": 40.0,
                "pressure_drop": value,
                "pressure_drag_coefficient": 2.0 * value,
                "wake_length": 0.5,
                "reverse_flow_fraction": 0.05,
                "min_streamwise_velocity": -0.2,
                "max_speed": 2.3,
            }
        )
    return rows


def test_transient_config_loads_from_study_file(tmp_path):
    path = tmp_path / "study.json"
    path.write_text(
        json.dumps(
            {
                "transient_analysis": {
                    "sample_interval": 5,
                    "averaging_start_fraction": 0.5,
                    "steadiness_tolerance": 0.02,
                    "minimum_average_samples": 4,
                }
            }
        ),
        encoding="utf-8",
    )

    config = TransientAnalysisConfig.from_study_file(path)

    assert config.sample_interval == 5
    assert np.isclose(config.averaging_start_fraction, 0.5)
    assert np.isclose(config.steadiness_tolerance, 0.02)
    assert config.minimum_average_samples == 4


def test_summarize_history_marks_flat_window_steady():
    config = TransientAnalysisConfig(
        averaging_start_fraction=0.5,
        steadiness_tolerance=0.01,
        minimum_average_samples=5,
    )
    summary = summarize_history(_history([1.0] * 20), config)

    assert summary["steady_by_window_test"] is True
    assert summary["steadiness_status"] == "STEADY"
    assert np.isclose(summary["pressure_drop_mean"], 1.0)
    assert np.isclose(summary["pressure_drop_std"], 0.0)


def test_summarize_history_marks_drifting_window_unsteady():
    config = TransientAnalysisConfig(
        averaging_start_fraction=0.5,
        steadiness_tolerance=0.01,
        minimum_average_samples=5,
    )
    values = [1.0] * 10 + list(np.linspace(1.0, 1.3, 10))
    summary = summarize_history(_history(values), config)

    assert summary["steady_by_window_test"] is False
    assert summary["steadiness_status"] == "UNSTEADY/DEVELOPING"
    assert float(summary["max_relative_drift"]) > 0.01


def test_summarize_history_rejects_too_short_average_window():
    config = TransientAnalysisConfig(
        averaging_start_fraction=0.9,
        minimum_average_samples=5,
    )

    try:
        summarize_history(_history([1.0] * 10), config)
    except ValueError as exc:
        assert "too few samples" in str(exc)
    else:
        raise AssertionError("Expected averaging-window validation error.")
