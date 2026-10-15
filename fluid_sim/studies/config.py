from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StudyRunConfig:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.label.strip():
            raise ValueError("Study run label cannot be empty.")
        if "/" in self.label or "\\" in self.label:
            raise ValueError("Study run labels cannot contain path separators.")
        for path in self.overrides:
            if not path.strip():
                raise ValueError("Override paths cannot be empty.")


@dataclass(frozen=True)
class StudyPlotConfig:
    x: str
    y: str
    filename: str
    title: str
    x_label: str
    y_label: str
    log_x: bool = False
    log_y: bool = False

    def validate(self) -> None:
        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("filename", self.filename),
            ("title", self.title),
            ("x_label", self.x_label),
            ("y_label", self.y_label),
        ):
            if not value.strip():
                raise ValueError(f"Study plot {name} cannot be empty.")
        if "/" in self.filename or "\\" in self.filename:
            raise ValueError("Study plot filename must be a filename, not a path.")


@dataclass(frozen=True)
class StudyConfig:
    name: str
    base_case: str
    output_directory: str
    runs: list[StudyRunConfig]
    kind: str = "parameter_sweep"
    keep_case_plots: bool = False
    keep_case_animation: bool = False
    plots: list[StudyPlotConfig] = field(default_factory=list)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Study name cannot be empty.")
        if not self.base_case.strip():
            raise ValueError("Study base_case cannot be empty.")
        if not self.output_directory.strip():
            raise ValueError("Study output_directory cannot be empty.")
        if self.kind not in {"parameter_sweep", "grid_convergence"}:
            raise ValueError(
                "Study kind must be 'parameter_sweep' or 'grid_convergence'."
            )
        if len(self.runs) < 2:
            raise ValueError("A study requires at least two runs.")

        labels = [run.label for run in self.runs]
        if len(labels) != len(set(labels)):
            raise ValueError("Study run labels must be unique.")

        for run in self.runs:
            run.validate()
        for plot in self.plots:
            plot.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StudyConfig":
        study = cls(
            name=data["name"],
            kind=data.get("kind", "parameter_sweep"),
            base_case=data["base_case"],
            output_directory=data["output_directory"],
            keep_case_plots=data.get("keep_case_plots", False),
            keep_case_animation=data.get("keep_case_animation", False),
            runs=[StudyRunConfig(**item) for item in data.get("runs", [])],
            plots=[StudyPlotConfig(**item) for item in data.get("plots", [])],
        )
        study.validate()
        return study


def load_study(path: str | Path) -> StudyConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return StudyConfig.from_dict(data)
