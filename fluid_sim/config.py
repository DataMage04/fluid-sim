from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class FluidConfig:
    density: float = 1.0
    dynamic_viscosity: float = 0.01

    @property
    def kinematic_viscosity(self) -> float:
        return self.dynamic_viscosity / self.density

    def validate(self) -> None:
        if self.density <= 0.0:
            raise ValueError("Fluid density must be positive.")
        if self.dynamic_viscosity < 0.0:
            raise ValueError("Dynamic viscosity cannot be negative.")


@dataclass
class DomainConfig:
    nx: int = 60
    ny: int = 30
    width: float = 2.0
    height: float = 1.0

    def validate(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Grid dimensions nx and ny must be positive.")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Domain width and height must be positive.")


@dataclass
class RectangleObstacle:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def validate(self) -> None:
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError(
                "Obstacle minimum coordinates must be below maximum coordinates."
            )

    def overlaps_domain(self, domain: DomainConfig) -> bool:
        return not (
            self.x_max <= 0.0
            or self.x_min >= domain.width
            or self.y_max <= 0.0
            or self.y_min >= domain.height
        )


@dataclass
class InletConfig:
    profile: str = "poiseuille"
    mean_velocity: float = 1.0

    def validate(self) -> None:
        if self.profile not in {"poiseuille", "uniform"}:
            raise ValueError(
                "Inlet profile must be either 'poiseuille' or 'uniform'."
            )
        if self.mean_velocity < 0.0:
            raise ValueError("Mean inlet velocity cannot be negative.")


@dataclass
class TimeConfig:
    dt: float = 0.005
    steps: int = 1000

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("Time step dt must be positive.")
        if self.steps <= 0:
            raise ValueError("Number of time steps must be positive.")


@dataclass
class SolverConfig:
    characteristic_length: float = 0.4

    def validate(self) -> None:
        if self.characteristic_length <= 0.0:
            raise ValueError("Characteristic length must be positive.")


@dataclass
class ValidationConfig:
    enabled: bool = True
    x_location: float = 1.5

    def validate(self) -> None:
        if self.x_location < 0.0:
            raise ValueError("Validation x_location cannot be negative.")


@dataclass
class EngineeringConfig:
    enabled: bool = False
    upstream_x: float = 0.25
    downstream_x: float = 1.75
    sample_width: float = 0.1
    reference_pressure: float = 0.0

    def validate(self, domain: DomainConfig) -> None:
        if not self.enabled:
            return
        if self.upstream_x < 0.0 or self.upstream_x > domain.width:
            raise ValueError("Engineering upstream_x must lie inside the domain.")
        if self.downstream_x < 0.0 or self.downstream_x > domain.width:
            raise ValueError("Engineering downstream_x must lie inside the domain.")
        if self.upstream_x >= self.downstream_x:
            raise ValueError("Engineering upstream_x must be smaller than downstream_x.")
        if self.sample_width <= 0.0:
            raise ValueError("Engineering sample_width must be positive.")


@dataclass
class OutputConfig:
    directory: str = "results/default"
    save_plots: bool = True
    save_animation: bool = False
    show_final_plot: bool = True
    frame_interval: int = 5
    animation_fps: int = 20
    diagnostics_interval: int = 100

    def validate(self) -> None:
        if not self.directory.strip():
            raise ValueError("Output directory cannot be empty.")
        if self.frame_interval <= 0:
            raise ValueError("frame_interval must be positive.")
        if self.animation_fps <= 0:
            raise ValueError("animation_fps must be positive.")
        if self.diagnostics_interval <= 0:
            raise ValueError("diagnostics_interval must be positive.")


@dataclass
class CaseConfig:
    name: str = "default"
    fluid: FluidConfig = field(default_factory=FluidConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    inlet: InletConfig = field(default_factory=InletConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    engineering: EngineeringConfig = field(default_factory=EngineeringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    obstacles: list[RectangleObstacle] = field(default_factory=list)

    @property
    def reynolds_number(self) -> float:
        if self.fluid.dynamic_viscosity == 0.0:
            return float("inf")
        return (
            self.fluid.density
            * self.inlet.mean_velocity
            * self.solver.characteristic_length
            / self.fluid.dynamic_viscosity
        )

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Case name cannot be empty.")

        self.fluid.validate()
        self.domain.validate()
        self.inlet.validate()
        self.time.validate()
        self.solver.validate()
        self.validation.validate()
        self.engineering.validate(self.domain)
        self.output.validate()

        if (
            self.validation.enabled
            and self.validation.x_location > self.domain.width
        ):
            raise ValueError(
                "Validation x_location must lie inside the configured domain."
            )

        for obstacle in self.obstacles:
            obstacle.validate()
            if not obstacle.overlaps_domain(self.domain):
                raise ValueError(
                    "Configured obstacle does not overlap the simulation domain."
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseConfig":
        case = cls(
            name=data.get("name", "default"),
            fluid=FluidConfig(**data.get("fluid", {})),
            domain=DomainConfig(**data.get("domain", {})),
            inlet=InletConfig(**data.get("inlet", {})),
            time=TimeConfig(**data.get("time", {})),
            solver=SolverConfig(**data.get("solver", {})),
            validation=ValidationConfig(**data.get("validation", {})),
            engineering=EngineeringConfig(**data.get("engineering", {})),
            output=OutputConfig(**data.get("output", {})),
            obstacles=[
                RectangleObstacle(**item)
                for item in data.get("obstacles", [])
            ],
        )
        case.validate()
        return case

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_case(path: str | Path) -> CaseConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return CaseConfig.from_dict(data)


def save_case(case: CaseConfig, path: str | Path) -> None:
    case.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(case.to_dict(), file, indent=2)
        file.write("\n")
