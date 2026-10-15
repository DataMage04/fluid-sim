"""Automated parameter studies and grid-convergence analysis."""

from fluid_sim.studies.config import StudyConfig, StudyPlotConfig, StudyRunConfig, load_study
from fluid_sim.studies.runner import run_study

__all__ = [
    "StudyConfig",
    "StudyPlotConfig",
    "StudyRunConfig",
    "load_study",
    "run_study",
]
