from __future__ import annotations

import argparse

from fluid_sim.studies.runner import run_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a configured CFD parameter or grid-convergence study."
    )
    parser.add_argument(
        "study",
        help="Path to the study JSON file.",
    )
    args = parser.parse_args()
    run_study(args.study)


if __name__ == "__main__":
    main()
