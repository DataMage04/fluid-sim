from __future__ import annotations

import argparse

from fluid_sim.studies.runner import reprocess_study, run_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or reprocess a configured CFD parameter/grid-convergence study."
    )
    parser.add_argument(
        "study",
        help="Path to the study JSON file.",
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help=(
            "Reuse the existing summary.csv to regenerate convergence analysis, "
            "plots, and the study report without rerunning CFD cases."
        ),
    )
    args = parser.parse_args()

    if args.reuse_results:
        reprocess_study(args.study)
    else:
        run_study(args.study)


if __name__ == "__main__":
    main()
