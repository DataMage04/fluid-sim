import argparse

from fluid_sim.config import load_case
from fluid_sim.simulation import run_case


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a configured 2D incompressible fluid simulation."
    )
    parser.add_argument(
        "case",
        nargs="?",
        default="cases/poiseuille.json",
        help=(
            "Path to a JSON case file "
            "(default: cases/poiseuille.json)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    case = load_case(args.case)
    run_case(case)


if __name__ == "__main__":
    main()
