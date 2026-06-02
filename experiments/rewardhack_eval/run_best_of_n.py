from __future__ import annotations

from _cli import runner_parser
from rewardhack_gym.experiments import run_best_of_n_experiment


def main() -> None:
    args = runner_parser("Run a RewardHack-Gym Best-of-N experiment.").parse_args()
    run_dir = run_best_of_n_experiment(
        config_path=args.config,
        out=args.out,
        dry_run=args.dry_run,
        dummy_model_mode=args.dummy_model_mode,
        overwrite=args.overwrite,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
