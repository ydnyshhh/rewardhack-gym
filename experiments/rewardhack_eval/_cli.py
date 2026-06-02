from __future__ import annotations

import argparse


def runner_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dummy-model-mode",
        choices=("canonical_true", "canonical_exploit", "random_bad"),
        default="canonical_true",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser

