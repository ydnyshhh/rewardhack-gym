# Prompting And RL Workflows

RewardHack-Gym is intentionally usable as a backend for post-training experiments.

## Prompting

Use `PromptRunner` when you have a callable model or agent that maps a task to a final output. The runner returns a full `Trajectory`, not just a scalar score.

## Best-of-N

Use `BestOfNRunner` to sample multiple outputs and select under the official verifier. This is an important research setting because best-of-N often amplifies reward hacking:

- official score increases
- oracle score can stagnate or fall
- false-pass rates can rise sharply

## Rejection Filtering

Use `RejectionFilter` to retain only trajectories above an official threshold. This mirrors common SFT filtering pipelines and makes it easy to compare official-clean vs oracle-clean subsets.

## RL Reward Adapters

`RewardAdapter` and `AsyncRewardAdapter` expose official and oracle scores as reward-like records, along with:

- verifier gap
- false-pass status
- exploit labels
- rollout and evaluator version metadata

These records are designed to plug cleanly into external training systems without bundling a trainer into the package.
