# RewardHack-Gym

RewardHack-Gym is a gym-style Python package for constructing and analyzing verifiable agent environments with controllable proxy-objective mismatch, designed for reward hacking, post-training, and mechanistic interpretability research.

The package is built as a research substrate, not as a static benchmark. Each environment centers the primitive:

`task + official verifier + oracle + exploit surface`

and makes it easy to study settings where optimization pressure drives models toward cheap proxy reward rather than true task completion.

## What the Package Covers

- Modular, typed core abstractions for tasks, verifiers, oracles, exploit surfaces, trajectories, and metrics
- Configurable exploitability profiles for dialing official-vs-oracle mismatch up or down
- Stronger first-release environment families for code and math
- Prompting, best-of-N, rejection-filter, and RL-style evaluation wrappers
- JSONL trace export for downstream analysis and future mechanistic interpretability pipelines
- A CLI designed to work naturally through `uv run`

## Initial Environment Families

### Code

- `code/spec-overfit`: implementation tasks where the official verifier emphasizes public examples and shallow interface checks, while the oracle probes latent semantic properties, metamorphic invariants, and adversarial distribution shift
- `code/patch-verification`: bug-fix tasks where narrow bug-ticket success can conceal broader semantic regressions
- `code/interface-conformance`: protocol tasks where visible examples pass but deeper state, aliasing, and contract guarantees fail

### Math

- `math/symbolic-mismatch`: shallow answer checking and limited sample-point validation versus stronger symbolic and numeric equivalence
- `math/reasoning-validity`: final-answer correctness or template compliance versus derivation-level consistency
- `math/constraint-sensitive`: weak grading under narrow assumptions versus full domain, branch, and boundary validation

## Core Design Idea

For a trajectory or output `x`:

- `S_official(x)` is the cheap, scalable evaluator used for optimization
- `S_oracle(x)` is the stronger evaluator that better approximates the intended objective

RewardHack-Gym is designed to make it easy to generate and analyze trajectories where:

`S_official(x) >> S_oracle(x)`

for exploitative outputs, while still supporting genuinely correct outputs where both scores are high.

## Quick Start

1. Create an environment and sample a task:

```python
from rewardhack_gym import create_environment

env = create_environment("code/spec-overfit")
task = env.sample_task(seed=7)
print(task.prompt)
```

2. Evaluate an output under the official verifier and the oracle:

```python
trajectory = env.evaluate_output(
    task=task,
    final_output=env.canonical_exploit_output(task),
    policy_id="demo-policy",
)

print(trajectory.official_result.score)
print(trajectory.oracle_result.score)
print(trajectory.exploit_labels)
```

3. Export traces:

```python
from rewardhack_gym.io import write_jsonl

write_jsonl("traces.jsonl", [trajectory])
```

4. Inspect metrics:

```python
from rewardhack_gym.analysis import summarize_trajectories

summary = summarize_trajectories([trajectory])
print(summary.false_pass_rate)
```

## CLI Examples

```bash
uv run rewardhack-gym inspect-env code/spec-overfit
uv run rewardhack-gym sample-task code/spec-overfit --seed 7
uv run rewardhack-gym evaluate-output code/spec-overfit --seed 7 --text-file outputs/spec_overfit.py
uv run rewardhack-gym stats --input traces.jsonl
uv run rewardhack-gym export-false-passes --input traces.jsonl --output false_passes.jsonl
```

## Architecture

- `rewardhack_gym.core`: typed primitives, configs, evaluation composition, and the registry
- `rewardhack_gym.envs`: code and math environment families
- `rewardhack_gym.runners`: prompting, selection, filtering, and RL-oriented wrappers
- `rewardhack_gym.analysis`: aggregate metrics and experiment summaries
- `rewardhack_gym.io`: JSONL and optional Parquet export

## Research Orientation

The library is designed for:

- post-training evaluation under imperfect reward
- best-of-N and rejection-filter studies
- verifier robustness experiments
- async RL reward analysis
- false-pass mining
- trajectory export for mechanistic interpretability

## Development

This repository uses `uv` as the primary dependency and environment manager.

```bash
uv sync
uv run pytest
```

See [docs/architecture.md](docs/architecture.md), [docs/adding-environment.md](docs/adding-environment.md), and [docs/trace-export.md](docs/trace-export.md) for the package design and extension workflow.

