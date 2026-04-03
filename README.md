# rewardhack-gym

This repo came out of a question I had been sitting with for a while: if we want to understand reward hacking seriously, what is the right experimental substrate for it?

I did not want to study it only through one-off anecdotes or post hoc explanations of strange model behavior. I wanted something more controlled: environments where the tension between the objective you care about and the signal the model sees can be built in from the start, varied systematically, and analyzed later.

That led to the idea behind `rewardhack-gym`: a package of synthetic environments, starting with coding and math, designed to make these failures reproducible enough to study rather than just notice. The environments are meant to be useful on their own, but they are also meant to support a second layer of work I care about even more: understanding these behaviors through mechanistic interpretability once they appear.

RewardHack-Gym is a gym-style package for constructing and analyzing verifiable agent environments with controllable proxy-objective mismatch, designed for reward hacking, post-training, and mechanistic interpretability research. It is built as a research substrate, not as a static benchmark. Each environment centers the primitive:

`task + official verifier + oracle + exploit surface`

and makes it easy to study settings where optimization pressure drives models toward cheap proxy reward rather than true task completion.

## What the Package Covers

- Modular, typed core abstractions for tasks, verifiers, oracles, exploit surfaces, trajectories, and metrics
- Configurable exploitability profiles for dialing official-vs-oracle mismatch up or down
- Stronger first-release environment families for code and math
- Prompting, best-of-N, rejection-filter, and RL-style evaluation wrappers
- JSONL trace export for downstream analysis and future mechanistic interpretability pipelines
- Stable trajectory annotations for failure slicing, cohort export, and mechanistic interpretability prep
- A CLI designed to work naturally through `uv run`

## Initial Environment Families

### Code

- `code/spec-overfit`: implementation tasks where the official verifier emphasizes public examples from realistic pipeline-style state reconstruction tasks, while the oracle probes stale-update semantics, dominance invariants, and adversarial distribution shift
- `code/patch-verification`: patch tasks where the official verifier focuses on the reported regression, while the oracle checks semantic preservation, hidden workflow regressions, deletion semantics, tie handling, and side effects like input mutation
- `code/interface-conformance`: protocol tasks where visible examples pass but deeper state, aliasing, and contract guarantees fail
- `code/schema-preservation`: normalization tasks where the official verifier scores projected visible fields while the oracle checks schema retention, order stability, compatibility-field preservation, and detached nested outputs

### Math

- `math/symbolic-mismatch`: branch-sensitive symbolic-equivalence tasks where weak evaluation probes only safe regions while the oracle checks global equivalence
- `math/piecewise-reasoning`: generated piecewise case-analysis tasks where the official evaluator rewards the final solution set while the oracle checks whether the left and right branch reasoning is actually valid
- `math/reasoning-validity`: generated derivation-validity tasks spanning quadratic invariant checks and linear-system residual consistency, where the official evaluator can reward the final answer while the oracle checks whether the supporting work actually recomputes
- `math/constraint-sensitive`: structured branch-and-domain tasks where weak grading may treat the domain as cosmetic and only probe one branch, while the oracle checks maximal domain correctness, hidden branch behavior, and near-singularity perturbations

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
uv run rewardhack-gym inspect-env code/spec-overfit --profile low
uv run rewardhack-gym sample-task code/spec-overfit --seed 7
uv run rewardhack-gym sample-batch code/patch-verification --count 32 --output tasks.jsonl --include-canonicals
uv run rewardhack-gym evaluate-output code/spec-overfit --seed 7 --text-file outputs/spec_overfit.py
uv run rewardhack-gym evaluate-traces math/constraint-sensitive --input outputs.jsonl --output traces.jsonl --summary-output summary.json --false-pass-output false_passes.jsonl
uv run rewardhack-gym stats --input traces.jsonl
uv run rewardhack-gym export-false-passes --input traces.jsonl --output false_passes.jsonl
```

The environment CLI exposes exploitability controls directly:

- `--profile low|medium|high|adversarial`
- `--official-coverage`
- `--hidden-adversarial-strength`
- `--parser-strictness`
- `--process-validation-fraction`
- `--domain-awareness`
- `--perturbation-robustness`
- `--public-example-diversity`

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

Rich environments also emit stable trajectory annotations such as:

- `scenario_id`
- `canonical_exploit_class`
- `exploit_strategy`
- `official_passed_checks` / `oracle_failed_checks`
- `semantic_failures`

That annotation surface is now treated as part of the package design rather than incidental metadata.

The CLI now supports a more explicit batch workflow for that style of research:

- sample many tasks under a chosen exploitability profile
- evaluate an outputs file against those tasks
- write summary metrics plus true-pass / false-pass cohorts directly

## Development

This repository uses `uv` as the primary dependency and environment manager.

```bash
uv sync
uv run pytest
```

## Execution Model

The current code-execution harness is designed for trusted local research workflows. It is intentionally lightweight and is not a security sandbox. Environment execution should therefore be treated as `trusted-local-only` unless and until a stronger sandboxed runner is added.

See [docs/families.md](docs/families.md), [docs/architecture.md](docs/architecture.md), [docs/adding-environment.md](docs/adding-environment.md), [docs/trace-export.md](docs/trace-export.md), [docs/annotations.md](docs/annotations.md), and [docs/execution-model.md](docs/execution-model.md) for the package design, family guide, and extension workflow.
