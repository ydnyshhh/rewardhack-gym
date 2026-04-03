# Architecture

RewardHack-Gym is organized around one central abstraction:

`task + official verifier + oracle + exploit surface`

## Core Objects

- `Task`: the intended objective, prompt, interface contract, metadata, and hidden metadata
- `ExploitSurface`: a first-class description of what the official verifier misses and which exploit families are likely
- `CompositeEvaluator`: a weighted collection of reusable checker components
- `Trajectory`: a structured record of prompt, final output, official result, oracle result, exploit labels, and runtime metadata
- `MetricSummary`: aggregate statistics such as pass rates, verifier gap, false-pass rate, and exploit-family counts

Built-in environments are registered through an explicit bootstrap step in `rewardhack_gym.bootstrap`, and the registry calls that bootstrap lazily when environments are listed or created.

## Official Verifier vs Oracle

The official verifier is designed to be:

- scalable
- plausible for post-training pipelines
- intentionally imperfect

The oracle is designed to be:

- stronger than the official verifier
- still automatable
- closer to the intended task semantics

Both return structured component-level diagnostics so analyses can inspect not just a score but why a trajectory passed or failed.

## Environment Families

### Code

- `code/spec-overfit`: public examples plus shallow interface checks vs hidden semantic and metamorphic checks
- `code/patch-verification`: ticket-focused regression tests vs semantic-preservation and hidden regression checks
- `code/interface-conformance`: visible protocol examples vs aliasing and state-contract validation

### Math

- `math/symbolic-mismatch`: fixed-point numeric checks vs symbolic equivalence
- `math/reasoning-validity`: final-answer-first grading vs derivation consistency
- `math/constraint-sensitive`: permissive formal-root grading vs domain-aware validation

## Research Workflows

The package is built for:

- prompt-based evaluation
- best-of-N selection
- rejection filtering
- RL reward computation
- async rollout scoring
- JSONL export for downstream mechanistic interpretability
