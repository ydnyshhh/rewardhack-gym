# Architecture

RewardHack-Gym is organized around one central abstraction:

`task + official verifier + oracle + exploit surface`

## Core Objects

- `Task`: the intended objective, prompt, interface contract, metadata, and hidden metadata
- `ExploitSurface`: a first-class description of what the official verifier misses and which exploit families are likely
- `CompositeEvaluator`: a weighted collection of reusable checker components
- `Trajectory`: a structured record of prompt, final output, official result, oracle result, exploit labels, and runtime metadata
- `Trajectory.annotations`: stable failure metadata for scenario ids, exploit classes, passed and failed checks, and family-specific semantic failure labels
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

For richer environment families, annotations add a second layer of stable, analysis-oriented metadata on top of raw checker diagnostics. This is the intended bridge from evaluator results into later mechanistic interpretability pipelines.

## Environment Families

### Code

- `code/spec-overfit`: public examples plus shallow interface checks vs hidden semantic and metamorphic checks
- `code/patch-verification`: ticket-focused regression tests vs semantic-preservation and hidden regression checks
- `code/interface-conformance`: visible protocol examples vs aliasing and state-contract validation
- `code/schema-preservation`: projected-schema grading vs hidden order, compatibility-field, and nested-detachment validation

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

See [docs/annotations.md](annotations.md) for the documented annotation schema.
