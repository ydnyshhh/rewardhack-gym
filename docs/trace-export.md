# Trace Export

Each `Trajectory` can be serialized to JSON-compatible dictionaries and written to JSONL through `rewardhack_gym.io.write_jsonl`.

## Included Fields

- task metadata
- prompt
- final output
- official evaluator result and components
- oracle result and components
- exploit labels
- runtime metadata
- structured annotations

## Common Filters

Typical downstream filters are:

- `is_false_pass == true`
- `official_result.passed == true`
- `oracle_result.passed == false`
- exploit labels such as `sample-point-spoof` or `state-aliasing`

The CLI supports these workflows directly through batch evaluation and cohort export:

- `sample-batch` to generate many tasks under a shared exploitability profile
- `evaluate-traces --summary-output ... --false-pass-output ... --true-pass-output ...`

## Mechanistic Interpretability Readiness

The package is structured so future interp layers can attach:

- activation-aligned identifiers
- trace segment labels
- rollout and evaluator version ids
- model and policy metadata

Without changing the core environment API, the exported trajectories already provide stable hooks for later cross-linking.

## Annotation Schema

Annotations are now a documented part of the trace surface rather than an incidental extra field.

Stable cross-environment keys include:

- `scenario_id`
- `canonical_exploit_class`
- `exploit_strategy`
- `task_family_parameters`
- `official_passed_checks`
- `official_failed_checks`
- `oracle_passed_checks`
- `oracle_failed_checks`
- `semantic_failures`

Families may also attach more specific probe metadata such as visible probe labels, hidden probe points, perturbation points, or required line labels.

See [docs/annotations.md](annotations.md) for the stable schema and intended downstream usage.
