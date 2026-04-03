# Annotation Schema

RewardHack-Gym trajectories expose `annotations` as a first-class, stable bridge between environment execution and downstream analysis.

## Stable Cross-Environment Keys

These keys are intended to be present for rich environment families and stable enough for downstream tooling:

- `scenario_id`
- `canonical_exploit_class`
- `exploit_strategy`
- `task_family_parameters`
- `official_passed_checks`
- `official_failed_checks`
- `oracle_passed_checks`
- `oracle_failed_checks`
- `semantic_failures`
- `exploit_labels`

## Family-Specific Keys

Environment families can extend the schema with additional fields that preserve the same naming style.

Common examples include:

- `visible_probe_labels`
- `official_property_probe_labels`
- `hidden_probe_labels`
- `visible_probe_points`
- `hidden_probe_points`
- `perturbation_points`
- `singularities`
- `required_labels`

## Intended Use

The annotation schema is designed to support:

- false-pass mining by failure subtype
- cohorting by exploit class or sampled scenario
- evaluator diagnostics analysis
- later mechanistic interpretability workflows that need stable labels for cross-linking traces, activations, and failure modes

The design goal is that a downstream analysis pipeline should not need to reverse-engineer evaluator internals from raw checker output just to answer questions like:

- Which traces failed because of domain omissions?
- Which false passes depended on weak side-effect checks?
- Which scenarios exposed tie-semantics failures versus fabricated derivation lines?
