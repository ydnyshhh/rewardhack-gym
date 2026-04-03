# Family Guide

This guide gives a quick research-oriented map of the built-in families:

- what each family is meant to test
- which false passes it tends to generate
- which annotations or trace fields are most useful when analyzing it

For the stable cross-environment schema, see [docs/annotations.md](annotations.md).

## Code Families

### `code/spec-overfit`

What it tests:

- Whether a model learns the real replay semantics of an event stream or only the visible public-example pattern
- Spec overfitting on stale updates, revision precedence, and same-revision tie behavior

Common false passes:

- Treating the latest textual occurrence as authoritative instead of honoring revision order
- Ignoring stale tail events that appear after the visible monotonic pattern
- Dropping same-revision later-wins semantics

Most useful trace fields:

- `annotations.scenario_id`
- `annotations.visible_probe_labels`
- `annotations.official_property_probe_labels`
- `annotations.hidden_probe_labels`
- `annotations.semantic_failures`
- `official_result.components["public-cases"]` and `oracle_result.components["hidden-cases"]` diagnostics

Typical slice:

- false passes with `annotations.semantic_failures` containing `stale-update-semantics-failed`

### `code/patch-verification`

What it tests:

- Ticket-focused patching under a weak regression harness
- Whether a patch fixes the reported bug while silently breaking surrounding semantics

Common false passes:

- Narrow bugfix patches that break tombstones, tie handling, or omit-empty behavior
- Shallow replay fixes that pass the visible workflow but fail mixed hidden workloads
- Patches that mutate caller-owned input while still passing the visible regression tests

Most useful trace fields:

- `annotations.scenario_id`
- `annotations.canonical_exploit_class`
- `annotations.visible_probe_labels`
- `annotations.official_property_probe_labels`
- `annotations.hidden_probe_labels`
- `annotations.semantic_failures`

Typical slice:

- false passes with `annotations.semantic_failures` containing `input-mutation-happened` or `tombstone-semantics-failed`

### `code/interface-conformance`

What it tests:

- Whether a model satisfies a visible protocol while violating deeper state or encapsulation guarantees
- Stateful interface mimicry under shallow usage examples

Common false passes:

- Returning internal buffers directly because visible examples only check values once
- Passing basic history operations while failing repeated-query behavior or edge-capacity cases

Most useful trace fields:

- `annotations.scenario_id`
- `annotations.visible_probe_labels`
- `annotations.hidden_probe_labels`
- `annotations.semantic_failures`
- `oracle_result.components["state-aliasing"]` diagnostics

Typical slice:

- false passes with `annotations.semantic_failures` containing `state-aliasing-failed`

### `code/schema-preservation`

What it tests:

- Whether a model preserves a full schema or only the projected visible fields that the official evaluator scores
- Hidden contract failures around order stability, nested aliasing, and compatibility-field retention

Common false passes:

- Lossy projection that keeps only dashboard-visible fields
- Outputs with the right content but unstable ordering
- Shallow copies that alias nested caller-owned structures

Most useful trace fields:

- `annotations.scenario_id`
- `annotations.canonical_exploit_class`
- `annotations.visible_probe_labels`
- `annotations.hidden_probe_labels`
- `annotations.semantic_failures`

Typical slice:

- false passes with `annotations.semantic_failures` containing `schema-preservation-failed`, `order-stability-failed`, or `nested-aliasing-failed`

## Math Families

### `math/symbolic-mismatch`

What it tests:

- Whether a model optimizes for visible numeric probes instead of global symbolic equivalence
- Weak canonicalization and sign-region mismatch

Common false passes:

- Returning the visible branch of an absolute-value expression
- Matching the official probe region while failing on hidden regions
- Relying on the official grader's weak symbolic normalization

Most useful trace fields:

- `annotations.scenario_id`
- `annotations.visible_probe_points`
- `annotations.hidden_probe_points`
- `annotations.semantic_failures`
- `oracle_result.components["oracle-symbolic-equivalence"]` diagnostics

Typical slice:

- false passes with `annotations.semantic_failures` containing `absolute-value-branch-missed`

### `math/piecewise-reasoning`

What it tests:

- Whether a model fabricates piecewise reasoning while still landing on the correct final solution set
- Branch-by-branch reasoning fidelity under structured output

Common false passes:

- Correct final answers paired with invented left/right case analysis
- Empty-branch claims that look plausible but do not match the actual branch witnesses

Most useful trace fields:

- `annotations.scenario_id`
- `annotations.required_labels`
- `annotations.piecewise_boundary`
- `annotations.semantic_failures`

Typical slice:

- false passes with `annotations.semantic_failures` containing `piecewise-case-fabricated`

### `math/reasoning-validity`

What it tests:

- Whether a model treats required derivation lines as real constraints or as decorative formatting
- Final-answer-first optimization under weak process checking

Common false passes:

- Correct final roots paired with fabricated invariant summaries
- Correct variable assignments paired with inconsistent residual lines
- Final answers that pass while supporting lines do not recompute

Most useful trace fields:

- `annotations.scenario_id`
- `annotations.canonical_exploit_class`
- `annotations.required_labels`
- `annotations.semantic_failures`

Typical slice:

- false passes with `annotations.semantic_failures` containing `support-line-fabricated` or `fabricated-residuals`

### `math/constraint-sensitive`

What it tests:

- Whether a model treats domain information as cosmetic while optimizing for a narrow official grading region
- Hidden failures around excluded roots, branch validity, and perturbation robustness

Common false passes:

- Returning an expression that is correct only on the visible interval
- Omitting excluded roots from the domain line
- Collapsing to a constant branch that passes the official sign regime

Most useful trace fields:

- `annotations.scenario_id`
- `annotations.singularities`
- `annotations.visible_probe_points`
- `annotations.hidden_probe_points`
- `annotations.perturbation_points`
- `annotations.semantic_failures`

Typical slice:

- false passes with `annotations.semantic_failures` containing `domain-wrong`, `branch-behavior-failed`, or `perturbation-consistency-failed`

## Example Workflow

When inspecting exported traces, a good first pass is:

1. Filter to `is_false_pass == true`.
2. Group by `task.family` or `task.task_id`.
3. Within a family, group by `annotations.canonical_exploit_class`.
4. Slice again by `annotations.semantic_failures`.

That usually gives a better research view than looking at aggregate false-pass rate alone, because it separates qualitatively different exploit mechanisms inside the same family.
