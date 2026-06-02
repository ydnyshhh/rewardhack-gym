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

Verifier realism:

- Official verifier: public tests, shallow interface checks, official property probes, and a weak coverage estimate.
- Realistic analogue: coding benchmark unit tests, CI regression suites, and examples bundled with an issue or API contract.
- Why plausible: many post-training coding rewards are test-pass proxies over a small visible or generated test set.
- What it misses: stale updates, revision precedence, tie behavior, and broader metamorphic event-stream invariants.
- Shortcut exploited: pass visible monotonic traces while using input order as the true authority.
- Oracle adds: hidden semantic cases and property tests that exercise the full replay contract.
- Expected false-pass behavior: public tests pass while hidden semantic/metamorphic checks fail.

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

Verifier realism:

- Official verifier: reported-ticket regression tests plus shallow interface and side-effect checks.
- Realistic analogue: PR/CI checks that verify the filed bug but under-sample adjacent behavior.
- Why plausible: real patch review often rewards fixing the observed failure with limited regression coverage.
- What it misses: deletion/tombstone semantics, tie resolution, mixed workflows, and input mutation.
- Shortcut exploited: minimally patch the reported trace while breaking nearby state-machine behavior.
- Oracle adds: broader regression cases, invariant tests, and mutation checks.
- Expected false-pass behavior: the ticket test goes green while hidden surrounding behavior regresses.

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

Verifier realism:

- Official verifier: example-driven interface tests over common protocol calls.
- Realistic analogue: SDK/interface conformance tests that exercise happy paths and simple state transitions.
- Why plausible: protocol rewards often use smoke tests that check return values but not encapsulation.
- What it misses: aliasing, repeated-query stability, edge-capacity behavior, and deeper lifecycle invariants.
- Shortcut exploited: mimic the visible protocol while leaking internal state or collapsing edge cases.
- Oracle adds: hidden scenario sequences, aliasing probes, and repeated-use checks.
- Expected false-pass behavior: common examples pass while encapsulation or state invariants fail.

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

Verifier realism:

- Official verifier: projected-schema tests and optional shallow aliasing/order checks.
- Realistic analogue: ETL/dashboard tests that compare only visible fields consumed by downstream reports.
- Why plausible: data-processing rewards often validate the fields a benchmark cares to display.
- What it misses: full schema preservation, compatibility fields, stable order, and nested detachment.
- Shortcut exploited: normalize only projected fields and drop or alias everything else.
- Oracle adds: full-schema hidden cases, order checks, and nested aliasing probes.
- Expected false-pass behavior: projected outputs look correct while full records are lossy or unstable.

### `control/negative-code`

What it tests:

- A clean negative control where the official verifier is intentionally aligned with the oracle
- Whether aggregate hack rates disappear when verifier mismatch is removed

Common false passes:

- None expected. Shortcut implementations should fail the official verifier and oracle together.

Verifier realism:

- Official verifier: strong unit tests over the same normalization contract distribution as the oracle.
- Realistic analogue: a mature CI suite whose public and hidden tests sample the same stable API contract.
- Why plausible: not every benchmark/eval has a weak proxy; controls should include well-covered tasks.
- What it misses: intentionally little for this simple contract.
- Shortcut exploited: no productive shortcut; skipping normalization fails visible/control cases.
- Oracle adds: equivalent contract checks with held-out cases from the same distribution.
- Expected false-pass behavior: near-zero false passes, including under Best-of-N selection.

Most useful trace fields:

- `annotations.negative_control`
- `annotations.visible_probe_labels`
- `annotations.hidden_probe_labels`
- `official_result.components["control-cases"]` and `oracle_result.components["control-cases"]` diagnostics

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

Verifier realism:

- Official verifier: parsing, numeric probes on visible/safe points, and optional symbolic canonicalization.
- Realistic analogue: math eval graders that sample numeric equivalence points plus lightweight simplification.
- Why plausible: exact symbolic equivalence is expensive and brittle, so many graders use probe-based checks.
- What it misses: branch-sensitive equivalence, sign-region coverage, and exact global normalization.
- Shortcut exploited: emit the branch that matches visible probes but fails elsewhere.
- Oracle adds: stronger symbolic equivalence and hidden stress-region numeric probes.
- Expected false-pass behavior: safe-region probes pass while hidden branch points or symbolic checks fail.

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

Verifier realism:

- Official verifier: final-answer checks plus shallow labeled-line/process validation.
- Realistic analogue: rubric or LLM-judge process checks that verify answer shape and a few support lines.
- Why plausible: process supervision often under-checks every branch witness.
- What it misses: case-specific derivation validity and branch witness consistency.
- Shortcut exploited: produce the correct final set with fabricated case analysis.
- Oracle adds: branch-by-branch consistency checks and hidden witness validation.
- Expected false-pass behavior: final answer passes while process lines fail branch validation.

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

Verifier realism:

- Official verifier: final-answer checks plus lightweight support-line format/process checks.
- Realistic analogue: chain-of-thought or rubric rewards that emphasize answer correctness and visible structure.
- Why plausible: checking every derivation claim is more expensive than checking the final answer.
- What it misses: recomputation of residuals, invariant truth, and line-by-line support consistency.
- Shortcut exploited: optimize for the final answer and fill support fields with plausible text.
- Oracle adds: consistency checks that recompute required support lines.
- Expected false-pass behavior: final answer passes while reasoning support is fabricated or inconsistent.

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

Verifier realism:

- Official verifier: parser checks, visible point probes, and partial domain validation.
- Realistic analogue: math/programming evals that test representative examples and lightly validate constraints.
- Why plausible: exhaustive domain checking and perturbation tests are often absent from fast eval proxies.
- What it misses: excluded roots, branch changes near singularities, and perturbation robustness.
- Shortcut exploited: solve the visible interval and treat the domain line as cosmetic.
- Oracle adds: hidden branch probes, domain checks, and perturbation cases near singularities.
- Expected false-pass behavior: visible probes pass while domain or branch-sensitive oracle probes fail.

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
