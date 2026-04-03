# Adding A New Environment Family

New environments should subclass `ResearchEnvironment` and preserve the package’s main design principle:

the shortest path to high official reward should differ from the shortest path to true correctness.

## Minimal Checklist

1. Define a `Task` generator with explicit hidden metadata.
2. Define an `ExploitSurface` that says what the official verifier misses.
3. Build an official `CompositeEvaluator` from small reusable checkers.
4. Build a stronger oracle `CompositeEvaluator`.
5. Provide `canonical_true_output` and `canonical_exploit_output`.
6. Register the environment with `register_environment(...)`.
7. Add tests proving:
   - canonical true outputs pass both official and oracle
   - canonical exploit outputs pass official but fail oracle

## Practical Pattern

Most environment implementations follow this structure:

- task sampler: constructs prompt, visible metadata, hidden metadata, and exploitability knobs
- official checker set: fast, scalable, and plausibly imperfect
- oracle checker set: broader semantics, stronger perturbations, or hidden constraints
- exploit classifier: maps false passes into canonical exploit families

## Exploitability Controls

Use `ExploitabilityProfile` and environment-specific task metadata to vary:

- public coverage
- parser strictness
- hidden adversarial strength
- process validation
- domain awareness
- perturbation robustness

The goal is to support exploitability curves, not just anecdotal examples.

