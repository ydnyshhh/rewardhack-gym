# Execution Model

RewardHack-Gym currently supports lightweight local execution for coding environments through a trusted research runner.

## Trust Model

- code execution is intended for trusted local research use
- the current runner is not a hardened security sandbox
- environment authors should assume submitted code may be adversarial unless run in a stronger external sandbox

## Current Behavior

The runtime intentionally keeps the harness small and deterministic:

- parses source with Python `ast`
- executes submissions in a restricted namespace
- exposes only a curated set of builtins and helper modules
- records the trust model in checker diagnostics

This improves ergonomics for local experiments, but it is not sufficient as a secure multi-tenant execution boundary.

## Near-Term Guidance

If you use RewardHack-Gym today:

- run coding environments only on trusted machines
- do not expose the current execution harness as a public service
- prefer offline or isolated worker execution for larger experiments

## Upgrade Path

The long-term production path is to move code execution behind a stronger runner model, for example:

- process-level sandboxing
- containerized or VM-backed execution
- per-run filesystem and network isolation
- explicit resource and syscall controls

The package architecture is set up so those execution upgrades can be introduced without changing the task / verifier / oracle abstractions.
