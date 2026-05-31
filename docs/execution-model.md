# Execution Model

RewardHack-Gym now exposes code execution through an explicit backend interface. The old in-process runner is still available as `LocalTrustedBackend` for trusted research compatibility, but production integrations should use `SubprocessBackend`, `DockerBackend`, or a future `PrimeSandboxBackend`.

## Trust Model

- `LocalTrustedBackend` executes submitted code in the current Python process and is trusted-local-only.
- `SubprocessBackend` runs submitted function cases in a child process with timeouts, process-tree kill on timeout, stdout/stderr limits, output-object limits, blocked imports, blocked filesystem builtins, and an isolated temporary working directory.
- `DockerBackend` runs the same worker inside a per-run container with `--network none`, memory limits, and CPU limits when Docker is available.
- `PrimeSandboxBackend` is a placeholder for a future Prime-native execution backend.

## Current Behavior

The backend result schema is:

```python
@dataclass
class ExecutionResult:
    status: Literal[
        "passed",
        "failed",
        "syntax_error",
        "runtime_error",
        "timeout",
        "memory_limit",
        "sandbox_error",
    ]
    case_results: list[dict]
    stdout: str
    stderr: str
    duration_seconds: float
    backend: str
```

Use `run_function_cases` / `run_function_cases_sync` for backend-mediated execution. Built-in code environment checkers now send behavioral cases through the selected backend. `compile_submission` is limited to AST-level syntax and symbol-presence checks; it does not execute submissions.

## Near-Term Guidance

If you use RewardHack-Gym today:

- prefer `SubprocessBackend` for local adversarial testing
- prefer `DockerBackend` for stronger local isolation when Docker is available
- do not expose `LocalTrustedBackend` as a public service
- use a Prime-native sandbox once available for hosted multi-tenant execution

## Upgrade Path

The package architecture keeps task, verifier, and oracle abstractions separate from the execution backend, so Prime can replace the local backend with a hosted sandbox without changing task schemas.
