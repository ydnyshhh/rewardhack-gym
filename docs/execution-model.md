# Execution Model

RewardHack-Gym exposes code execution through an explicit backend interface. The old in-process runner is still available as `LocalTrustedBackend` for trusted research compatibility, while production integrations can select `SubprocessBackend`, `DockerBackend`, or `PrimeSandboxBackend`.

## Trust Model

- `LocalTrustedBackend` executes submitted code in the current Python process and is trusted-local-only.
- `SubprocessBackend` runs submitted function cases in a child process with timeouts, process-tree kill on timeout, stdout/stderr limits, output-object limits, blocked imports, blocked filesystem builtins, and an isolated temporary working directory.
- `DockerBackend` runs the same worker inside a per-run container with `--network none`, memory limits, and CPU limits when Docker is available.
- `PrimeSandboxBackend` uses the optional `prime-sandboxes` SDK to create a disposable Prime sandbox with `network_access=False`, upload the RewardHack worker and payload, run the worker with a per-command timeout, and delete the sandbox afterward.

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

Execution limits are first-class `EnvironmentConfig` fields:

- `code_execution_backend`
- `code_execution_timeout_seconds`
- `code_execution_memory_mb`
- `code_execution_stdout_limit_chars`
- `code_execution_stderr_limit_chars`
- `code_execution_max_output_object_size`
- `prime_sandbox_image`
- `prime_sandbox_timeout_minutes`
- `prime_sandbox_cpu_cores`

## Near-Term Guidance

If you use RewardHack-Gym today:

- prefer `SubprocessBackend` for local adversarial testing
- prefer `DockerBackend` for stronger local isolation when Docker is available
- prefer `PrimeSandboxBackend` for Prime-hosted multi-tenant runs
- do not expose `LocalTrustedBackend` as a public service
- install `rewardhack-gym[prime-sandbox]` or `prime-sandboxes` directly before selecting `code_execution_backend="prime_sandbox"`

## Windows Test Note

The live adversarial infinite-loop timeout test is skipped on Windows. In this Codex/PowerShell environment, the virtualenv Python launcher can leave a base-interpreter child running after an interrupted `while True` probe, which makes local verification unsafe. The timeout watchdog is covered by a fake-process unit test on Windows, while the live adversarial worker test should run on Linux CI, Docker, or Prime sandboxes.

## Upgrade Path

The package architecture keeps task, verifier, and oracle abstractions separate from the execution backend, so Prime can replace the local backend with a hosted sandbox without changing task schemas.
