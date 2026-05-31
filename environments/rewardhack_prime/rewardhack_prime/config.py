from __future__ import annotations

import verifiers as vf

SUPPORTED_PROFILES = ("aligned", "low", "medium", "high", "adversarial")
SUPPORTED_CODE_EXECUTION_BACKENDS = (
    "local",
    "local_trusted",
    "trusted",
    "subprocess",
    "docker",
    "prime",
    "prime_sandbox",
)


class RewardHackTasksetConfig(vf.TasksetConfig):
    family: str = "code/spec-overfit"
    profile: str = "medium"
    split: str = "eval"
    num_tasks: int = 100
    seed: int = 0
    reward_mode: str = "official_only"
    reward_penalty_lambda: float = 1.0
    include_oracle_metrics: bool = True
    expose_canonical_outputs: bool = False
    code_execution_backend: str = "subprocess"
    code_execution_timeout_seconds: float = 2.0
    code_execution_memory_mb: int = 256
    code_execution_stdout_limit_chars: int = 20_000
    code_execution_stderr_limit_chars: int = 20_000
    code_execution_max_output_object_size: int = 20_000
    prime_sandbox_image: str = "python:3.12-slim"
    prime_sandbox_timeout_minutes: int = 10
    prime_sandbox_cpu_cores: int = 1
