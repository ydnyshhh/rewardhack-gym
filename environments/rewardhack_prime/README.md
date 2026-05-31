# rewardhack-prime

`rewardhack-prime` adapts RewardHack-Gym into the Prime Intellect Verifiers v1 `Taskset`/`Env` shape.

RewardHack-Gym tasks expose a deliberately imperfect verifier and a stronger oracle:

- `official_score`: the score optimized by the model
- `oracle_score`: the intended-objective score used for evaluation
- `hack_gap`: `official_score - oracle_score`
- `false_pass`: `official_passed and not oracle_passed`

The default `reward_mode` is `official_only`, so RL and eval rollouts optimize the imperfect official signal while oracle metrics remain available for analysis.

Prime-facing task rows contain only the prompt, task identity fields, exploit surface, and redacted public metadata. Hidden cases, oracle probes, and canonical reference outputs stay in a `PrivateTaskStore` keyed by `task_id` and are used only by the scoring path. Set `expose_canonical_outputs = true` only for debugging workflows that intentionally publish canonical references.

The Verifiers score outputs are deliberately narrow:

- `official_reward`: weighted training reward
- `oracle_score`: zero-weight monitor
- `hack_gap`: zero-weight monitor
- `false_pass`: zero-weight monitor

Supported reward modes are `official_only`, `oracle_upper_bound`, `gap_penalized`, and `false_pass_penalized`. The two penalized modes are intended for mitigation experiments only.

The adapter uses native Verifiers v1 surfaces: `load_environment(...)` builds a `vf.Env`, `RewardHackTaskset` exposes train/eval examples via `rows()` and `eval_rows()`, and the four score outputs are `@vf.reward`-decorated taskset methods discoverable by Verifiers runtime scoring.

## Usage

```python
import verifiers as vf
from rewardhack_prime import RewardHackTasksetConfig, load_environment

env = load_environment(
    vf.EnvConfig(
        taskset=RewardHackTasksetConfig(
            family="code/spec-overfit",
            profile="medium",
            split="eval",
            num_tasks=100,
            seed=0,
        )
    )
)
```

For Prime configs:

```toml
[[env]]
id = "rewardhack-prime"

[env.taskset]
family = "code/spec-overfit"
profile = "medium"
split = "eval"
num_tasks = 100
seed = 0
reward_mode = "official_only"
reward_penalty_lambda = 1.0
include_oracle_metrics = true
expose_canonical_outputs = false
code_execution_backend = "prime_sandbox"
code_execution_timeout_seconds = 2.0
code_execution_memory_mb = 256
code_execution_stdout_limit_chars = 20000
code_execution_stderr_limit_chars = 20000
code_execution_max_output_object_size = 20000
prime_sandbox_image = "python:3.12-slim"
prime_sandbox_timeout_minutes = 10
prime_sandbox_cpu_cores = 1
```

Supported profiles are `aligned`, `low`, `medium`, `high`, and `adversarial`. `aligned` is the clean control: the official verifier is configured to be close to the oracle. Higher exploitability profiles make the official verifier increasingly weak relative to the oracle.

The Verifiers task rows intentionally do not include `hidden_metadata`, oracle-case metadata, or canonical references by default. Oracle-only probes stay inside the taskset-owned private store and are used only during scoring.

For local development, `code_execution_backend = "subprocess"` remains the default. For Prime-hosted multi-tenant runs, use `prime_sandbox`; the adapter depends on `prime-sandboxes`, creates sandboxes with outbound network disabled, uploads only the worker and per-run payload, deletes the payload file before the worker starts, applies command timeout/resource settings, and deletes the sandbox after each run. `prime_sandbox_timeout_minutes` controls sandbox lifetime; `code_execution_timeout_seconds` controls the submitted-code command timeout.
