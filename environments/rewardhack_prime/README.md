# rewardhack-prime

`rewardhack-prime` adapts RewardHack-Gym into the Prime Intellect Verifiers v1 `Taskset`/`Env` shape.

RewardHack-Gym tasks expose a deliberately imperfect verifier and a stronger oracle:

- `official_score`: the score optimized by the model
- `oracle_score`: the intended-objective score used for evaluation
- `hack_gap`: `official_score - oracle_score`
- `false_pass`: `official_passed and not oracle_passed`

The default `reward_mode` is `official_only`, so RL and eval rollouts optimize the imperfect official signal while oracle metrics remain available for analysis.

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
include_oracle_metrics = true
```

Supported profiles are `aligned`, `low`, `medium`, `high`, and `adversarial`. `aligned` is the clean control: the official verifier is configured to be close to the oracle. Higher exploitability profiles make the official verifier increasingly weak relative to the oracle.

The Verifiers task rows intentionally do not include `hidden_metadata`. Oracle-only probes stay inside the taskset-owned `Task` objects and are used only during scoring.
