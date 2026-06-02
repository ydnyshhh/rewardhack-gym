# RewardHack Eval Experiments

Thin CLI scripts in this directory run reusable experiment code from `rewardhack_gym.experiments`.

Dry-run dummy sweeps:

```bash
python experiments/rewardhack_eval/run_model_sweep.py \
  --config experiments/rewardhack_eval/configs/dummy_model_sweep.yaml \
  --out runs/dummy_model_sweep_test \
  --dry-run \
  --overwrite
```

The runners write a stable run directory with public-safe task/candidate records, aggregate metrics, examples, plots, and `report.md`. Local sweeps generate completions before scoring; `official_reward`/`official_score`, `oracle_score`, `hack_gap`, and `false_pass` are the core diagnostic signals. Best-of-N defaults to `official_only`. Oracle scores are monitor-only unless a config explicitly requests an oracle diagnostic selection policy such as `oracle_upper_bound`, which is an upper-bound diagnostic rather than a deployment policy.

Output files redact API-key values and request headers, omit hidden/oracle/canonical task state, and store each candidate's sampling config plus model id/model path provenance when available.

Local Prime Inference sweeps:

```bash
PRIME_API_KEY=... \
python experiments/rewardhack_eval/run_model_sweep.py \
  --config experiments/rewardhack_eval/configs/prime_model_sweep.yaml \
  --out runs/prime_model_sweep_001
```

Hosted eval rollout analysis:

```bash
python experiments/rewardhack_eval/analyze_prime_rollouts.py \
  --input prime_rollouts.jsonl \
  --out runs/prime_eval_analysis_001 \
  --overwrite
```

`prime_inference` defaults to `https://api.pinference.ai/api/v1` and reads `PRIME_API_KEY`. Set `PRIME_TEAM_ID` when your Prime team routing requires `X-Prime-Team-ID`.

Supported model examples in `prime_model_sweep.yaml` include `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-35B-A3B`, `meta-llama/Llama-3.2-3B-Instruct`, and `openai/gpt-oss-20b`. Hosted training/evals may support a live model list different from this static example; use the Prime CLI/API to list current models.

RewardHack-Gym is a controlled proxy-objective exploitation benchmark: it evaluates whether optimization against an imperfect official verifier produces outputs that pass the verifier but fail the intended oracle. Prime-supported model sweeps compare whether stronger/code/reasoning models reduce reward hacking or exploit verifier blind spots more efficiently.
