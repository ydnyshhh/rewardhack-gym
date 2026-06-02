# RewardHack Eval Experiments

Thin CLI scripts in this directory run reusable experiment code from `rewardhack_gym.experiments`.

```bash
python experiments/rewardhack_eval/run_best_of_n.py \
  --config experiments/rewardhack_eval/configs/best_of_n.yaml \
  --out runs/best_of_n_demo \
  --dry-run \
  --dummy-model-mode canonical_exploit
```

The runners write a stable run directory with public-safe task/candidate records, aggregate metrics, examples, plots, and `report.md`. Local sweeps generate completions before scoring; official and oracle scores are evaluation signals after generation. Best-of-N defaults to `official_only`. Oracle scores are monitor-only unless a config explicitly requests an oracle diagnostic selection policy such as `oracle_upper_bound`, which is an upper-bound diagnostic rather than a deployment policy.

Output files redact API-key values and request headers, omit hidden/oracle/canonical task state, and store each candidate's sampling config plus model id/model path provenance when available.

Prime/OpenAI-compatible model sweep:

```bash
set PRIME_API_KEY=...
python experiments/rewardhack_eval/run_model_sweep.py \
  --config experiments/rewardhack_eval/configs/prime_model_sweep.yaml \
  --out runs/prime_model_sweep_001
```

Hosted Prime/Verifiers rollout analysis:

```bash
python experiments/rewardhack_eval/analyze_prime_rollouts.py \
  --input prime_rollouts.jsonl \
  --out runs/prime_eval_analysis_001 \
  --overwrite
```

`prime_inference` defaults to `https://api.pinference.ai/api/v1` and reads `PRIME_API_KEY`. Set `PRIME_TEAM_ID` when your Prime team routing requires `X-Prime-Team-ID`.
