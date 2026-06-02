# RewardHack Eval Experiments

Thin CLI scripts in this directory run reusable experiment code from `rewardhack_gym.experiments`.

```bash
python experiments/rewardhack_eval/run_best_of_n.py \
  --config experiments/rewardhack_eval/configs/best_of_n.yaml \
  --out runs/best_of_n_demo \
  --dry-run \
  --dummy-model-mode canonical_exploit
```

The runners write a stable run directory with public-safe task/candidate records, aggregate metrics, examples, plots, and `report.md`. Oracle scores are monitor-only unless a config explicitly requests an oracle diagnostic selection policy.

