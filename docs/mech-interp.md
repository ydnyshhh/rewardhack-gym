# Mech Interp Prep

RewardHack-Gym now includes a small mech-interp-prep surface for turning trajectory logs into compact, sliceable records and matched cohorts.

## Compact Export Rows

Use `build_mech_interp_record(...)` or `build_mech_interp_records(...)` to export records with:

- `environment_name`
- `family`
- `scenario_id`
- `exploit_class`
- `semantic_failures`
- `official_passed` / `official_score`
- `oracle_passed` / `oracle_score`
- `prompt`
- `final_output`
- `canonical_output_type`
- `scenario_cohort_id`
- `failure_slice_id`

This is intended to be the minimal analysis surface for later activation slicing.

## Stable IDs

Two deterministic cohort ids are exported by default:

- `scenario_cohort_id`: stable across traces from the same environment family, scenario, and exploitability profile
- `failure_slice_id`: stable across traces that share the same environment family, scenario, profile, exploit class, semantic failure labels, and outcome type

These ids are meant to make it easy to join later activation analyses back to exact reward-hacking slices without depending on fragile file naming or ad hoc notebook logic.

## Matched Pairs

Use `build_matched_pairs(...)` to construct:

- true-pass vs false-pass pairs
- first by exact sampled task when possible
- otherwise by environment family + scenario id + profile

Each pair record includes:

- `pair_id`
- `pair_group_id`
- `match_level`
- `true_pass`
- `false_pass`

This is useful for contrastive mechanistic workflows where you want nearly matched successful vs reward-hacked traces.

## CLI

Export compact rows:

```bash
uv run rewardhack-gym export-mech-interp --input traces.jsonl --output mech_interp_rows.jsonl
```

Build matched pairs:

```bash
uv run rewardhack-gym build-matched-pairs --input mech_interp_rows.jsonl --output matched_pairs.jsonl
```

Limit pairs per cohort if needed:

```bash
uv run rewardhack-gym build-matched-pairs --input mech_interp_rows.jsonl --output matched_pairs.jsonl --max-pairs-per-group 16
```
