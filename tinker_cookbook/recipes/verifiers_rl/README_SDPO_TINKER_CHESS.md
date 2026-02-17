# SDPO in Tinker for Chess (with W&B and SDPO-Repo Hyperparameter Parity)

This guide documents a practical end-to-end workflow for training a chess model with SDPO in `tinker-cookbook`, using:

- verifiers-style RL environments,
- Stockfish 18 hints + move verification,
- Weights & Biases logging,
- and hyperparameter settings mapped from the official SDPO repository.

The training entrypoint is:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.sdpo_train ...
```

## 1. What this implementation does

The SDPO path in this repo is implemented in:

- `/Users/vincent/Documents/SDPO/tinker-cookbook/tinker_cookbook/sdpo/train.py`
- `/Users/vincent/Documents/SDPO/tinker-cookbook/tinker_cookbook/sdpo/utils.py`
- `/Users/vincent/Documents/SDPO/tinker-cookbook/tinker_cookbook/sdpo/chess_hints.py`
- `/Users/vincent/Documents/SDPO/tinker-cookbook/tinker_cookbook/recipes/verifiers_rl/sdpo_train.py`

Core behavior:

- Token-level SDPO advantages from teacher-vs-rollout logprobs.
- On-policy training only (`updates_per_batch` must be `1`).
- Teacher reprompting with successful peer solutions and optional environment feedback.
- Optional teacher regularization (`trust_region`, `ema`, `none`).
- Optional full-logit distillation (`distillation_topk` + tail).
- Optional Stockfish hints and depth-20 verification feedback.
- Optional Syzygy probing in endgames when configured.

## 2. Prerequisites

Install project dependencies:

```bash
cd /Users/vincent/Documents/SDPO/tinker-cookbook
uv sync --extra verifiers --extra chess --extra wandb
```

Install Stockfish 18 and ensure it is on `PATH`, or provide absolute path via `stockfish_path`.

For Syzygy endgame probing, download tablebases and set `stockfish_syzygy_path`.

### Bootstrap helpers (new)

Auto-tune Stockfish flags from detected CPU/RAM:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.stockfish_autotune output_format=json
```

Install the best Stockfish release asset for detected CPU instructions:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.install_stockfish
```

Install Syzygy endgame tables (default `3-4-5`, about 0.9 GB):

```bash
python -m tinker_cookbook.recipes.verifiers_rl.install_syzygy
```

Optional larger installs:

- `pieces=6` (about 149 GB)
- `pieces=all` (includes 7-man, about 17 TB; usually impractical)

## 3. W&B setup

Authenticate once:

```bash
wandb login
```

Recommended environment variables:

```bash
export WANDB_PROJECT=sdpo-chess
export WANDB_ENTITY=<your_wandb_team_or_user>
export WANDB_MODE=online
```

Pass these through training args:

```bash
wandb_project=$WANDB_PROJECT wandb_name=sdpo-chess-run-01
```

Notes:

- Metrics are also written to local `metrics.jsonl` under `log_path`.
- Resume is automatic from the latest checkpoint in `log_path`.

## 4. SDPO repo hyperparameter parity

This section maps the main SDPO knobs from `lasgroup/SDPO` to Tinker flags.

Primary SDPO sources used:

- `verl/trainer/config/sdpo.yaml`
- `experiments/rich_feedback/run_sdpo.sh`
- `experiments/generalization/run_sdpo_all.sh`

### Mapping table

| SDPO repo key | Tinker flag | Typical value(s) |
|---|---|---|
| `data.train_batch_size` | `groups_per_batch` | `32` |
| `actor_rollout_ref.rollout.n` | `group_size` | `8` |
| `actor_rollout_ref.actor.optim.lr` | `learning_rate` | `1e-6` or `1e-5` |
| `algorithm.rollout_correction.rollout_is=token` | `advantage_mode` | `token` |
| `actor...self_distillation.success_reward_threshold` | `success_reward_threshold` | `0.5` |
| `actor...self_distillation.dont_reprompt_on_self_success` | `dont_reprompt_on_self_success` | `true` |
| `actor...self_distillation.include_environment_feedback` | `include_environment_feedback` | `true` or `false` |
| `actor...self_distillation.environment_feedback_only_without_solution` | `environment_feedback_only_without_solution` | `true` |
| `actor...self_distillation.remove_thinking_from_demonstration` | `remove_thinking_from_demonstration` | `true` |
| `actor...self_distillation.max_reprompt_len` | `max_reprompt_tokens` | `0` (disable truncation) |
| `actor...self_distillation.reprompt_truncation` | `reprompt_truncation` | `right` |
| `actor...self_distillation.full_logit_distillation` | `full_logit_distillation` | `true` |
| `actor...self_distillation.distillation_topk` | `distillation_topk` | `20` or `100` |
| `actor...self_distillation.alpha` | `teacher_mix_alpha` | `1.0` or `0.5` |
| `actor...self_distillation.teacher_regularization` | `teacher_regularization` | `ema` |
| `actor...self_distillation.teacher_update_rate` / `ema_update_rate` | `teacher_mix_alpha` (for EMA mixture in this implementation) | `0.01` or `0.05` |

Important:

- In Tinker SDPO, `teacher_mix_alpha` is used for teacher-mixture weighting.
- In the SDPO repo, `alpha` is KL interpolation and `teacher_update_rate` / `ema_update_rate` controls EMA update.
- They are related but not identical; use the presets below as practical parity settings.
- `max_reprompt_tokens=0` disables truncation so long teacher hints are preserved.
- `student_max_thinking_tokens` bounds `<think>...</think>` token count during rollout (`2000` default, `0` disables).
- `student_forced_answer_tokens` reserves a short continuation pass so outputs that stay in thinking mode
  still get a final answer attempt (`32` default).
- Keep `max_tokens` above `student_max_thinking_tokens + student_forced_answer_tokens`.
- For Stockfish efficiency, default shared mode is `stockfish_shared_hint_and_verification_eval=true` with
  `stockfish_shared_eval_mode=two_pass` (wide hints + deep scoring), and
  `stockfish_verification_multipv=1` for fast deep truth labels (raise only if you need broad deep candidates).
- `stockfish_persistent_cache_dir` enables cross-run reuse of Stockfish analyses/verifications.

## 5. Ready-to-run parity commands

### Dataset wiring (HF puzzles + games mix)

Use the built-in `hf-chess-mix` environment so training samples positions from both:
- `Lichess/chess-puzzles` (expanded to per-move puzzle examples with solver-side moves by default),
- `Lichess/standard-chess-games` (multiple random plies sampled per game, with Stockfish-labeled moves by default).

Example `vf_env_args` payload:

```json
{
  "max_examples": 20000,
  "puzzles_fraction": 0.5,
  "puzzle_solver_moves_only": true,
  "max_puzzle_solver_moves_per_puzzle": -1,
  "game_positions_per_game": 3,
  "game_answer_mode": "stockfish",
  "use_stockfish_game_reward": true,
  "stockfish_path": "stockfish",
  "stockfish_depth": 20,
  "stockfish_syzygy_path": "/path/to/syzygy",
  "stockfish_syzygy_max_pieces": 5,
  "stockfish_persistent_cache_dir": "/path/to/shared-stockfish-cache",
  "stockfish_shared_hint_and_verification_eval": true,
  "stockfish_shared_eval_mode": "two_pass",
  "stockfish_verification_multipv": 1,
  "game_reward_syzygy_wdl_scale": 1.0,
  "game_reward_syzygy_dtz_scale": 20.0,
  "game_reward_pv_overlap_bonus": 0.05,
  "game_reward_pv_motif_plies": 6,
  "game_reward_use_confidence_weighting": true,
  "game_reward_confidence_neutral": 0.5,
  "game_reward_confidence_nodes_reference": 500000,
  "game_reward_confidence_seldepth_factor": 1.5,
  "max_scan_rows_per_source": 200000,
  "oversample_factor": 4,
  "shuffle_buffer_size": 10000,
  "min_puzzle_rating": 1200,
  "min_game_average_elo": 1600,
  "min_game_ply": 4,
  "max_game_ply": 100
}
```

### Game reward model (Stockfish-labeled game positions)

For `source=lichess_game`, reward uses legality plus smooth distance to Stockfish best:

1. Parse predicted move from model output.
2. If illegal/unparsed: reward `= 0`.
3. Compute base quality:
   - `q = exp(-(best_expected_score - predicted_expected_score) / game_reward_expected_score_temperature)`
   - fallback `q = exp(-cp_loss / game_reward_cp_loss_scale)` if expected-score comparison is unavailable.
4. Apply Syzygy penalties (when TB probes are available):
   - `q *= exp(-wdl_penalty / game_reward_syzygy_wdl_scale)`
   - `q *= exp(-dtz_penalty / game_reward_syzygy_dtz_scale)`.
5. Add PV motif overlap bonus:
   - `q += game_reward_pv_overlap_bonus * overlap(best_pv_tail, predicted_pv_tail)`.
6. Optionally confidence-weight `q` using Stockfish search stability (depth/seldepth/nodes):
   - `q <- neutral + confidence * (q - neutral)`.
7. Map to final reward:
   - `reward = game_reward_legal_floor + (1 - game_reward_legal_floor) * q`
   - add `game_reward_best_move_bonus` when predicted move equals Stockfish best move.

### Trap-analysis hints

When Stockfish hints are enabled, the rendered teacher hint now includes a
`Trap analysis (future-state refutations)` section for high-`delta_E` moves:

- severity labels (`critical`, `major`, `moderate`, `minor`);
- short refutation tail from PV lookahead;
- motif tags (material drop, king-safety tactic, tactical-capture sequence, unsound checking idea, etc.).

For `source=lichess_puzzle`, reward remains exact match to the selected puzzle-line target move.

### A) Rich-feedback-style SDPO parity (from `run_sdpo.sh`)

```bash
cd /Users/vincent/Documents/SDPO/tinker-cookbook

python -m tinker_cookbook.recipes.verifiers_rl.sdpo_train \
  vf_env_id=hf-chess-mix \
  vf_env_args='{"max_examples":20000,"puzzles_fraction":0.5,"puzzle_solver_moves_only":true,"game_positions_per_game":3,"game_answer_mode":"stockfish","use_stockfish_game_reward":true,"stockfish_path":"stockfish","stockfish_depth":20,"stockfish_syzygy_path":"/path/to/syzygy","stockfish_syzygy_max_pieces":5,"game_reward_pv_overlap_bonus":0.05,"game_reward_use_confidence_weighting":true,"min_game_ply":4,"max_game_ply":100,"min_game_average_elo":1600}' \
  model_name=Qwen/Qwen3-8B \
  groups_per_batch=32 \
  group_size=8 \
  learning_rate=1e-6 \
  advantage_mode=token \
  teacher_regularization=ema \
  teacher_mix_alpha=0.01 \
  full_logit_distillation=true \
  distillation_topk=20 \
  distillation_add_tail=true \
  success_reward_threshold=0.5 \
  dont_reprompt_on_self_success=true \
  include_environment_feedback=true \
  environment_feedback_only_without_solution=true \
  remove_thinking_from_demonstration=true \
  student_max_thinking_tokens=2000 \
  student_forced_answer_tokens=32 \
  max_reprompt_tokens=0 \
  reprompt_truncation=right \
  updates_per_batch=1 \
  wandb_project=$WANDB_PROJECT \
  wandb_name=sdpo-rich-feedback-parity \
  log_path=/tmp/tinker-examples/sdpo-chess-rich-parity
```

### B) Generalization-style SDPO parity (from `run_sdpo_all.sh`)

```bash
cd /Users/vincent/Documents/SDPO/tinker-cookbook

python -m tinker_cookbook.recipes.verifiers_rl.sdpo_train \
  vf_env_id=hf-chess-mix \
  vf_env_args='{"max_examples":20000,"puzzles_fraction":0.5,"puzzle_solver_moves_only":true,"game_positions_per_game":3,"game_answer_mode":"stockfish","use_stockfish_game_reward":true,"stockfish_path":"stockfish","stockfish_depth":20,"stockfish_syzygy_path":"/path/to/syzygy","stockfish_syzygy_max_pieces":5,"game_reward_pv_overlap_bonus":0.05,"game_reward_use_confidence_weighting":true,"min_game_ply":4,"max_game_ply":100,"min_game_average_elo":1600}' \
  model_name=Qwen/Qwen3-8B \
  groups_per_batch=32 \
  group_size=8 \
  learning_rate=1e-5 \
  advantage_mode=token \
  teacher_regularization=ema \
  teacher_mix_alpha=0.05 \
  full_logit_distillation=true \
  distillation_topk=100 \
  distillation_add_tail=true \
  success_reward_threshold=0.5 \
  dont_reprompt_on_self_success=true \
  include_environment_feedback=false \
  environment_feedback_only_without_solution=true \
  remove_thinking_from_demonstration=true \
  student_max_thinking_tokens=2000 \
  student_forced_answer_tokens=32 \
  max_reprompt_tokens=0 \
  reprompt_truncation=right \
  updates_per_batch=1 \
  wandb_project=$WANDB_PROJECT \
  wandb_name=sdpo-generalization-parity \
  log_path=/tmp/tinker-examples/sdpo-chess-generalization-parity
```

## 6. Chess-specific hinting + verification

Enable Stockfish hints and feedback:

```bash
enable_stockfish_hints=true \
stockfish_path=/path/to/stockfish \
stockfish_depth=14 \
stockfish_multipv=5 \
stockfish_verification_depth=20 \
stockfish_verification_sample_rate=1.0 \
student_max_thinking_tokens=2000 \
student_forced_answer_tokens=32 \
stockfish_analysis_time_limit_sec=0.2 \
stockfish_engine_max_retries=1 \
include_stockfish_move_feedback=true \
stockfish_feedback_cp_loss_threshold=20 \
stockfish_include_fen_decode=true \
stockfish_include_ascii_board=true \
stockfish_include_search_stats=true \
stockfish_syzygy_path=/path/to/syzygy \
stockfish_syzygy_max_pieces=5
```

Relevant metrics:

- `sdpo/stockfish_hint_available_fraction`
- `sdpo/stockfish_hint_used_fraction`
- `sdpo/stockfish_verification_candidate_fraction`
- `sdpo/stockfish_verification_scheduled_fraction`
- `sdpo/stockfish_verified_fraction`
- `sdpo/stockfish_legal_move_fraction`
- `sdpo/stockfish_best_move_fraction`
- `sdpo/stockfish_feedback_fraction`
- `sdpo/stockfish_avg_cp_loss`
- `sdpo/stockfish_accuracy` (best-move accuracy on verified samples)
- `sdpo/stockfish_acpl` (average centipawn loss alias)
- `chess/acc`
- `chess/acpl`
- `sdpo/stockfish_estimated_cp_loss_fraction`

## 7. Qualitative debug examples

To inspect model behavior over time, enable periodic example dumps:

```bash
debug_examples_every_n_steps=10 \
debug_examples_per_step=3 \
debug_examples_max_text_chars=4000 \
debug_examples_file_name=sdpo_debug_examples.jsonl
```

Each logged record includes:

- prompt (rendered from messages)
- model output
- expected answer (from environment state; falls back to Stockfish best move when available)
- Stockfish hint text
- reward and Stockfish verification fields (cp-loss, best/predicted move, legality)

Output file:

- `<log_path>/sdpo_debug_examples.jsonl`
- `<log_path>/long_text.jsonl` (human-readable snapshots; mirrored to W&B as `sdpo/debug_examples/latest` and `sdpo/debug_examples/table`)

Metric:

- `sdpo/debug_examples_logged`

## 8. Suggested W&B dashboard panels

Create panels for:

- Learning signal: `sdpo/mean_advantage`, `sdpo/mean_abs_advantage`.
- Distillation coverage: `sdpo/reprompt_sample_fraction`, `sdpo/feedback_used_fraction`.
- Outcome quality: environment reward metric + `sdpo/success_sample_fraction`.
- Chess move quality: `sdpo/stockfish_avg_cp_loss`, `sdpo/stockfish_legal_move_fraction`.
- Data quality: `sdpo/num_skipped_samples`, `sdpo/num_zero_adv_samples`.

## 9. Reproducibility checklist

- Pin model ID and exact environment version.
- Keep `log_path` stable per run family for resume behavior.
- Log all CLI args in W&B config.
- Keep Stockfish binary version fixed (18).
- Keep Syzygy path/version fixed if enabled.
- Tune Stockfish throughput:
  - lower `stockfish_verification_sample_rate` (for example `0.25`) to verify only a deterministic subset of samples;
  - set `stockfish_analysis_time_limit_sec` (for example `0.1`–`0.3`) to cap slow positions;
  - keep cache limits high enough (`stockfish_max_*_cache_entries`) for your dataset reuse pattern.

### High-CPU node profile (example: 180 CPU / 720GB RAM)

Use many workers with low thread count per engine:

```bash
stockfish_num_workers=48 \
stockfish_threads=3 \
stockfish_hash_mb=2048 \
stockfish_analysis_time_limit_sec=0.15 \
stockfish_verification_sample_rate=0.25
```

Rationale:

- `num_workers * threads` near 120-160 usually gives better throughput than one giant-threaded engine.
- Keep 20-40 CPU cores for Python/env overhead and OS.
- Raise `stockfish_hash_mb` only if RAM headroom remains stable under full concurrency.

## 10. Common pitfalls

- `strict_single_turn=true` will fail on multi-turn traces. Flatten to one completion segment or set `strict_single_turn=false`.
- If `stockfish_estimated_cp_loss_fraction` is high, many cp-loss values came from WDL/fallback rather than raw cp deltas.
- Very long hint templates can dominate context and slow training; trim with `stockfish_hint_max_good_moves`, `stockfish_hint_max_bad_moves`, and `max_*_items`.
