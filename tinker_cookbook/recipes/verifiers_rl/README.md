# RL Training with Tinker + Environments Hub (Verifiers)

[Verifiers](https://github.com/primeintellect-ai/verifiers) is a library for creating RL environments for LLMs, including many community implementations featured on Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments). This recipe allows all text-based environments from the Environments Hub to be used with Tinker for RL training.

To use this recipe, you need to have your chosen environment module (a self-contained Python package) installed in your project. You can install environments from the Environments Hub using the `prime` CLI:

```bash
uv tool install prime # or pipx install prime
prime env install user/env-id # ex. prime env install primeintellect/reverse-text
```

Examples:
- [primeintellect/reverse-text](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text)
- [primeintellect/alphabet-sort](https://app.primeintellect.ai/dashboard/environments/primeintellect/alphabet-sort)
- [primeintellect/math-python](https://app.primeintellect.ai/dashboard/environments/primeintellect/math-python)
- [will/wordle](https://app.primeintellect.ai/dashboard/environments/will/wordle)

You can then run the recipe with the following command, where `vf_env_id` is the ID (just `env-id`) of the environment, and `vf_env_args` is an optional JSON string of arguments to pass when loading the environment.

```bash
python -m tinker_cookbook.recipes.verifiers_rl.train vf_env_id=env-id vf_env_args='{}' ...
```

The reverse-text example as configured should climb from ~0.2 to ~0.35 in 32 steps.

For a detailed chess-focused SDPO guide with W&B setup and SDPO-repo hyperparameter parity mappings, see:

- [`README_SDPO_TINKER_CHESS.md`](./README_SDPO_TINKER_CHESS.md)

## SDPO Training Recipe

This folder also includes an SDPO recipe for token-level, on-policy SDPO with verifiers environments:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.sdpo_train \
  vf_env_id=env-id \
  vf_env_args='{}' \
  model_name=Qwen/Qwen3-4B-Instruct-2507 \
  groups_per_batch=32 \
  group_size=8
```

The SDPO recipe:
- runs in on-policy mode only with one optimizer update per freshly sampled batch (`updates_per_batch=1`);
- builds teacher reprompts from successful peer solutions and/or environment feedback;
- can optionally add Stockfish 18 chess hints (WDL expected-score and threat summaries) to teacher reprompts when FEN is present in state/prompt;
- supports teacher regularization via `trust_region` (fixed reference sampler) or `ema` (EMA distribution over recent on-policy samplers);
- supports optional full-logit distillation with top-k + tail approximation (`full_logit_distillation`, `distillation_topk`, `distillation_add_tail`);
- supports configurable Tinker RL loss settings (`loss_fn`, `loss_fn_config_json`) within the on-policy update;
- fails fast on multi-turn traces by default (`strict_single_turn=True`).
- supports optional SDPO+GRPO mixing (`grpo_mix_lambda`) and `token` vs `sequence` advantage modes.
- supports reprompt truncation behavior (`reprompt_truncation=right|left|error`).

Expected SDPO metrics in `metrics.jsonl`:
- `sdpo/success_group_fraction`
- `sdpo/success_sample_fraction`
- `sdpo/feedback_available_fraction`
- `sdpo/feedback_used_fraction`
- `sdpo/reprompt_sample_fraction`
- `sdpo/mean_advantage`
- `sdpo/mean_abs_advantage`
- `sdpo/num_zero_adv_samples`
- `sdpo/num_skipped_samples`
- `sdpo/grpo_mix_lambda`
- `sdpo/full_logit_distillation`
- `sdpo/topk_overlap_fraction`
- `sdpo/updates_per_batch`
- `sdpo/stockfish_hints_enabled`
- `sdpo/stockfish_move_verification_enabled`
- `sdpo/stockfish_verification_sample_rate`
- `sdpo/stockfish_verification_depth`
- `sdpo/stockfish_verification_multipv`
- `sdpo/stockfish_hint_available_fraction`
- `sdpo/stockfish_hint_used_fraction`
- `sdpo/stockfish_verification_candidate_fraction`
- `sdpo/stockfish_verification_scheduled_fraction`
- `sdpo/stockfish_verified_fraction`
- `sdpo/stockfish_legal_move_fraction`
- `sdpo/stockfish_best_move_fraction`
- `sdpo/stockfish_feedback_fraction`
- `sdpo/stockfish_avg_cp_loss`
- `sdpo/stockfish_accuracy` (best-move accuracy over verified samples)
- `sdpo/stockfish_acpl` (alias of average centipawn loss)
- `chess/acc` (alias of `sdpo/stockfish_accuracy`)
- `chess/acpl` (alias of `sdpo/stockfish_acpl`)
- `sdpo/stockfish_estimated_cp_loss_fraction`

### Chess + Stockfish Hints

Use the built-in mixed Hugging Face chess environment (`hf-chess-mix`) to train on both
Lichess puzzles and sampled game positions:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.sdpo_train \
  vf_env_id=hf-chess-mix \
  vf_env_args='{"max_examples":20000,"puzzles_fraction":0.5,"game_positions_per_game":3,"puzzle_solver_moves_only":true,"game_answer_mode":"stockfish","use_stockfish_game_reward":true,"stockfish_path":"stockfish","stockfish_depth":20,"min_game_ply":4,"max_game_ply":100,"min_game_average_elo":1600}' \
  model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  groups_per_batch=32 \
  group_size=4
```

`hf-chess-mix` streams:
- puzzles from `Lichess/chess-puzzles` and expands each puzzle into per-move examples
  from the puzzle line, defaulting to solver-side moves only (skipping forced opponent replies);
- games from `Lichess/standard-chess-games` by sampling multiple random plies per game and
  creating `(FEN at ply, move)` examples where move labels can come from Stockfish.

Final training rows are sampled from the mixed puzzle/game candidate pools.

Key `hf-chess-mix` args in `vf_env_args`:
- `max_examples` (required workload size inside env);
- `puzzles_fraction` (0..1 mix ratio);
- `max_scan_rows_per_source`, `oversample_factor`, `shuffle_buffer_size` (streaming/sample controls);
- `puzzle_solver_moves_only` (default `true`) and `max_puzzle_solver_moves_per_puzzle` (default `-1`, all selected puzzle moves);
- `game_positions_per_game` (default `3`, set `-1` to include all candidate plies);
- `game_answer_mode`: `stockfish` (default) or `pgn`;
- `use_stockfish_game_reward` (default `true`): for game positions, reward is legality + Stockfish quality scaling;
- `game_reward_legal_floor`, `game_reward_best_move_bonus`, `game_reward_expected_score_temperature`, `game_reward_cp_loss_scale`;
- `stockfish_syzygy_path`, `stockfish_syzygy_max_pieces` for exact endgame outcome shaping;
- `game_reward_syzygy_wdl_scale`, `game_reward_syzygy_dtz_scale` for Syzygy delta penalties;
- `game_reward_pv_overlap_bonus`, `game_reward_pv_motif_plies` for PV motif-overlap reward;
- `game_reward_use_confidence_weighting`, `game_reward_confidence_neutral`,
  `game_reward_confidence_nodes_reference`, `game_reward_confidence_seldepth_factor`
  for search-stability weighting;
- `stockfish_path`, `stockfish_depth`, `stockfish_multipv`, `stockfish_threads`, `stockfish_hash_mb`, `stockfish_num_workers`;
- `min_puzzle_rating`, `min_game_average_elo`, `min_game_ply`, `max_game_ply` (quality/position filters).

SDPO Stockfish efficiency knobs:
- `stockfish_shared_hint_and_verification_eval` (default `true`): use one shared Stockfish request path when both hint generation and move verification are active for a sample;
- `stockfish_shared_eval_mode`:
  - `two_pass` (default): wide hint pass (`stockfish_depth`, `stockfish_multipv`) plus deep scoring pass (`stockfish_verification_depth`, `stockfish_verification_multipv`);
  - `single`: one deep pass shared by both hint rendering and scoring;
- `stockfish_verification_multipv` (default `1`): deep-pass width for verification (set >1 only when you want broader deep candidate context).
- `stockfish_persistent_cache_dir` (default: `log_path/stockfish_cache` in SDPO recipe): persistent on-disk cache reused across resumed/rerun jobs.

Game reward details for `source=lichess_game`:
- Illegal or unparseable move: `reward = 0`.
- Base quality:
  - `q = exp(-delta_E / T)` when Stockfish WDL expected scores are available;
  - fallback `q = exp(-cp_loss / S)` when expected scores are unavailable.
- Syzygy shaping (when tablebase data is available):
  - `q *= exp(-wdl_penalty / syzygy_wdl_scale)`
  - `q *= exp(-dtz_penalty / syzygy_dtz_scale)` for same-WDL lines.
- PV motif shaping:
  - `q += pv_overlap_bonus * overlap(best_pv_tail, predicted_pv_tail)`.
- Confidence weighting (optional):
  - compute confidence from `depth`, `seldepth`, and `nodes`;
  - blend toward neutral quality: `q <- neutral + confidence * (q - neutral)`.
- Final reward:
  - `reward = legal_floor + (1-legal_floor) * q + best_move_bonus_if_exact`.

For chess tasks where the prompt contains a FEN, enable Stockfish-driven hints (using WDL expected score, not centipawns):

```bash
python -m tinker_cookbook.recipes.verifiers_rl.sdpo_train \
  vf_env_id=your-chess-env \
  vf_env_args='{}' \
  model_name=Qwen/Qwen3-4B-Instruct-2507 \
  enable_stockfish_hints=true \
  stockfish_path=/path/to/stockfish \
  stockfish_depth=14 \
  stockfish_multipv=5 \
  stockfish_num_workers=16 \
  stockfish_threads=2 \
  student_max_thinking_tokens=2000 \
  stockfish_verification_depth=20 \
  stockfish_verification_sample_rate=1.0 \
  stockfish_analysis_time_limit_sec=0.2 \
  stockfish_engine_max_retries=1 \
  include_stockfish_move_feedback=true \
  stockfish_feedback_cp_loss_threshold=20 \
  stockfish_include_fen_decode=true \
  stockfish_include_ascii_board=true \
  stockfish_include_search_stats=true \
  stockfish_syzygy_path=/path/to/syzygy \
  stockfish_wdl_model=sf
```

If enabled, teacher reprompts can include:
- root WDL expected score (`E = p(win) + 0.5 * p(draw)`);
- root search stats (depth/seldepth/nodes/nps/tbhits when available);
- optional Syzygy tablebase root outcome/DTZ for <= `stockfish_syzygy_max_pieces` (default `5`, aligned with `3-4-5` tablebases);
- top candidate moves with `delta_E` vs the best line;
- threat summaries (hanging pieces, threatened pieces, checking opportunities);
- FEN-decoded board context (material, king squares, pieces under pressure, weak king-zone squares);
- "moves likely to be bad" explanations with refutation context when available.
- dedicated "Trap analysis (future-state refutations)" section that classifies bad moves by severity and motifs
  (material drop, king-safety tactic, tactical-capture sequence, etc.) using PV lookahead;
- detailed Stockfish move-verification feedback including:
  - model predicted move vs Stockfish best move at depth 20;
  - cp-loss with source (`centipawn`, `wdl_scaled`, or fallback penalty if score is unavailable);
  - best and predicted PV lines for concrete guidance.
- teacher prompt context is unbounded by default (`max_reprompt_tokens=0`) so long hints are retained.

`student_max_thinking_tokens` limits tokens inside `<think>...</think>` blocks during student rollout
generation (set `0` to disable).
`student_forced_answer_tokens` reserves a small continuation budget to force a final answer pass
when a rollout remains in thinking mode.
When thinking is enabled, set `max_tokens` comfortably above `student_max_thinking_tokens`
(for example, `max_tokens >= student_max_thinking_tokens + 32`).

Default teacher templates treat solution/feedback/hints as private context and explicitly instruct
the teacher to answer as if independently derived, without referencing those auxiliary signals.

Helper utilities for chess runtime setup:

- `python -m tinker_cookbook.recipes.verifiers_rl.stockfish_autotune`
- `python -m tinker_cookbook.recipes.verifiers_rl.install_stockfish`
- `python -m tinker_cookbook.recipes.verifiers_rl.install_syzygy`

To log periodic qualitative debug samples (prompt, model output, expected answer, Stockfish hint):

```bash
python -m tinker_cookbook.recipes.verifiers_rl.sdpo_train \
  ... \
  debug_examples_every_n_steps=10 \
  debug_examples_per_step=3 \
  debug_examples_max_text_chars=4000 \
  debug_examples_file_name=sdpo_debug_examples.jsonl
```

This writes JSONL records to `<log_path>/sdpo_debug_examples.jsonl` (or your configured file name),
adds `sdpo/debug_examples_logged` to `metrics.jsonl`, and also emits long-form text snapshots into
`<log_path>/long_text.jsonl`. With W&B enabled, snapshots are mirrored to:

- `sdpo/debug_examples/latest` (latest batch text dump),
- `sdpo/debug_examples/table` (accumulated rollout/debug text table).

To create a starter JSONL dataset of random FENs from Lichess puzzles + games:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.chess_dataset \
  output_path=/tmp/chess_positions.jsonl \
  num_positions=2000 \
  puzzles_fraction=0.6 \
  include_stockfish_hints=true \
  stockfish_path=/path/to/stockfish
```

You can also evaluate offline:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.evaluate vf_env_id=env-id vf_env_args='{}' ...
```

This recipe also includes a standalone `AsyncOpenAI`-compatible client implemented with Tinker, which can be adapted for other applications.

**Potential footgun:**
- Some Environments Hub implementations involve users writing their own `<think>` parsers (e.g. for use with reasoning RL starting on Instruct models). Despite being Instruct models, the Qwen3 models/tokenizers all use the same tokenizer chat template, which will strip any observed `<think>` sections automatically (which may be inadvertently penalized by reward functions). Users should either modify the renderer, tokenizer chat template, or environment module if observing issues with thinking sections from Qwen3 models.
