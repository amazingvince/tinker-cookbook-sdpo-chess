# Chess SDPO on Tinker

This repository is a slimmed-down codebase focused on training chess next-move models with SDPO on top of Tinker.

## What Is Included

- SDPO trainer and utilities:
  - `tinker_cookbook/sdpo/train.py`
  - `tinker_cookbook/sdpo/utils.py`
  - `tinker_cookbook/sdpo/chess_hints.py`
- Verifiers RL recipe entrypoint:
  - `tinker_cookbook/recipes/verifiers_rl/sdpo_train.py`
- Chess data tooling:
  - `tinker_cookbook/recipes/verifiers_rl/chess_dataset.py`
  - `tinker_cookbook/recipes/verifiers_rl/hf_chess_mix.py` (verifiers env for mixed HF games+puzzles)
- Stockfish setup + tuning helpers:
  - `tinker_cookbook/recipes/verifiers_rl/stockfish_autotune.py`
  - `tinker_cookbook/recipes/verifiers_rl/install_stockfish.py`
  - `tinker_cookbook/recipes/verifiers_rl/install_syzygy.py`
- Recipe docs:
  - `tinker_cookbook/recipes/verifiers_rl/README.md`
  - `tinker_cookbook/recipes/verifiers_rl/README_SDPO_TINKER_CHESS.md`

Most unrelated cookbook recipes/docs have been removed to keep this repo focused.

## Quick Start

1. Set `TINKER_API_KEY`.
2. Install deps:

```bash
uv sync
```

3. Run an end-to-end real training job (recommended):

```bash
./run_real_sdpo_chess.sh
```

The script will:
- sync dependencies;
- optionally run tests;
- install/reuse Stockfish;
- optionally install Syzygy 3-5 tablebases;
- autotune Stockfish workers/threads/hash;
- launch SDPO training on `hf-chess-mix` with on-policy updates (`updates_per_batch=1`).

Common overrides:

```bash
MODEL_NAME=Qwen/Qwen3-30B-A3B TRAIN_STEPS=300 WANDB_PROJECT=your_project ./run_real_sdpo_chess.sh
```

```bash
RUN_TESTS=0 INSTALL_SYZYGY=0 ./run_real_sdpo_chess.sh
```

Rolling-buffer overrides (few-thousand active positions with refresh):

```bash
BUFFER_SIZE=5000 BUFFER_SOURCE_POOL_SIZE=20000 DATASET_NUM_BATCHES=400 DATASET_REFRESH_ROWS_PER_BATCH=128 ./run_real_sdpo_chess.sh
```

180-CPU node overrides (use all CPUs for Stockfish workers):

```bash
RESERVE_CPU_FRACTION=0 RESERVE_CPU_MIN=0 MAX_WORKERS=180 PREFERRED_THREADS_PER_WORKER=1 ./run_real_sdpo_chess.sh
```

4. Run SDPO training manually:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.sdpo_train \
  vf_env_id=hf-chess-mix \
  vf_env_args='{"max_examples":20000,"puzzles_fraction":0.5,"puzzle_solver_moves_only":true,"game_positions_per_game":3,"game_answer_mode":"stockfish","use_stockfish_game_reward":true,"stockfish_path":"stockfish","stockfish_depth":20,"stockfish_syzygy_path":"/path/to/syzygy","stockfish_syzygy_max_pieces":5,"game_reward_pv_overlap_bonus":0.05,"game_reward_use_confidence_weighting":true,"min_game_ply":4,"max_game_ply":100,"min_game_average_elo":1600}' \
  model_name=Qwen/Qwen3-4B-Instruct-2507 \
  enable_stockfish_hints=true \
  stockfish_path=/path/to/stockfish \
  stockfish_syzygy_path=/path/to/syzygy \
  wandb_project=your_project
```

## Notes

- Teacher hint truncation is disabled by default: `max_reprompt_tokens=0`.
- Student thinking budget is configurable with `student_max_thinking_tokens`.
- Stockfish analyses/verifications are persisted by default under `log_path/stockfish_cache`
  (override with `stockfish_persistent_cache_dir`) so resumed/rerun jobs can reuse prior compute.
- Periodic qualitative debug examples can be enabled with:
  - `debug_examples_every_n_steps`
  - `debug_examples_per_step`
  - `debug_examples_file_name`
