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

3. Run SDPO training:

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
