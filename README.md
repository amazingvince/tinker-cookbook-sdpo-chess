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
  vf_env_id=your_chess_env \
  vf_env_args='{}' \
  model_name=Qwen/Qwen3-4B-Instruct-2507 \
  enable_stockfish_hints=true \
  stockfish_path=/path/to/stockfish \
  stockfish_syzygy_path=/path/to/syzygy \
  wandb_project=your_project
```

## Notes

- Teacher hint truncation is disabled by default: `max_reprompt_tokens=0`.
- Student thinking budget is configurable with `student_max_thinking_tokens`.
- Periodic qualitative debug examples can be enabled with:
  - `debug_examples_every_n_steps`
  - `debug_examples_per_step`
  - `debug_examples_file_name`

