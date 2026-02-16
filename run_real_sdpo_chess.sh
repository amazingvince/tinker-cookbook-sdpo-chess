#!/usr/bin/env bash
set -euo pipefail

IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

require_cmd uv

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is required. Put it in .env or export it in your shell." >&2
  exit 1
fi

# Run settings (override via env vars).
: "${MODEL_NAME:=Qwen/Qwen3-30B-A3B}"
: "${GROUPS_PER_BATCH:=32}"
: "${GROUP_SIZE:=4}"
: "${TRAIN_STEPS:=200}"
: "${DATASET_N:=-1}"
: "${DATASET_SEED:=0}"
: "${BUFFER_SIZE:=5000}"
: "${BUFFER_SOURCE_POOL_SIZE:=20000}"
: "${DATASET_NUM_BATCHES:=${TRAIN_STEPS}}"
: "${DATASET_SAMPLE_WITH_REPLACEMENT:=true}"
: "${DATASET_REFRESH_ROWS_PER_BATCH:=128}"
: "${MAX_EXAMPLES:=${BUFFER_SOURCE_POOL_SIZE}}"
: "${MAX_TOKENS:=512}"
: "${TEMPERATURE:=1.0}"
: "${LEARNING_RATE:=1e-5}"
: "${LORA_RANK:=32}"
: "${NUM_SUBSTEPS:=1}"
: "${SAVE_EVERY:=10}"
: "${TTL_SECONDS:=604800}"
: "${RUN_TESTS:=1}"
: "${INSTALL_SYZYGY:=1}"
: "${SYZYGY_PIECES:=345}"
: "${SYZYGY_PARALLEL_DOWNLOADS:=8}"
: "${SYZYGY_DIR:=$HOME/.local/share/syzygy/standard}"
: "${STOCKFISH_INSTALL_ROOT:=$HOME/.local/stockfish}"
: "${STOCKFISH_SYMLINK_PATH:=$HOME/.local/bin/stockfish}"
: "${STOCKFISH_OVERWRITE:=false}"
: "${STOCKFISH_CACHE_DIR:=$HOME/.cache/tinker-cookbook/stockfish-cache}"
: "${RESERVE_CPU_FRACTION:=0.0}"
: "${RESERVE_CPU_MIN:=0}"
: "${MAX_WORKERS:=180}"
: "${PREFERRED_THREADS_PER_WORKER:=1}"
: "${HASH_BUDGET_FRACTION:=0.35}"
: "${HINT_STOCKFISH_DEPTH:=14}"
: "${HINT_STOCKFISH_MULTIPV:=5}"
: "${VERIFY_STOCKFISH_DEPTH:=20}"
: "${VERIFY_STOCKFISH_MULTIPV:=1}"
: "${GAME_STOCKFISH_DEPTH:=20}"
: "${GAME_STOCKFISH_MULTIPV:=8}"
: "${STOCKFISH_WDL_MODEL:=sf}"
: "${STOCKFISH_SYZYGY_MAX_PIECES:=5}"
: "${STOCKFISH_SHARED_EVAL_MODE:=two_pass}"
: "${MAX_CONCURRENT_TEACHER_LOGPROBS:=64}"
: "${MAX_CONCURRENT_GENERATION:=32}"
: "${MAX_CONCURRENT_SCORING:=32}"
: "${STUDENT_MAX_THINKING_TOKENS:=256}"
: "${DEBUG_EXAMPLES_EVERY_N_STEPS:=10}"
: "${DEBUG_EXAMPLES_PER_STEP:=2}"
: "${DEBUG_EXAMPLES_MAX_TEXT_CHARS:=4000}"
: "${PUZZLES_FRACTION:=0.5}"
: "${PUZZLE_SOLVER_MOVES_ONLY:=true}"
: "${MAX_PUZZLE_SOLVER_MOVES_PER_PUZZLE:=-1}"
: "${GAME_POSITIONS_PER_GAME:=3}"
: "${MIN_PUZZLE_RATING:=1200}"
: "${MIN_GAME_ELO:=1600}"
: "${MIN_GAME_PLY:=4}"
: "${MAX_GAME_PLY:=100}"
: "${MAX_SCAN_ROWS_PER_SOURCE:=200000}"
: "${OVERSAMPLE_FACTOR:=4}"
: "${SHUFFLE_BUFFER_SIZE:=10000}"
: "${GAME_ANSWER_MODE:=stockfish}"
: "${USE_STOCKFISH_GAME_REWARD:=true}"
: "${WANDB_PROJECT:=sdpo-chess}"
: "${WANDB_NAME_PREFIX:=sdpo-chess-real}"
: "${LOG_ROOT:=$SCRIPT_DIR/runs/sdpo_chess}"
: "${BEHAVIOR_IF_LOG_DIR_EXISTS:=resume}"

mkdir -p "$LOG_ROOT"
mkdir -p "$STOCKFISH_CACHE_DIR"

model_slug="$(printf '%s' "$MODEL_NAME" | tr '/:' '__' | tr -cd '[:alnum:]_.-')"
run_ts="$(date +%Y%m%d_%H%M%S)"
run_name="${WANDB_NAME_PREFIX}_${model_slug}_${run_ts}"
: "${LOG_PATH:=$LOG_ROOT/$run_name}"

echo "==> Syncing Python dependencies"
uv sync --extra verifiers --extra chess --extra wandb

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "==> Running test suite"
  uv run python -m pytest -q
fi

echo "==> Installing or reusing Stockfish"
stockfish_install_output="$(
  uv run python -m tinker_cookbook.recipes.verifiers_rl.install_stockfish \
    "install_root=${STOCKFISH_INSTALL_ROOT}" \
    "symlink_path=${STOCKFISH_SYMLINK_PATH}" \
    "overwrite=${STOCKFISH_OVERWRITE}"
)"
echo "$stockfish_install_output"

detected_stockfish_path="$(printf '%s\n' "$stockfish_install_output" | awk -F= '/stockfish_path=/{print $2}' | tail -n1 | xargs || true)"
if [[ -n "${STOCKFISH_PATH:-}" ]]; then
  if [[ -x "$STOCKFISH_PATH" ]]; then
    detected_stockfish_path="$STOCKFISH_PATH"
  else
    echo "Warning: ignoring invalid STOCKFISH_PATH from environment: ${STOCKFISH_PATH}" >&2
  fi
fi
if [[ -z "$detected_stockfish_path" && -x "$STOCKFISH_SYMLINK_PATH" ]]; then
  detected_stockfish_path="$STOCKFISH_SYMLINK_PATH"
fi
if [[ -z "$detected_stockfish_path" ]]; then
  echo "Could not determine Stockfish path." >&2
  exit 1
fi
STOCKFISH_PATH="$detected_stockfish_path"
if [[ ! -x "$STOCKFISH_PATH" ]]; then
  echo "Resolved STOCKFISH_PATH is not executable: ${STOCKFISH_PATH}" >&2
  exit 1
fi

if [[ "$INSTALL_SYZYGY" == "1" ]]; then
  echo "==> Installing or updating Syzygy tablebases (pieces=${SYZYGY_PIECES})"
  uv run python -m tinker_cookbook.recipes.verifiers_rl.install_syzygy \
    "output_dir=${SYZYGY_DIR}" \
    "pieces=${SYZYGY_PIECES}" \
    "parallel_downloads=${SYZYGY_PARALLEL_DOWNLOADS}" \
    "skip_existing=true" \
    "continue_on_error=true"
fi

if [[ -d "$SYZYGY_DIR" ]]; then
  has_syzygy_files="$(find "$SYZYGY_DIR" -type f \( -name '*.rtbw' -o -name '*.rtbz' \) -print -quit || true)"
  if [[ -z "$has_syzygy_files" ]]; then
    SYZYGY_DIR=""
  fi
else
  SYZYGY_DIR=""
fi

echo "==> Autotuning Stockfish worker configuration"
autotune_json="$(
  uv run python -m tinker_cookbook.recipes.verifiers_rl.stockfish_autotune \
    "output_format=json" \
    "reserve_cpu_fraction=${RESERVE_CPU_FRACTION}" \
    "reserve_cpu_min=${RESERVE_CPU_MIN}" \
    "max_workers=${MAX_WORKERS}" \
    "preferred_threads_per_worker=${PREFERRED_THREADS_PER_WORKER}" \
    "hash_budget_fraction=${HASH_BUDGET_FRACTION}"
)"
echo "$autotune_json"

export AUTOTUNE_JSON="$autotune_json"
# Global IFS omits spaces, so force a local space delimiter for this parse.
IFS=' ' read -r SF_NUM_WORKERS SF_THREADS SF_HASH_MB SF_ANALYSIS_SEC SF_VERIFY_RATE <<<"$(
  uv run python - <<'PY'
import json
import os
cfg = json.loads(os.environ["AUTOTUNE_JSON"])
print(
    cfg["stockfish_num_workers"],
    cfg["stockfish_threads"],
    cfg["stockfish_hash_mb"],
    cfg["stockfish_analysis_time_limit_sec"],
    cfg["stockfish_verification_sample_rate"],
)
PY
)"
unset AUTOTUNE_JSON

export MAX_EXAMPLES
export PUZZLES_FRACTION
export PUZZLE_SOLVER_MOVES_ONLY
export MAX_PUZZLE_SOLVER_MOVES_PER_PUZZLE
export GAME_POSITIONS_PER_GAME
export MIN_PUZZLE_RATING
export MIN_GAME_ELO
export MIN_GAME_PLY
export MAX_GAME_PLY
export MAX_SCAN_ROWS_PER_SOURCE
export OVERSAMPLE_FACTOR
export SHUFFLE_BUFFER_SIZE
export GAME_ANSWER_MODE
export USE_STOCKFISH_GAME_REWARD
export STOCKFISH_PATH
export GAME_STOCKFISH_DEPTH
export GAME_STOCKFISH_MULTIPV
export SF_THREADS
export SF_HASH_MB
export STOCKFISH_WDL_MODEL
export SYZYGY_DIR
export STOCKFISH_SYZYGY_MAX_PIECES
export STOCKFISH_CACHE_DIR
export SF_NUM_WORKERS
export VERIFY_STOCKFISH_MULTIPV

vf_env_args="$(
  uv run python - <<'PY'
import json
import os

def env_int(name: str) -> int:
    return int(os.environ[name])

def env_float(name: str) -> float:
    return float(os.environ[name])

def env_bool(name: str) -> bool:
    return os.environ[name].strip().lower() in {"1", "true", "yes", "on"}

syzygy_path = os.environ.get("SYZYGY_DIR", "").strip() or None

cfg = {
    "max_examples": env_int("MAX_EXAMPLES"),
    "puzzles_fraction": env_float("PUZZLES_FRACTION"),
    "puzzle_solver_moves_only": env_bool("PUZZLE_SOLVER_MOVES_ONLY"),
    "max_puzzle_solver_moves_per_puzzle": env_int("MAX_PUZZLE_SOLVER_MOVES_PER_PUZZLE"),
    "game_positions_per_game": env_int("GAME_POSITIONS_PER_GAME"),
    "min_puzzle_rating": env_int("MIN_PUZZLE_RATING"),
    "min_game_average_elo": env_int("MIN_GAME_ELO"),
    "min_game_ply": env_int("MIN_GAME_PLY"),
    "max_game_ply": env_int("MAX_GAME_PLY"),
    "max_scan_rows_per_source": env_int("MAX_SCAN_ROWS_PER_SOURCE"),
    "oversample_factor": env_int("OVERSAMPLE_FACTOR"),
    "shuffle_buffer_size": env_int("SHUFFLE_BUFFER_SIZE"),
    "game_answer_mode": os.environ["GAME_ANSWER_MODE"],
    "use_stockfish_game_reward": env_bool("USE_STOCKFISH_GAME_REWARD"),
    "stockfish_path": os.environ["STOCKFISH_PATH"],
    "stockfish_depth": env_int("GAME_STOCKFISH_DEPTH"),
    "stockfish_multipv": env_int("GAME_STOCKFISH_MULTIPV"),
    "stockfish_threads": env_int("SF_THREADS"),
    "stockfish_hash_mb": env_int("SF_HASH_MB"),
    "stockfish_wdl_model": os.environ["STOCKFISH_WDL_MODEL"],
    "stockfish_syzygy_path": syzygy_path,
    "stockfish_syzygy_max_pieces": env_int("STOCKFISH_SYZYGY_MAX_PIECES"),
    "stockfish_persistent_cache_dir": os.environ["STOCKFISH_CACHE_DIR"],
    "stockfish_num_workers": env_int("SF_NUM_WORKERS"),
    "stockfish_verification_multipv": env_int("VERIFY_STOCKFISH_MULTIPV"),
}

print(json.dumps(cfg, separators=(",", ":"), ensure_ascii=True))
PY
)"

echo "==> Launching SDPO chess run"
echo "run_name=${run_name}"
echo "log_path=${LOG_PATH}"
echo "model_name=${MODEL_NAME}"
echo "buffer_size=${BUFFER_SIZE} source_pool=${MAX_EXAMPLES} num_batches=${DATASET_NUM_BATCHES}"
echo "stockfish_path=${STOCKFISH_PATH}"
echo "stockfish_num_workers=${SF_NUM_WORKERS} stockfish_threads=${SF_THREADS} stockfish_hash_mb=${SF_HASH_MB}"
if [[ -n "$SYZYGY_DIR" ]]; then
  echo "syzygy_path=${SYZYGY_DIR}"
else
  echo "syzygy_path=<disabled>"
fi

train_cmd=(
  uv run python -m tinker_cookbook.recipes.verifiers_rl.sdpo_train
  "vf_env_id=hf-chess-mix"
  "vf_env_args=${vf_env_args}"
  "dataset_n=${DATASET_N}"
  "dataset_seed=${DATASET_SEED}"
  "dataset_buffer_size=${BUFFER_SIZE}"
  "dataset_num_batches=${DATASET_NUM_BATCHES}"
  "dataset_sample_with_replacement=${DATASET_SAMPLE_WITH_REPLACEMENT}"
  "dataset_refresh_rows_per_batch=${DATASET_REFRESH_ROWS_PER_BATCH}"
  "groups_per_batch=${GROUPS_PER_BATCH}"
  "group_size=${GROUP_SIZE}"
  "model_name=${MODEL_NAME}"
  "lora_rank=${LORA_RANK}"
  "num_substeps=${NUM_SUBSTEPS}"
  "learning_rate=${LEARNING_RATE}"
  "max_tokens=${MAX_TOKENS}"
  "temperature=${TEMPERATURE}"
  "max_concurrent_generation=${MAX_CONCURRENT_GENERATION}"
  "max_concurrent_scoring=${MAX_CONCURRENT_SCORING}"
  "save_every=${SAVE_EVERY}"
  "ttl_seconds=${TTL_SECONDS}"
  "log_path=${LOG_PATH}"
  "behavior_if_log_dir_exists=${BEHAVIOR_IF_LOG_DIR_EXISTS}"
  "wandb_project=${WANDB_PROJECT}"
  "wandb_name=${run_name}"
  "teacher_regularization=trust_region"
  "teacher_mix_alpha=0.05"
  "success_reward_threshold=0.5"
  "dont_reprompt_on_self_success=true"
  "include_environment_feedback=true"
  "environment_feedback_only_without_solution=true"
  "remove_thinking_from_demonstration=true"
  "enable_stockfish_hints=true"
  "stockfish_path=${STOCKFISH_PATH}"
  "stockfish_depth=${HINT_STOCKFISH_DEPTH}"
  "stockfish_multipv=${HINT_STOCKFISH_MULTIPV}"
  "stockfish_num_workers=${SF_NUM_WORKERS}"
  "stockfish_threads=${SF_THREADS}"
  "stockfish_hash_mb=${SF_HASH_MB}"
  "stockfish_wdl_model=${STOCKFISH_WDL_MODEL}"
  "stockfish_analysis_time_limit_sec=${SF_ANALYSIS_SEC}"
  "stockfish_syzygy_max_pieces=${STOCKFISH_SYZYGY_MAX_PIECES}"
  "stockfish_persistent_cache_dir=${STOCKFISH_CACHE_DIR}"
  "enable_stockfish_move_verification=true"
  "stockfish_verification_sample_rate=${SF_VERIFY_RATE}"
  "stockfish_verification_depth=${VERIFY_STOCKFISH_DEPTH}"
  "stockfish_verification_multipv=${VERIFY_STOCKFISH_MULTIPV}"
  "stockfish_shared_hint_and_verification_eval=true"
  "stockfish_shared_eval_mode=${STOCKFISH_SHARED_EVAL_MODE}"
  "include_stockfish_move_feedback=true"
  "stockfish_feedback_cp_loss_threshold=0.0"
  "max_reprompt_tokens=0"
  "strict_single_turn=true"
  "updates_per_batch=1"
  "loss_fn=importance_sampling"
  "student_max_thinking_tokens=${STUDENT_MAX_THINKING_TOKENS}"
  "max_concurrent_teacher_logprobs=${MAX_CONCURRENT_TEACHER_LOGPROBS}"
  "debug_examples_every_n_steps=${DEBUG_EXAMPLES_EVERY_N_STEPS}"
  "debug_examples_per_step=${DEBUG_EXAMPLES_PER_STEP}"
  "debug_examples_max_text_chars=${DEBUG_EXAMPLES_MAX_TEXT_CHARS}"
)

if [[ -n "${BASE_URL:-}" ]]; then
  train_cmd+=("base_url=${BASE_URL}")
fi
if [[ -n "${SYZYGY_DIR}" ]]; then
  train_cmd+=("stockfish_syzygy_path=${SYZYGY_DIR}")
fi

printf 'Command:\n%s\n' "${train_cmd[*]}"
"${train_cmd[@]}"

echo "==> Run finished"
echo "Logs: ${LOG_PATH}"
