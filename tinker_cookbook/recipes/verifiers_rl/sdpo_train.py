from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Literal

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersRLDatasetBuilder
from tinker_cookbook.sdpo import train as sdpo_train


@chz.chz
class CLIConfig:
    # model configuration
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    base_url: str | None = None
    load_checkpoint_path: str | None = None

    # environment configuration
    vf_env_id: str = "reverse-text"
    vf_env_args: str | None = None  # JSON string
    dataset_n: int = -1
    dataset_seed: int | None = None
    dataset_buffer_size: int = -1
    dataset_num_batches: int = -1
    dataset_sample_with_replacement: bool = False
    dataset_refresh_rows_per_batch: int = 0

    # training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 32
    num_substeps: int = 1
    learning_rate: float = 1e-5
    max_tokens: int = 512
    temperature: float = 1.0
    max_concurrent_generation: int = -1
    max_concurrent_scoring: int = -1

    # logging/checkpointing
    save_every: int = 10
    ttl_seconds: int | None = 604800
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    debug_examples_every_n_steps: int = 0
    debug_examples_per_step: int = 2
    debug_examples_max_text_chars: int = 4000
    debug_examples_file_name: str = "sdpo_debug_examples.jsonl"
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # SDPO configuration
    success_reward_threshold: float = 0.5
    dont_reprompt_on_self_success: bool = True
    include_environment_feedback: bool = True
    environment_feedback_only_without_solution: bool = True
    remove_thinking_from_demonstration: bool = True
    teacher_regularization: Literal["trust_region", "ema", "none"] = "trust_region"
    teacher_mix_alpha: float = 0.05
    ema_teacher_history: int = 8
    full_logit_distillation: bool = False
    distillation_topk: int | None = None
    distillation_add_tail: bool = True
    reprompt_template: str = (
        "{prompt}{solution}{feedback}{hints}\n"
        "Use the additional context above only as private guidance.\n"
        "Now answer the original question in your own style.\n"
        "Do not mention, quote, or allude to hints, feedback, reference solutions, Stockfish, "
        "engines, or external tools.\n"
    )
    solution_template: str = (
        "\nTeacher-only reference solution (private context):\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\nTeacher-only feedback from an unsuccessful earlier attempt (private context):\n\n"
        "{feedback_raw}\n\n"
    )
    feedback_keys_csv: str = "feedback,error,errors,judge_feedback"
    enable_stockfish_hints: bool = False
    stockfish_path: str = "stockfish"
    stockfish_depth: int = 14
    stockfish_multipv: int = 5
    stockfish_num_workers: int = 1
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 128
    stockfish_wdl_model: str = "sf"
    stockfish_max_pv_plies: int = 6
    stockfish_hint_max_good_moves: int = 3
    stockfish_hint_max_bad_moves: int = 3
    stockfish_hint_bad_move_threshold: float = 0.05
    stockfish_include_fen_decode: bool = True
    stockfish_include_ascii_board: bool = True
    stockfish_include_search_stats: bool = True
    stockfish_analysis_time_limit_sec: float | None = None
    stockfish_engine_max_retries: int = 1
    stockfish_max_root_cache_entries: int = 8192
    stockfish_max_move_cache_entries: int = 32768
    stockfish_max_verification_cache_entries: int = 65536
    stockfish_max_piece_pressure_items: int = 8
    stockfish_max_weak_square_items: int = 8
    stockfish_syzygy_path: str | None = None
    stockfish_syzygy_max_pieces: int = 5
    stockfish_persistent_cache_dir: str | None = None
    stockfish_unknown_score_cp_loss: float = 80.0
    stockfish_hints_template: str = (
        "\nTeacher-only analysis notes (private, do not reference explicitly):\n\n"
        "{stockfish_hints}\n\n"
    )
    stockfish_hints_only_without_solution: bool = False
    enable_stockfish_move_verification: bool = True
    stockfish_verification_sample_rate: float = 1.0
    stockfish_verification_depth: int = 20
    stockfish_verification_multipv: int = 1
    stockfish_shared_hint_and_verification_eval: bool = True
    stockfish_shared_eval_mode: Literal["single", "two_pass"] = "two_pass"
    stockfish_illegal_move_cp_loss: float = 1000.0
    include_stockfish_move_feedback: bool = True
    stockfish_feedback_cp_loss_threshold: float = 0.0
    max_reprompt_tokens: int = 0
    reprompt_truncation: Literal["left", "right", "error"] = "right"
    strict_single_turn: bool = True
    max_concurrent_teacher_logprobs: int = 64
    student_max_thinking_tokens: int = 0
    grpo_mix_lambda: float = 0.0
    advantage_mode: Literal["token", "sequence"] = "token"
    updates_per_batch: int = 1
    loss_fn: Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"] = (
        "importance_sampling"
    )
    loss_fn_config_json: str | None = None


async def cli_main(cli_config: CLIConfig, env: Any | None):
    _ = env

    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"verifiers_rl_sdpo_{model_name_short}_gp{cli_config.groups_per_batch}_"
        f"gs{cli_config.group_size}_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_"
        f"{date_and_time}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/verifiers_rl_sdpo/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    env_args = json.loads(cli_config.vf_env_args) if cli_config.vf_env_args else {}
    loss_fn_config = (
        json.loads(cli_config.loss_fn_config_json) if cli_config.loss_fn_config_json else None
    )

    dataset_builder = VerifiersRLDatasetBuilder(
        vf_env_id=cli_config.vf_env_id,
        vf_env_args=env_args,
        groups_per_batch=cli_config.groups_per_batch,
        dataset_n=cli_config.dataset_n,
        dataset_seed=cli_config.dataset_seed,
        dataset_buffer_size=cli_config.dataset_buffer_size,
        dataset_num_batches=cli_config.dataset_num_batches,
        dataset_sample_with_replacement=cli_config.dataset_sample_with_replacement,
        dataset_refresh_rows_per_batch=cli_config.dataset_refresh_rows_per_batch,
    )

    cfg = sdpo_train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        log_path=log_path,
        lora_rank=cli_config.lora_rank,
        temperature=cli_config.temperature,
        num_substeps=cli_config.num_substeps,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        save_every=cli_config.save_every,
        ttl_seconds=cli_config.ttl_seconds,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        debug_examples_every_n_steps=cli_config.debug_examples_every_n_steps,
        debug_examples_per_step=cli_config.debug_examples_per_step,
        debug_examples_max_text_chars=cli_config.debug_examples_max_text_chars,
        debug_examples_file_name=cli_config.debug_examples_file_name,
        group_size=cli_config.group_size,
        max_concurrent_generation=cli_config.max_concurrent_generation,
        max_concurrent_scoring=cli_config.max_concurrent_scoring,
        success_reward_threshold=cli_config.success_reward_threshold,
        dont_reprompt_on_self_success=cli_config.dont_reprompt_on_self_success,
        include_environment_feedback=cli_config.include_environment_feedback,
        environment_feedback_only_without_solution=(
            cli_config.environment_feedback_only_without_solution
        ),
        remove_thinking_from_demonstration=cli_config.remove_thinking_from_demonstration,
        teacher_regularization=cli_config.teacher_regularization,
        teacher_mix_alpha=cli_config.teacher_mix_alpha,
        ema_teacher_history=cli_config.ema_teacher_history,
        full_logit_distillation=cli_config.full_logit_distillation,
        distillation_topk=cli_config.distillation_topk,
        distillation_add_tail=cli_config.distillation_add_tail,
        reprompt_template=cli_config.reprompt_template,
        solution_template=cli_config.solution_template,
        feedback_template=cli_config.feedback_template,
        feedback_keys_csv=cli_config.feedback_keys_csv,
        enable_stockfish_hints=cli_config.enable_stockfish_hints,
        stockfish_path=cli_config.stockfish_path,
        stockfish_depth=cli_config.stockfish_depth,
        stockfish_multipv=cli_config.stockfish_multipv,
        stockfish_num_workers=cli_config.stockfish_num_workers,
        stockfish_threads=cli_config.stockfish_threads,
        stockfish_hash_mb=cli_config.stockfish_hash_mb,
        stockfish_wdl_model=cli_config.stockfish_wdl_model,
        stockfish_max_pv_plies=cli_config.stockfish_max_pv_plies,
        stockfish_hint_max_good_moves=cli_config.stockfish_hint_max_good_moves,
        stockfish_hint_max_bad_moves=cli_config.stockfish_hint_max_bad_moves,
        stockfish_hint_bad_move_threshold=cli_config.stockfish_hint_bad_move_threshold,
        stockfish_include_fen_decode=cli_config.stockfish_include_fen_decode,
        stockfish_include_ascii_board=cli_config.stockfish_include_ascii_board,
        stockfish_include_search_stats=cli_config.stockfish_include_search_stats,
        stockfish_analysis_time_limit_sec=cli_config.stockfish_analysis_time_limit_sec,
        stockfish_engine_max_retries=cli_config.stockfish_engine_max_retries,
        stockfish_max_root_cache_entries=cli_config.stockfish_max_root_cache_entries,
        stockfish_max_move_cache_entries=cli_config.stockfish_max_move_cache_entries,
        stockfish_max_verification_cache_entries=cli_config.stockfish_max_verification_cache_entries,
        stockfish_max_piece_pressure_items=cli_config.stockfish_max_piece_pressure_items,
        stockfish_max_weak_square_items=cli_config.stockfish_max_weak_square_items,
        stockfish_syzygy_path=cli_config.stockfish_syzygy_path,
        stockfish_syzygy_max_pieces=cli_config.stockfish_syzygy_max_pieces,
        stockfish_persistent_cache_dir=cli_config.stockfish_persistent_cache_dir,
        stockfish_unknown_score_cp_loss=cli_config.stockfish_unknown_score_cp_loss,
        stockfish_hints_template=cli_config.stockfish_hints_template,
        stockfish_hints_only_without_solution=cli_config.stockfish_hints_only_without_solution,
        enable_stockfish_move_verification=cli_config.enable_stockfish_move_verification,
        stockfish_verification_sample_rate=cli_config.stockfish_verification_sample_rate,
        stockfish_verification_depth=cli_config.stockfish_verification_depth,
        stockfish_verification_multipv=cli_config.stockfish_verification_multipv,
        stockfish_shared_hint_and_verification_eval=(
            cli_config.stockfish_shared_hint_and_verification_eval
        ),
        stockfish_shared_eval_mode=cli_config.stockfish_shared_eval_mode,
        stockfish_illegal_move_cp_loss=cli_config.stockfish_illegal_move_cp_loss,
        include_stockfish_move_feedback=cli_config.include_stockfish_move_feedback,
        stockfish_feedback_cp_loss_threshold=cli_config.stockfish_feedback_cp_loss_threshold,
        max_reprompt_tokens=cli_config.max_reprompt_tokens,
        reprompt_truncation=cli_config.reprompt_truncation,
        strict_single_turn=cli_config.strict_single_turn,
        max_concurrent_teacher_logprobs=cli_config.max_concurrent_teacher_logprobs,
        student_max_thinking_tokens=cli_config.student_max_thinking_tokens,
        grpo_mix_lambda=cli_config.grpo_mix_lambda,
        advantage_mode=cli_config.advantage_mode,
        updates_per_batch=cli_config.updates_per_batch,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=loss_fn_config,
    )

    await sdpo_train.main(cfg)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config, None))
