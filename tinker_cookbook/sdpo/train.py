from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import re
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import chz
import tinker
import torch
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.sdpo.chess_hints import (
    StockfishHintConfig,
    StockfishHintPool,
    extract_fen_from_state,
)
from tinker_cookbook.sdpo.utils import (
    build_sdpo_datum,
    build_teacher_messages,
    extract_feedback_text,
    maybe_strip_thinking,
    select_solution_idx,
    trust_region_mix_logprob,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)

_REFERENCE_SAMPLER_FILE = "sdpo_reference_sampler_path.txt"
_THINK_OPEN_RE = re.compile(r"<think\b", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)


class MultiTurnStateError(ValueError):
    """Raised when strict_single_turn=True and a multi-turn rollout is encountered."""


@dataclass
class RolloutSample:
    prompt_messages: list[renderers.Message]
    fen: str | None
    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_logprobs: list[float]
    reward: float
    feedback_text: str | None
    expected_answer: str | None
    response_text: str


@dataclass
class GroupSdpoStats:
    num_samples: int = 0
    success_samples: int = 0
    success_groups: int = 0
    feedback_available_samples: int = 0
    feedback_used_samples: int = 0
    stockfish_hint_available_samples: int = 0
    stockfish_hint_used_samples: int = 0
    stockfish_verification_candidate_samples: int = 0
    stockfish_verification_scheduled_samples: int = 0
    stockfish_verified_samples: int = 0
    stockfish_legal_move_samples: int = 0
    stockfish_best_move_samples: int = 0
    stockfish_cp_loss_sum: float = 0.0
    stockfish_estimated_cp_loss_samples: int = 0
    stockfish_feedback_samples: int = 0
    reprompt_samples: int = 0
    zero_adv_samples: int = 0
    skipped_samples: int = 0
    advantage_sum: float = 0.0
    advantage_abs_sum: float = 0.0
    advantage_count: int = 0
    topk_overlap_count: int = 0
    topk_total_count: int = 0
    env_metric_sums: dict[str, float] = field(default_factory=dict)
    env_metric_counts: dict[str, int] = field(default_factory=dict)


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder
    model_name: str
    max_tokens: int
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))

    # model/training setup
    lora_rank: int = 32
    temperature: float = 1.0
    num_substeps: int = 1
    base_url: str | None = None
    load_checkpoint_path: str | None = None

    # checkpoint/logging
    save_every: int = 10
    ttl_seconds: int | None = 604800
    wandb_project: str | None = None
    wandb_name: str | None = None
    debug_examples_every_n_steps: int = 0
    debug_examples_per_step: int = 2
    debug_examples_max_text_chars: int = 4000
    debug_examples_file_name: str = "sdpo_debug_examples.jsonl"

    # verifiers rollout
    group_size: int = 8
    max_concurrent_generation: int = -1
    max_concurrent_scoring: int = -1

    # SDPO-specific fields
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
        "Now answer the original question.\n"
        "Output exactly one legal move in UCI format and nothing else.\n"
        "Do not include analysis, commentary, or <think> blocks.\n"
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
    student_max_thinking_tokens: int = 2000
    grpo_mix_lambda: float = 0.0
    advantage_mode: Literal["token", "sequence"] = "token"
    updates_per_batch: int = 1
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None


def _parse_feedback_keys(feedback_keys_csv: str) -> list[str]:
    return [key.strip() for key in feedback_keys_csv.split(",") if key.strip()]


def _hash_to_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def _answer_text_after_think_blocks(response_text: str) -> str:
    if not response_text:
        return ""
    last_close: re.Match[str] | None = None
    for match in _THINK_CLOSE_RE.finditer(response_text):
        last_close = match
    if last_close is not None:
        return response_text[last_close.end() :].strip()
    if _THINK_OPEN_RE.search(response_text) is not None:
        return ""
    return response_text.strip()


def _should_run_stockfish_verification(
    sample: RolloutSample,
    sample_rate: float,
) -> bool:
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    key = f"{sample.fen or ''}|{sample.response_text}"
    return _hash_to_unit_interval(key) < sample_rate


async def _stockfish_analyze_and_render_async(
    stockfish_client: Any,
    fen: str,
) -> str:
    analyze_async = getattr(stockfish_client, "analyze_and_render_async", None)
    if callable(analyze_async):
        return await analyze_async(fen)
    return await asyncio.to_thread(stockfish_client.analyze_and_render, fen)


async def _stockfish_verify_predicted_move_async(
    stockfish_client: Any,
    fen: str,
    predicted_text: str,
    depth: int,
    multipv: int | None,
    illegal_move_cp_loss: float,
) -> Any:
    verify_async = getattr(stockfish_client, "verify_predicted_move_async", None)
    if callable(verify_async):
        try:
            return await verify_async(
                fen=fen,
                predicted_text=predicted_text,
                depth=depth,
                verification_multipv=multipv,
                illegal_move_cp_loss=illegal_move_cp_loss,
            )
        except TypeError:
            return await verify_async(
                fen=fen,
                predicted_text=predicted_text,
                depth=depth,
                illegal_move_cp_loss=illegal_move_cp_loss,
            )

    verify_sync = getattr(stockfish_client, "verify_predicted_move", None)
    if verify_sync is None:
        raise AttributeError("stockfish_client is missing verify_predicted_move")
    try:
        return await asyncio.to_thread(
            verify_sync,
            fen,
            predicted_text,
            depth,
            illegal_move_cp_loss,
            multipv,
        )
    except TypeError:
        return await asyncio.to_thread(
            verify_sync,
            fen,
            predicted_text,
            depth,
            illegal_move_cp_loss,
        )


async def _stockfish_analyze_and_verify_async(
    stockfish_client: Any,
    *,
    fen: str,
    predicted_text: str,
    hint_depth: int,
    hint_multipv: int,
    verification_depth: int,
    verification_multipv: int,
    illegal_move_cp_loss: float,
    mode: Literal["single", "two_pass"],
) -> tuple[str, Any | None]:
    analyze_and_verify_async = getattr(stockfish_client, "analyze_and_verify_async", None)
    if callable(analyze_and_verify_async):
        return await analyze_and_verify_async(
            fen=fen,
            predicted_text=predicted_text,
            hint_depth=hint_depth,
            hint_multipv=hint_multipv,
            verification_depth=verification_depth,
            verification_multipv=verification_multipv,
            illegal_move_cp_loss=illegal_move_cp_loss,
            mode=mode,
        )

    analyze_and_verify = getattr(stockfish_client, "analyze_and_verify", None)
    if callable(analyze_and_verify):
        return await asyncio.to_thread(
            analyze_and_verify,
            fen,
            predicted_text,
            hint_depth=hint_depth,
            hint_multipv=hint_multipv,
            verification_depth=verification_depth,
            verification_multipv=verification_multipv,
            illegal_move_cp_loss=illegal_move_cp_loss,
            mode=mode,
        )

    # Compatibility fallback for older stockfish clients without combined APIs.
    hints_text = await _stockfish_analyze_and_render_async(stockfish_client, fen)
    verification = await _stockfish_verify_predicted_move_async(
        stockfish_client=stockfish_client,
        fen=fen,
        predicted_text=predicted_text,
        depth=verification_depth,
        multipv=verification_multipv,
        illegal_move_cp_loss=illegal_move_cp_loss,
    )
    return hints_text, verification


def _normalize_prompt_messages(prompt_value: Any) -> list[renderers.Message]:
    if isinstance(prompt_value, list):
        if all(isinstance(item, Mapping) for item in prompt_value):
            return [dict(item) for item in prompt_value]
    if isinstance(prompt_value, str) and prompt_value.strip():
        return [{"role": "user", "content": prompt_value.strip()}]
    raise ValueError("Unable to extract prompt messages from verifiers state")


def _coerce_text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_coerce_text(item) for item in value]
        joined = "\n".join(part for part in parts if part)
        return joined if joined else None
    if isinstance(value, Mapping):
        for key in ("answer", "expected_answer", "text", "content", "value", "move", "uci"):
            nested = _coerce_text(value.get(key))
            if nested:
                return nested
    return None


def _accumulate_numeric_env_metrics(
    stats: GroupSdpoStats,
    metrics_value: Any,
) -> None:
    if not isinstance(metrics_value, Mapping):
        return
    for key, value in metrics_value.items():
        if not isinstance(key, str) or not key:
            continue
        if isinstance(value, bool):
            numeric_value = float(value)
        elif isinstance(value, (int, float)):
            numeric_value = float(value)
        else:
            continue
        stats.env_metric_sums[key] = stats.env_metric_sums.get(key, 0.0) + numeric_value
        stats.env_metric_counts[key] = stats.env_metric_counts.get(key, 0) + 1


def _extract_expected_answer_text(state: Mapping[str, Any]) -> str | None:
    answer_keys = [
        "expected_answer",
        "answer",
        "target_answer",
        "ground_truth_answer",
        "ground_truth",
        "reference_answer",
        "label",
        "solution",
    ]

    for key in answer_keys:
        value = _coerce_text(state.get(key))
        if value:
            return value

    for container_key in ("rollout_input", "info", "metadata", "extra_info", "reward_extra_info"):
        container = state.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key in answer_keys:
            value = _coerce_text(container.get(key))
            if value:
                return value

    for container_key in ("info", "metadata", "extra_info", "reward_extra_info"):
        container = state.get(container_key)
        if not isinstance(container, Mapping):
            continue
        stockfish_best = _coerce_text(container.get("stockfish_best_move"))
        if stockfish_best:
            return stockfish_best

    return None


def _prompt_messages_to_text(prompt_messages: Sequence[renderers.Message]) -> str:
    chunks: list[str] = []
    for message in prompt_messages:
        role = str(message.get("role", "unknown")).strip() or "unknown"
        content = renderers.format_content_as_string(message.get("content", ""))
        if content:
            chunks.append(f"{role}: {content}")
    return "\n\n".join(chunks)


def _truncate_debug_text(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...[truncated {len(text) - max_chars} chars]"


def _format_debug_examples_long_text(
    batch_index: int,
    debug_examples: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = [f"SDPO debug examples for batch {batch_index}"]
    for i, example in enumerate(debug_examples):
        lines.append("")
        lines.append(f"[example {i}]")
        for key in (
            "group_index",
            "sample_index_in_group",
            "reward",
            "fen",
            "expected_answer",
            "stockfish_best_move",
            "stockfish_cp_loss",
        ):
            value = example.get(key)
            if value is not None and value != "":
                lines.append(f"{key}: {value}")
        for text_key in (
            "prompt",
            "model_output",
            "teacher_solution",
            "stockfish_hint",
            "combined_feedback",
        ):
            text_value = example.get(text_key)
            if isinstance(text_value, str) and text_value:
                lines.append(f"{text_key}:")
                lines.append(text_value)
    return "\n".join(lines)


def _append_debug_examples_jsonl(
    log_path: str,
    file_name: str,
    batch_index: int,
    debug_examples: Sequence[Mapping[str, Any]],
) -> str:
    out_path = file_name
    if not os.path.isabs(out_path):
        out_path = os.path.join(log_path, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for example in debug_examples:
            payload = {"batch_index": batch_index, "batch_number": batch_index + 1}
            payload.update(example)
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return out_path


def _extract_single_turn_tokens(
    state: Mapping[str, Any],
    strict_single_turn: bool,
) -> tuple[list[int], list[int], list[float]]:
    trajectory = state.get("trajectory")
    if not isinstance(trajectory, list):
        raise ValueError("State is missing trajectory list")

    completion_segments: list[Mapping[str, Any]] = []
    for step in trajectory:
        if not isinstance(step, Mapping):
            continue
        tokens = step.get("tokens")
        if not isinstance(tokens, Mapping):
            continue
        completion_ids = tokens.get("completion_ids")
        if isinstance(completion_ids, list) and len(completion_ids) > 0:
            completion_segments.append(tokens)

    if strict_single_turn and len(completion_segments) != 1:
        raise MultiTurnStateError(
            "strict_single_turn=True requires exactly one completion segment per state; "
            f"found {len(completion_segments)}. For multi-turn traces, either flatten to a single "
            "completion segment before SDPO training or set strict_single_turn=False."
        )
    if not completion_segments:
        raise ValueError("No completion tokens found in state trajectory")

    tokens = completion_segments[-1]
    prompt_ids = tokens.get("prompt_ids")
    completion_ids = tokens.get("completion_ids")
    completion_logprobs = tokens.get("completion_logprobs")

    if not isinstance(prompt_ids, list) or not all(isinstance(x, int) for x in prompt_ids):
        raise ValueError("State tokens.prompt_ids must be a list[int]")
    if not isinstance(completion_ids, list) or not all(isinstance(x, int) for x in completion_ids):
        raise ValueError("State tokens.completion_ids must be a list[int]")
    if not isinstance(completion_logprobs, list):
        raise ValueError("State tokens.completion_logprobs must be a list[float]")

    completion_logprobs_f = [float(x) for x in completion_logprobs]
    if len(completion_ids) != len(completion_logprobs_f):
        raise ValueError(
            "completion_ids and completion_logprobs length mismatch "
            f"({len(completion_ids)} != {len(completion_logprobs_f)})"
        )

    return list(prompt_ids), list(completion_ids), completion_logprobs_f


def _extract_completion_text(state: Mapping[str, Any]) -> str | None:
    completion = state.get("completion")
    if isinstance(completion, list):
        for message in reversed(completion):
            if not isinstance(message, Mapping):
                continue
            content = renderers.format_content_as_string(message.get("content", ""))
            if content:
                return content
    elif isinstance(completion, str):
        content = completion.strip()
        if content:
            return content

    for key in ("response", "output", "text", "assistant_response"):
        content = _coerce_text(state.get(key))
        if content:
            return content

    return None


def _extract_rollout_sample(
    state: Mapping[str, Any],
    feedback_keys: Sequence[str],
    renderer: renderers.Renderer,
    tokenizer: Tokenizer,
    strict_single_turn: bool,
) -> RolloutSample:
    prompt_messages = _normalize_prompt_messages(state.get("prompt"))
    fen = extract_fen_from_state(state, prompt_messages)
    try:
        prompt_tokens, completion_tokens, completion_logprobs = _extract_single_turn_tokens(
            state=state,
            strict_single_turn=strict_single_turn,
        )
        response_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    except MultiTurnStateError:
        raise
    except ValueError:
        completion_text = _extract_completion_text(state)
        if completion_text is None:
            raise
        prompt_tokens = renderer.build_generation_prompt(prompt_messages).to_ints()
        completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
        completion_logprobs = []
        response_text = completion_text

    reward = float(state.get("reward") or 0.0)
    feedback_text = extract_feedback_text(state, feedback_keys)
    expected_answer = _extract_expected_answer_text(state)

    return RolloutSample(
        prompt_messages=prompt_messages,
        fen=fen,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_logprobs=completion_logprobs,
        reward=reward,
        feedback_text=feedback_text,
        expected_answer=expected_answer,
        response_text=response_text,
    )


def _extract_completion_logprobs_from_full_sequence(
    logprobs: list[float | None],
    prompt_len: int,
    completion_len: int,
    sequence_label: str,
) -> list[float]:
    completion_logprobs = logprobs[prompt_len : prompt_len + completion_len]
    if len(completion_logprobs) != completion_len:
        raise ValueError(
            f"{sequence_label}: expected {completion_len} completion logprobs but got "
            f"{len(completion_logprobs)}"
        )
    if any(v is None for v in completion_logprobs):
        raise ValueError(f"{sequence_label}: completion logprobs contain None entries")
    return [float(v) for v in completion_logprobs if v is not None]


async def _maybe_compute_logprobs_with_semaphore(
    sampling_client: tinker.SamplingClient,
    model_input: tinker.ModelInput,
    semaphore: asyncio.Semaphore | None,
) -> list[float | None]:
    if semaphore is None:
        return await sampling_client.compute_logprobs_async(model_input)
    async with semaphore:
        return await sampling_client.compute_logprobs_async(model_input)


async def _maybe_sample_with_topk_with_semaphore(
    sampling_client: tinker.SamplingClient,
    model_input: tinker.ModelInput,
    topk_prompt_logprobs: int,
    semaphore: asyncio.Semaphore | None,
) -> Any:
    sampling_params = tinker.SamplingParams(max_tokens=1)
    if semaphore is None:
        return await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )
    async with semaphore:
        return await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )


def _extract_completion_topk_from_full_sequence(
    topk_logprobs: list[list[tuple[int, float]] | None],
    prompt_len: int,
    completion_len: int,
    sequence_label: str,
) -> list[list[tuple[int, float]]]:
    completion_topk = topk_logprobs[prompt_len : prompt_len + completion_len]
    if len(completion_topk) != completion_len:
        raise ValueError(
            f"{sequence_label}: expected {completion_len} completion top-k entries but got "
            f"{len(completion_topk)}"
        )

    normalized: list[list[tuple[int, float]]] = []
    for idx, entry in enumerate(completion_topk):
        if entry is None:
            normalized.append([])
            continue
        normalized_entry: list[tuple[int, float]] = []
        for token_id, logprob in entry:
            normalized_entry.append((int(token_id), float(logprob)))
        normalized.append(normalized_entry)
        if len(normalized_entry) == 0:
            logger.debug("%s: empty top-k entry at completion index %d", sequence_label, idx)

    return normalized


def _ema_distribution_weights(alpha: float, num_components: int) -> list[float]:
    if num_components <= 0:
        raise ValueError(f"num_components must be >= 1, got {num_components}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")

    if num_components == 1:
        return [1.0]

    weights: list[float] = []
    for i in range(num_components):
        if i < num_components - 1:
            weights.append(alpha * ((1.0 - alpha) ** i))
        else:
            # absorb all remaining probability mass into the oldest retained teacher
            weights.append((1.0 - alpha) ** i)

    total = sum(weights)
    return [weight / total for weight in weights]


def _mix_logprob_tensors(
    logprob_tensors: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    if len(logprob_tensors) != len(weights):
        raise ValueError(
            f"logprob_tensors and weights length mismatch ({len(logprob_tensors)} != {len(weights)})"
        )
    if len(logprob_tensors) == 0:
        raise ValueError("logprob_tensors must be non-empty")

    stacked = torch.stack(
        [
            tensor + math.log(weight)
            for tensor, weight in zip(logprob_tensors, weights, strict=True)
        ],
        dim=0,
    )
    return torch.logsumexp(stacked, dim=0)


def _compute_topk_tail_advantage(
    student_topk: Sequence[tuple[int, float]],
    teacher_topk: Sequence[tuple[int, float]],
    add_tail: bool,
) -> tuple[float | None, int, int]:
    student_map = {token_id: logprob for token_id, logprob in student_topk}
    teacher_map = {token_id: logprob for token_id, logprob in teacher_topk}
    shared_tokens = [token_id for token_id in student_map if token_id in teacher_map]

    if not shared_tokens:
        return None, 0, len(student_map)

    expected_advantage = 0.0
    shared_student_mass = 0.0
    shared_teacher_mass = 0.0
    for token_id in shared_tokens:
        lp_student = student_map[token_id]
        lp_teacher = teacher_map[token_id]
        p_student = math.exp(lp_student)
        expected_advantage += p_student * (lp_teacher - lp_student)
        shared_student_mass += p_student
        shared_teacher_mass += math.exp(lp_teacher)

    if add_tail:
        p_student_tail = max(1e-12, 1.0 - shared_student_mass)
        p_teacher_tail = max(1e-12, 1.0 - shared_teacher_mass)
        expected_advantage += p_student_tail * (
            math.log(p_teacher_tail) - math.log(p_student_tail)
        )

    return expected_advantage, len(shared_tokens), len(student_map)


def _mix_trust_region_topk_entries(
    teacher_topk: Sequence[tuple[int, float]],
    reference_topk: Sequence[tuple[int, float]],
    alpha: float,
) -> list[tuple[int, float]]:
    reference_map = {token_id: logprob for token_id, logprob in reference_topk}
    mixed_entries: list[tuple[int, float]] = []
    for token_id, teacher_logprob in teacher_topk:
        reference_logprob = reference_map.get(token_id)
        if reference_logprob is None:
            continue
        mixed_logprob = trust_region_mix_logprob(
            student_logprob=torch.tensor([teacher_logprob], dtype=torch.float32),
            reference_logprob=torch.tensor([reference_logprob], dtype=torch.float32),
            alpha=alpha,
        )[0].item()
        mixed_entries.append((token_id, mixed_logprob))
    return mixed_entries


def _mix_ema_topk_entries(
    topk_entries_by_component: Sequence[Sequence[tuple[int, float]]],
    weights: Sequence[float],
) -> list[tuple[int, float]]:
    if len(topk_entries_by_component) == 0:
        return []
    if len(topk_entries_by_component) != len(weights):
        raise ValueError(
            "topk_entries_by_component and weights must have the same length "
            f"({len(topk_entries_by_component)} != {len(weights)})"
        )

    token_sets = [{token_id for token_id, _ in entries} for entries in topk_entries_by_component]
    shared_tokens = set.intersection(*token_sets) if token_sets else set()
    if not shared_tokens:
        return []

    component_maps = [{token_id: logprob for token_id, logprob in entries} for entries in topk_entries_by_component]
    mixed_entries: list[tuple[int, float]] = []
    for token_id in shared_tokens:
        mixed_logprob = torch.logsumexp(
            torch.tensor(
                [
                    component_map[token_id] + math.log(weight)
                    for component_map, weight in zip(component_maps, weights, strict=True)
                ],
                dtype=torch.float32,
            ),
            dim=0,
        ).item()
        mixed_entries.append((token_id, mixed_logprob))

    return mixed_entries


def _truncate_teacher_prompt_tokens(
    prompt_tokens: list[int],
    max_reprompt_tokens: int,
    truncation: Literal["left", "right", "error"],
) -> list[int]:
    # max_reprompt_tokens <= 0 disables truncation entirely.
    if max_reprompt_tokens <= 0:
        return prompt_tokens

    if len(prompt_tokens) <= max_reprompt_tokens:
        return prompt_tokens

    if truncation == "right":
        return prompt_tokens[:max_reprompt_tokens]
    if truncation == "left":
        return prompt_tokens[-max_reprompt_tokens:]
    if truncation == "error":
        raise ValueError(
            "Reprompt prompt length exceeds max_reprompt_tokens with reprompt_truncation='error' "
            f"({len(prompt_tokens)} > {max_reprompt_tokens})"
        )

    raise ValueError(f"Unknown reprompt_truncation value: {truncation}")


async def _compute_sample_advantages(
    sample: RolloutSample,
    teacher_messages: list[renderers.Message],
    used_reprompt: bool,
    current_sampling_client: tinker.SamplingClient,
    reference_sampling_client: tinker.SamplingClient | None,
    ema_teacher_sampling_clients: Sequence[tinker.SamplingClient] | None,
    renderer: renderers.Renderer,
    teacher_regularization: Literal["trust_region", "ema", "none"],
    teacher_mix_alpha: float,
    full_logit_distillation: bool,
    distillation_topk: int | None,
    distillation_add_tail: bool,
    max_reprompt_tokens: int,
    reprompt_truncation: Literal["left", "right", "error"],
    semaphore: asyncio.Semaphore | None,
) -> tuple[list[float], int, int, list[float]]:
    rollout_logprobs = list(sample.completion_logprobs)
    if len(rollout_logprobs) != len(sample.completion_tokens):
        original_sequence_tokens = sample.prompt_tokens + sample.completion_tokens
        original_input = tinker.ModelInput.from_ints(original_sequence_tokens)
        full_rollout_logprobs = await _maybe_compute_logprobs_with_semaphore(
            sampling_client=current_sampling_client,
            model_input=original_input,
            semaphore=semaphore,
        )
        rollout_logprobs = _extract_completion_logprobs_from_full_sequence(
            logprobs=full_rollout_logprobs,
            prompt_len=len(sample.prompt_tokens),
            completion_len=len(sample.completion_tokens),
            sequence_label="rollout",
        )

    if not used_reprompt:
        return [0.0] * len(sample.completion_tokens), 0, 0, rollout_logprobs

    teacher_prompt_tokens = renderer.build_generation_prompt(teacher_messages).to_ints()
    teacher_prompt_tokens = _truncate_teacher_prompt_tokens(
        prompt_tokens=teacher_prompt_tokens,
        max_reprompt_tokens=max_reprompt_tokens,
        truncation=reprompt_truncation,
    )

    teacher_sequence_tokens = teacher_prompt_tokens + sample.completion_tokens
    teacher_input = tinker.ModelInput.from_ints(teacher_sequence_tokens)
    completion_len = len(sample.completion_tokens)

    async def _compute_completion_logprobs(
        sampling_client: tinker.SamplingClient,
        sequence_input: tinker.ModelInput,
        prompt_len: int,
        sequence_label: str,
    ) -> torch.Tensor:
        full_logprobs = await _maybe_compute_logprobs_with_semaphore(
            sampling_client=sampling_client,
            model_input=sequence_input,
            semaphore=semaphore,
        )
        completion_logprobs = _extract_completion_logprobs_from_full_sequence(
            logprobs=full_logprobs,
            prompt_len=prompt_len,
            completion_len=completion_len,
            sequence_label=sequence_label,
        )
        return torch.tensor(completion_logprobs, dtype=torch.float32)

    async def _compute_completion_logprobs_and_topk(
        sampling_client: tinker.SamplingClient,
        sequence_input: tinker.ModelInput,
        prompt_len: int,
        sequence_label: str,
        topk: int,
    ) -> tuple[torch.Tensor, list[list[tuple[int, float]]]]:
        sample_response = await _maybe_sample_with_topk_with_semaphore(
            sampling_client=sampling_client,
            model_input=sequence_input,
            topk_prompt_logprobs=topk,
            semaphore=semaphore,
        )
        if sample_response.prompt_logprobs is None:
            raise ValueError(f"{sequence_label}: prompt_logprobs missing from sample response")
        if sample_response.topk_prompt_logprobs is None:
            raise ValueError(f"{sequence_label}: topk_prompt_logprobs missing from sample response")

        completion_logprobs = _extract_completion_logprobs_from_full_sequence(
            logprobs=list(sample_response.prompt_logprobs),
            prompt_len=prompt_len,
            completion_len=completion_len,
            sequence_label=sequence_label,
        )
        completion_topk = _extract_completion_topk_from_full_sequence(
            topk_logprobs=list(sample_response.topk_prompt_logprobs),
            prompt_len=prompt_len,
            completion_len=completion_len,
            sequence_label=sequence_label,
        )
        return torch.tensor(completion_logprobs, dtype=torch.float32), completion_topk

    teacher_tensor: torch.Tensor
    teacher_completion_topk: list[list[tuple[int, float]]] | None = None

    if teacher_regularization == "ema":
        if not ema_teacher_sampling_clients:
            raise ValueError("EMA teacher regularization requires non-empty ema_teacher_sampling_clients")
        if full_logit_distillation:
            if distillation_topk is None or distillation_topk <= 0:
                raise ValueError(
                    "full_logit_distillation=True requires distillation_topk to be a positive integer"
                )
            ema_outputs = await asyncio.gather(
                *[
                    _compute_completion_logprobs_and_topk(
                        sampling_client=ema_client,
                        sequence_input=teacher_input,
                        prompt_len=len(teacher_prompt_tokens),
                        sequence_label=f"ema_teacher_{i}",
                        topk=distillation_topk,
                    )
                    for i, ema_client in enumerate(ema_teacher_sampling_clients)
                ]
            )
            ema_tensors = [output[0] for output in ema_outputs]
            ema_topk_entries = [output[1] for output in ema_outputs]
            weights = _ema_distribution_weights(teacher_mix_alpha, len(ema_tensors))
            teacher_tensor = _mix_logprob_tensors(ema_tensors, weights)
            teacher_completion_topk = []
            for entries_by_position in zip(*ema_topk_entries, strict=True):
                teacher_completion_topk.append(
                    _mix_ema_topk_entries(
                        topk_entries_by_component=entries_by_position,
                        weights=weights,
                    )
                )
        else:
            ema_tensors = await asyncio.gather(
                *[
                    _compute_completion_logprobs(
                        sampling_client=ema_client,
                        sequence_input=teacher_input,
                        prompt_len=len(teacher_prompt_tokens),
                        sequence_label=f"ema_teacher_{i}",
                    )
                    for i, ema_client in enumerate(ema_teacher_sampling_clients)
                ]
            )
            weights = _ema_distribution_weights(teacher_mix_alpha, len(ema_tensors))
            teacher_tensor = _mix_logprob_tensors(ema_tensors, weights)
    else:
        if full_logit_distillation:
            if distillation_topk is None or distillation_topk <= 0:
                raise ValueError(
                    "full_logit_distillation=True requires distillation_topk to be a positive integer"
                )
            teacher_tensor, teacher_completion_topk = await _compute_completion_logprobs_and_topk(
                sampling_client=current_sampling_client,
                sequence_input=teacher_input,
                prompt_len=len(teacher_prompt_tokens),
                sequence_label="teacher",
                topk=distillation_topk,
            )
        else:
            teacher_tensor = await _compute_completion_logprobs(
                sampling_client=current_sampling_client,
                sequence_input=teacher_input,
                prompt_len=len(teacher_prompt_tokens),
                sequence_label="teacher",
            )

        if teacher_regularization == "trust_region":
            if reference_sampling_client is None:
                raise ValueError("reference_sampling_client is required for trust_region regularization")

            reference_completion_topk: list[list[tuple[int, float]]] | None = None
            if full_logit_distillation:
                assert distillation_topk is not None
                reference_tensor, reference_completion_topk = await _compute_completion_logprobs_and_topk(
                    sampling_client=reference_sampling_client,
                    sequence_input=teacher_input,
                    prompt_len=len(teacher_prompt_tokens),
                    sequence_label="reference",
                    topk=distillation_topk,
                )
            else:
                reference_tensor = await _compute_completion_logprobs(
                    sampling_client=reference_sampling_client,
                    sequence_input=teacher_input,
                    prompt_len=len(teacher_prompt_tokens),
                    sequence_label="reference",
                )

            teacher_tensor = trust_region_mix_logprob(
                student_logprob=teacher_tensor,
                reference_logprob=reference_tensor,
                alpha=teacher_mix_alpha,
            )

            if full_logit_distillation and teacher_completion_topk is not None:
                assert reference_completion_topk is not None
                teacher_completion_topk = [
                    _mix_trust_region_topk_entries(
                        teacher_topk=teacher_entries,
                        reference_topk=reference_entries,
                        alpha=teacher_mix_alpha,
                    )
                    for teacher_entries, reference_entries in zip(
                        teacher_completion_topk,
                        reference_completion_topk,
                        strict=True,
                    )
                ]

    rollout_tensor = torch.tensor(rollout_logprobs, dtype=torch.float32)
    if not full_logit_distillation:
        advantages_tensor = teacher_tensor - rollout_tensor
        return advantages_tensor.tolist(), 0, 0, rollout_logprobs

    assert distillation_topk is not None
    assert teacher_completion_topk is not None

    original_sequence_tokens = sample.prompt_tokens + sample.completion_tokens
    original_input = tinker.ModelInput.from_ints(original_sequence_tokens)
    _, student_completion_topk = await _compute_completion_logprobs_and_topk(
        sampling_client=current_sampling_client,
        sequence_input=original_input,
        prompt_len=len(sample.prompt_tokens),
        sequence_label="student",
        topk=distillation_topk,
    )

    fallback_token_advantages = (teacher_tensor - rollout_tensor).tolist()

    advantages: list[float] = []
    topk_overlap_count = 0
    topk_total_count = 0
    for idx, (student_topk, teacher_topk) in enumerate(
        zip(student_completion_topk, teacher_completion_topk, strict=True)
    ):
        topk_advantage, overlap_count, total_count = _compute_topk_tail_advantage(
            student_topk=student_topk,
            teacher_topk=teacher_topk,
            add_tail=distillation_add_tail,
        )
        topk_overlap_count += overlap_count
        topk_total_count += total_count
        if topk_advantage is None:
            advantages.append(float(fallback_token_advantages[idx]))
        else:
            advantages.append(float(topk_advantage))

    return advantages, topk_overlap_count, topk_total_count, rollout_logprobs


async def build_group_sdpo_datums(
    states: Sequence[Mapping[str, Any]],
    config: Config,
    current_sampling_client: tinker.SamplingClient,
    reference_sampling_client: tinker.SamplingClient | None,
    ema_teacher_sampling_clients: Sequence[tinker.SamplingClient] | None,
    renderer: renderers.Renderer,
    tokenizer: Tokenizer,
    feedback_keys: Sequence[str],
    teacher_logprob_semaphore: asyncio.Semaphore | None,
    stockfish_hint_extractor: Any | None,
    group_index: int | None = None,
    debug_examples_sink: list[dict[str, Any]] | None = None,
    debug_examples_limit: int = 0,
    debug_examples_max_text_chars: int = 0,
) -> tuple[list[tinker.Datum], GroupSdpoStats]:
    records: list[RolloutSample] = []
    stats = GroupSdpoStats()

    for state in states:
        _accumulate_numeric_env_metrics(stats, state.get("metrics"))
        try:
            sample = _extract_rollout_sample(
                state=state,
                feedback_keys=feedback_keys,
                renderer=renderer,
                tokenizer=tokenizer,
                strict_single_turn=config.strict_single_turn,
            )
            records.append(sample)
        except MultiTurnStateError:
            raise
        except Exception as exc:
            stats.skipped_samples += 1
            logger.warning("Skipping invalid rollout sample: %s", exc)

    if not records:
        return [], stats

    rewards = [record.reward for record in records]
    group_mean_reward = float(sum(rewards) / len(rewards))
    grpo_advantages = [float(reward - group_mean_reward) for reward in rewards]
    stats.num_samples = len(records)
    stats.success_samples = sum(
        1 for reward in rewards if reward >= config.success_reward_threshold
    )
    stats.success_groups = 1 if stats.success_samples > 0 else 0

    stockfish_hints_by_fen: dict[str, str] = {}
    datum_tasks: list[
        asyncio.Task[
            tuple[
                tinker.Datum,
                list[float],
                bool,
                bool,
                bool,
                bool,
                bool,
                float,
                bool,
                bool,
                bool,
                bool,
                int,
                int,
            ]
        ]
    ] = []
    solution_texts: list[str | None] = [None] * len(records)
    hint_task_by_fen: dict[str, asyncio.Task[str]] = {}
    hint_task_by_idx: list[asyncio.Task[str] | None] = [None] * len(records)
    verification_task_by_idx: list[asyncio.Task[Any | None] | None] = [None] * len(records)
    shared_stockfish_task_by_idx: list[asyncio.Task[tuple[str, Any | None]] | None] = [None] * len(
        records
    )
    verification_candidate_by_idx: list[bool] = [False] * len(records)
    verification_scheduled_by_idx: list[bool] = [False] * len(records)

    async def _run_stockfish_hint_task(fen: str) -> str:
        try:
            if stockfish_hint_extractor is None:
                return ""
            return await _stockfish_analyze_and_render_async(stockfish_hint_extractor, fen)
        except Exception as exc:
            logger.warning("Failed to build Stockfish hints for FEN %s: %s", fen, exc)
            return ""

    async def _run_stockfish_verification_task(sample: RolloutSample) -> Any | None:
        try:
            if stockfish_hint_extractor is None or sample.fen is None:
                return None
            predicted_answer_text = _answer_text_after_think_blocks(sample.response_text)
            return await _stockfish_verify_predicted_move_async(
                stockfish_client=stockfish_hint_extractor,
                fen=sample.fen,
                predicted_text=predicted_answer_text,
                depth=config.stockfish_verification_depth,
                multipv=config.stockfish_verification_multipv,
                illegal_move_cp_loss=config.stockfish_illegal_move_cp_loss,
            )
        except Exception as exc:
            logger.warning(
                "Failed Stockfish move verification for FEN %s: %s",
                sample.fen,
                exc,
            )
            return None

    async def _run_shared_stockfish_task(sample: RolloutSample) -> tuple[str, Any | None]:
        try:
            if stockfish_hint_extractor is None or sample.fen is None:
                return "", None
            predicted_answer_text = _answer_text_after_think_blocks(sample.response_text)
            return await _stockfish_analyze_and_verify_async(
                stockfish_client=stockfish_hint_extractor,
                fen=sample.fen,
                predicted_text=predicted_answer_text,
                hint_depth=config.stockfish_depth,
                hint_multipv=config.stockfish_multipv,
                verification_depth=config.stockfish_verification_depth,
                verification_multipv=config.stockfish_verification_multipv,
                illegal_move_cp_loss=config.stockfish_illegal_move_cp_loss,
                mode=config.stockfish_shared_eval_mode,
            )
        except Exception as exc:
            logger.warning(
                "Failed shared Stockfish analyze+verify for FEN %s: %s",
                sample.fen,
                exc,
            )
            return "", None

    for i, sample in enumerate(records):
        solution_idx = select_solution_idx(
            rewards=rewards,
            sample_idx=i,
            success_reward_threshold=config.success_reward_threshold,
            dont_reprompt_on_self_success=config.dont_reprompt_on_self_success,
        )

        solution_text: str | None = None
        if solution_idx is not None:
            solution_text = records[solution_idx].response_text
            if config.remove_thinking_from_demonstration:
                solution_text = maybe_strip_thinking(solution_text)
        solution_texts[i] = solution_text

        should_use_hints = (
            config.enable_stockfish_hints
            and stockfish_hint_extractor is not None
            and sample.fen is not None
            and (not config.stockfish_hints_only_without_solution or not bool(solution_text))
        )
        stockfish_verification_candidate = (
            config.enable_stockfish_move_verification
            and stockfish_hint_extractor is not None
            and sample.fen is not None
        )
        stockfish_verification_scheduled = (
            stockfish_verification_candidate
            and _should_run_stockfish_verification(
                sample=sample,
                sample_rate=config.stockfish_verification_sample_rate,
            )
        )
        verification_candidate_by_idx[i] = stockfish_verification_candidate
        verification_scheduled_by_idx[i] = stockfish_verification_scheduled
        should_share_stockfish_eval = (
            should_use_hints
            and stockfish_verification_scheduled
            and config.stockfish_shared_hint_and_verification_eval
        )

        if should_share_stockfish_eval:
            shared_stockfish_task_by_idx[i] = asyncio.create_task(
                _run_shared_stockfish_task(sample)
            )
        elif should_use_hints and sample.fen is not None:
            hint_task = hint_task_by_fen.get(sample.fen)
            if hint_task is None:
                hint_task = asyncio.create_task(_run_stockfish_hint_task(sample.fen))
                hint_task_by_fen[sample.fen] = hint_task
            hint_task_by_idx[i] = hint_task

        if should_share_stockfish_eval:
            continue

        if stockfish_verification_scheduled:
            verification_task_by_idx[i] = asyncio.create_task(
                _run_stockfish_verification_task(sample)
            )

    if hint_task_by_fen:
        await asyncio.gather(*hint_task_by_fen.values())
        for fen, task in hint_task_by_fen.items():
            stockfish_hints_by_fen[fen] = task.result()

    verification_tasks = [task for task in verification_task_by_idx if task is not None]
    if verification_tasks:
        await asyncio.gather(*verification_tasks)
    shared_stockfish_tasks = [task for task in shared_stockfish_task_by_idx if task is not None]
    if shared_stockfish_tasks:
        await asyncio.gather(*shared_stockfish_tasks)

    for i, sample in enumerate(records):
        solution_text = solution_texts[i]
        stockfish_hints_text = None
        shared_task = shared_stockfish_task_by_idx[i]
        shared_verification: Any | None = None
        if shared_task is not None:
            shared_hint_text, shared_verification = shared_task.result()
            if shared_hint_text:
                stockfish_hints_text = shared_hint_text
        hint_task = hint_task_by_idx[i]
        if hint_task is not None and stockfish_hints_text is None:
            hint_text = hint_task.result()
            if hint_text:
                stockfish_hints_text = hint_text
        if sample.fen is not None and stockfish_hints_text:
            stockfish_hints_by_fen[sample.fen] = stockfish_hints_text

        stockfish_hint_used = (
            config.enable_stockfish_hints
            and stockfish_hints_text is not None
            and (not config.stockfish_hints_only_without_solution or not bool(solution_text))
        )

        combined_feedback_text = sample.feedback_text
        stockfish_verified = False
        stockfish_move_is_legal = False
        stockfish_cp_loss = 0.0
        stockfish_cp_loss_estimated = False
        stockfish_feedback_used = False
        stockfish_best_move_uci: str | None = None
        stockfish_predicted_move_uci: str | None = None
        stockfish_best_move_match = False
        stockfish_verification_candidate = verification_candidate_by_idx[i]
        stockfish_verification_scheduled = verification_scheduled_by_idx[i]
        verification_task = verification_task_by_idx[i]
        verification_obj: Any | None = None
        if shared_verification is not None:
            verification_obj = shared_verification
            stockfish_verified = True
            stockfish_move_is_legal = shared_verification.move_is_legal
            stockfish_cp_loss = float(shared_verification.cp_loss)
            stockfish_cp_loss_estimated = shared_verification.cp_loss_source != "centipawn"
            stockfish_best_move_uci = getattr(shared_verification, "best_move_uci", None)
            stockfish_predicted_move_uci = getattr(shared_verification, "predicted_move_uci", None)
            should_add_stockfish_feedback = (
                config.include_stockfish_move_feedback
                and stockfish_cp_loss >= config.stockfish_feedback_cp_loss_threshold
            )
            if should_add_stockfish_feedback:
                if combined_feedback_text:
                    combined_feedback_text = (
                        f"{combined_feedback_text}\n\n{shared_verification.feedback_text}"
                    )
                else:
                    combined_feedback_text = shared_verification.feedback_text
                stockfish_feedback_used = True
            if stockfish_best_move_uci and stockfish_predicted_move_uci:
                stockfish_best_move_match = (
                    stockfish_predicted_move_uci == stockfish_best_move_uci
                )
        elif verification_task is not None:
            verification = verification_task.result()
            if verification is not None:
                verification_obj = verification
                stockfish_verified = True
                stockfish_move_is_legal = verification.move_is_legal
                stockfish_cp_loss = float(verification.cp_loss)
                stockfish_cp_loss_estimated = verification.cp_loss_source != "centipawn"
                stockfish_best_move_uci = getattr(verification, "best_move_uci", None)
                stockfish_predicted_move_uci = getattr(verification, "predicted_move_uci", None)
                should_add_stockfish_feedback = (
                    config.include_stockfish_move_feedback
                    and stockfish_cp_loss >= config.stockfish_feedback_cp_loss_threshold
                )
                if should_add_stockfish_feedback:
                    if combined_feedback_text:
                        combined_feedback_text = (
                            f"{combined_feedback_text}\n\n{verification.feedback_text}"
                        )
                    else:
                        combined_feedback_text = verification.feedback_text
                    stockfish_feedback_used = True
                if stockfish_best_move_uci and stockfish_predicted_move_uci:
                    stockfish_best_move_match = (
                        stockfish_predicted_move_uci == stockfish_best_move_uci
                    )

        teacher_messages, feedback_used, used_reprompt = build_teacher_messages(
            prompt_messages=sample.prompt_messages,
            solution_text=solution_text,
            feedback_text=combined_feedback_text,
            reprompt_template=config.reprompt_template,
            solution_template=config.solution_template,
            feedback_template=config.feedback_template,
            include_environment_feedback=config.include_environment_feedback,
            environment_feedback_only_without_solution=config.environment_feedback_only_without_solution,
            hints_text=stockfish_hints_text,
            hints_template=config.stockfish_hints_template,
            include_hints=config.enable_stockfish_hints,
            hints_only_without_solution=config.stockfish_hints_only_without_solution,
        )

        if (
            debug_examples_sink is not None
            and debug_examples_limit > 0
            and len(debug_examples_sink) < debug_examples_limit
        ):
            expected_answer = sample.expected_answer or stockfish_best_move_uci
            debug_examples_sink.append(
                {
                    "group_index": group_index if group_index is not None else -1,
                    "sample_index_in_group": i,
                    "reward": float(sample.reward),
                    "fen": sample.fen,
                    "prompt": _truncate_debug_text(
                        _prompt_messages_to_text(sample.prompt_messages),
                        debug_examples_max_text_chars,
                    ),
                    "model_output": _truncate_debug_text(
                        sample.response_text,
                        debug_examples_max_text_chars,
                    ),
                    "expected_answer": _truncate_debug_text(
                        expected_answer,
                        debug_examples_max_text_chars,
                    ),
                    "teacher_solution": _truncate_debug_text(
                        solution_text,
                        debug_examples_max_text_chars,
                    ),
                    "combined_feedback": _truncate_debug_text(
                        combined_feedback_text,
                        debug_examples_max_text_chars,
                    ),
                    "stockfish_hint": _truncate_debug_text(
                        stockfish_hints_text,
                        debug_examples_max_text_chars,
                    ),
                    "used_reprompt": bool(used_reprompt),
                    "feedback_used": bool(feedback_used),
                    "stockfish_hint_used": bool(stockfish_hint_used),
                    "stockfish_verification_scheduled": bool(stockfish_verification_scheduled),
                    "stockfish_verified": bool(stockfish_verified),
                    "stockfish_predicted_move": stockfish_predicted_move_uci,
                    "stockfish_best_move": stockfish_best_move_uci,
                    "stockfish_move_is_legal": bool(stockfish_move_is_legal),
                    "stockfish_best_move_match": bool(stockfish_best_move_match),
                    "stockfish_cp_loss": float(stockfish_cp_loss),
                    "stockfish_cp_loss_estimated": bool(stockfish_cp_loss_estimated),
                    "stockfish_feedback_text": _truncate_debug_text(
                        getattr(verification_obj, "feedback_text", None),
                        debug_examples_max_text_chars,
                    ),
                }
            )

        async def _build_datum(
            rollout_sample: RolloutSample = sample,
            grpo_advantage: float = grpo_advantages[i],
            local_teacher_messages: list[renderers.Message] = teacher_messages,
            local_feedback_used: bool = feedback_used,
            local_used_reprompt: bool = used_reprompt,
            local_stockfish_hint_used: bool = stockfish_hint_used,
            local_stockfish_verified: bool = stockfish_verified,
            local_stockfish_move_is_legal: bool = stockfish_move_is_legal,
            local_stockfish_best_move_match: bool = stockfish_best_move_match,
            local_stockfish_cp_loss: float = stockfish_cp_loss,
            local_stockfish_cp_loss_estimated: bool = stockfish_cp_loss_estimated,
            local_stockfish_feedback_used: bool = stockfish_feedback_used,
            local_stockfish_verification_candidate: bool = stockfish_verification_candidate,
            local_stockfish_verification_scheduled: bool = stockfish_verification_scheduled,
        ) -> tuple[
            tinker.Datum,
            list[float],
            bool,
            bool,
            bool,
            bool,
            bool,
            bool,
            float,
            bool,
            bool,
            bool,
            bool,
            int,
            int,
        ]:
            advantages, topk_overlap_count, topk_total_count, rollout_logprobs = await _compute_sample_advantages(
                sample=rollout_sample,
                teacher_messages=local_teacher_messages,
                used_reprompt=local_used_reprompt,
                current_sampling_client=current_sampling_client,
                reference_sampling_client=reference_sampling_client,
                ema_teacher_sampling_clients=ema_teacher_sampling_clients,
                renderer=renderer,
                teacher_regularization=config.teacher_regularization,
                teacher_mix_alpha=config.teacher_mix_alpha,
                full_logit_distillation=config.full_logit_distillation,
                distillation_topk=config.distillation_topk,
                distillation_add_tail=config.distillation_add_tail,
                max_reprompt_tokens=config.max_reprompt_tokens,
                reprompt_truncation=config.reprompt_truncation,
                semaphore=teacher_logprob_semaphore,
            )
            if config.advantage_mode == "sequence" and advantages:
                sequence_advantage = float(sum(advantages) / len(advantages))
                advantages = [sequence_advantage] * len(advantages)
            if config.grpo_mix_lambda != 0.0:
                advantages = [
                    float((1.0 - config.grpo_mix_lambda) * advantage + (config.grpo_mix_lambda * grpo_advantage))
                    for advantage in advantages
                ]
            datum = build_sdpo_datum(
                prompt_tokens=rollout_sample.prompt_tokens,
                completion_tokens=rollout_sample.completion_tokens,
                rollout_logprobs=rollout_logprobs,
                advantages=advantages,
            )
            return (
                datum,
                advantages,
                local_feedback_used,
                local_used_reprompt,
                local_stockfish_hint_used,
                local_stockfish_verified,
                local_stockfish_move_is_legal,
                local_stockfish_best_move_match,
                local_stockfish_cp_loss,
                local_stockfish_cp_loss_estimated,
                local_stockfish_feedback_used,
                local_stockfish_verification_candidate,
                local_stockfish_verification_scheduled,
                topk_overlap_count,
                topk_total_count,
            )

        datum_tasks.append(asyncio.create_task(_build_datum()))

    data: list[tinker.Datum] = []
    for task in datum_tasks:
        (
            datum,
            advantages,
            feedback_used,
            used_reprompt,
            stockfish_hint_used,
            stockfish_verified,
            stockfish_move_is_legal,
            stockfish_best_move_match,
            stockfish_cp_loss,
            stockfish_cp_loss_estimated,
            stockfish_feedback_used,
            stockfish_verification_candidate,
            stockfish_verification_scheduled,
            topk_overlap_count,
            topk_total_count,
        ) = await task
        data.append(datum)

        if feedback_used:
            stats.feedback_used_samples += 1
        if stockfish_hint_used:
            stats.stockfish_hint_used_samples += 1
        if stockfish_verification_candidate:
            stats.stockfish_verification_candidate_samples += 1
        if stockfish_verification_scheduled:
            stats.stockfish_verification_scheduled_samples += 1
        if stockfish_verified:
            stats.stockfish_verified_samples += 1
            stats.stockfish_cp_loss_sum += stockfish_cp_loss
            if stockfish_best_move_match:
                stats.stockfish_best_move_samples += 1
            if stockfish_cp_loss_estimated:
                stats.stockfish_estimated_cp_loss_samples += 1
        if stockfish_move_is_legal:
            stats.stockfish_legal_move_samples += 1
        if stockfish_feedback_used:
            stats.stockfish_feedback_samples += 1
        if used_reprompt:
            stats.reprompt_samples += 1
        if all(abs(v) < 1e-12 for v in advantages):
            stats.zero_adv_samples += 1

        stats.advantage_sum += float(sum(advantages))
        stats.advantage_abs_sum += float(sum(abs(v) for v in advantages))
        stats.advantage_count += len(advantages)
        stats.topk_overlap_count += topk_overlap_count
        stats.topk_total_count += topk_total_count

    stats.feedback_available_samples = sum(1 for record in records if record.feedback_text is not None)
    stats.stockfish_hint_available_samples = sum(
        1
        for record in records
        if (
            config.enable_stockfish_hints
            and stockfish_hint_extractor is not None
            and record.fen is not None
            and bool(stockfish_hints_by_fen.get(record.fen))
        )
    )
    return data, stats


async def run_sdpo_batch_update(
    config: Config,
    training_client: tinker.TrainingClient,
    current_sampling_client: tinker.SamplingClient,
    reference_sampling_client: tinker.SamplingClient | None,
    ema_teacher_sampling_clients: Sequence[tinker.SamplingClient] | None,
    renderer: renderers.Renderer,
    tokenizer: Tokenizer,
    states_by_group: Sequence[Sequence[Mapping[str, Any]]],
    stockfish_hint_extractor: Any | None,
    debug_examples_sink: list[dict[str, Any]] | None = None,
    debug_examples_limit: int = 0,
    debug_examples_max_text_chars: int = 0,
) -> dict[str, Any]:
    if not 0.0 <= config.grpo_mix_lambda <= 1.0:
        raise ValueError(f"grpo_mix_lambda must be in [0, 1], got {config.grpo_mix_lambda}")
    if config.updates_per_batch != 1:
        raise ValueError(
            "SDPO training is on-policy only; updates_per_batch must be exactly 1 "
            f"(got {config.updates_per_batch})"
        )
    if config.teacher_regularization == "ema" and not 0.0 < config.teacher_mix_alpha <= 1.0:
        raise ValueError(
            f"teacher_mix_alpha must be in (0, 1] for ema regularization, got {config.teacher_mix_alpha}"
        )
    if not 0.0 <= config.stockfish_verification_sample_rate <= 1.0:
        raise ValueError(
            "stockfish_verification_sample_rate must be in [0, 1], got "
            f"{config.stockfish_verification_sample_rate}"
        )
    if config.stockfish_verification_multipv <= 0:
        raise ValueError(
            "stockfish_verification_multipv must be >= 1, got "
            f"{config.stockfish_verification_multipv}"
        )
    if config.stockfish_shared_eval_mode not in {"single", "two_pass"}:
        raise ValueError(
            "stockfish_shared_eval_mode must be 'single' or 'two_pass', got "
            f"{config.stockfish_shared_eval_mode}"
        )
    if debug_examples_limit < 0:
        raise ValueError(f"debug_examples_limit must be >= 0, got {debug_examples_limit}")
    if config.full_logit_distillation and (config.distillation_topk is None or config.distillation_topk <= 0):
        raise ValueError(
            "full_logit_distillation=True requires distillation_topk to be a positive integer"
        )

    feedback_keys = _parse_feedback_keys(config.feedback_keys_csv)
    teacher_logprob_semaphore = None
    if config.max_concurrent_teacher_logprobs > 0:
        teacher_logprob_semaphore = asyncio.Semaphore(config.max_concurrent_teacher_logprobs)

    all_data: list[tinker.Datum] = []
    total_stats = GroupSdpoStats()

    for i_group, states in enumerate(states_by_group):
        group_data, group_stats = await build_group_sdpo_datums(
            states=states,
            config=config,
            current_sampling_client=current_sampling_client,
            reference_sampling_client=reference_sampling_client,
            ema_teacher_sampling_clients=ema_teacher_sampling_clients,
            renderer=renderer,
            tokenizer=tokenizer,
            feedback_keys=feedback_keys,
            teacher_logprob_semaphore=teacher_logprob_semaphore,
            stockfish_hint_extractor=stockfish_hint_extractor,
            group_index=i_group,
            debug_examples_sink=debug_examples_sink,
            debug_examples_limit=debug_examples_limit,
            debug_examples_max_text_chars=debug_examples_max_text_chars,
        )
        all_data.extend(group_data)

        total_stats.num_samples += group_stats.num_samples
        total_stats.success_samples += group_stats.success_samples
        total_stats.success_groups += group_stats.success_groups
        total_stats.feedback_available_samples += group_stats.feedback_available_samples
        total_stats.feedback_used_samples += group_stats.feedback_used_samples
        total_stats.stockfish_hint_available_samples += group_stats.stockfish_hint_available_samples
        total_stats.stockfish_hint_used_samples += group_stats.stockfish_hint_used_samples
        total_stats.stockfish_verification_candidate_samples += (
            group_stats.stockfish_verification_candidate_samples
        )
        total_stats.stockfish_verification_scheduled_samples += (
            group_stats.stockfish_verification_scheduled_samples
        )
        total_stats.stockfish_verified_samples += group_stats.stockfish_verified_samples
        total_stats.stockfish_legal_move_samples += group_stats.stockfish_legal_move_samples
        total_stats.stockfish_best_move_samples += group_stats.stockfish_best_move_samples
        total_stats.stockfish_cp_loss_sum += group_stats.stockfish_cp_loss_sum
        total_stats.stockfish_estimated_cp_loss_samples += (
            group_stats.stockfish_estimated_cp_loss_samples
        )
        total_stats.stockfish_feedback_samples += group_stats.stockfish_feedback_samples
        total_stats.reprompt_samples += group_stats.reprompt_samples
        total_stats.zero_adv_samples += group_stats.zero_adv_samples
        total_stats.skipped_samples += group_stats.skipped_samples
        total_stats.advantage_sum += group_stats.advantage_sum
        total_stats.advantage_abs_sum += group_stats.advantage_abs_sum
        total_stats.advantage_count += group_stats.advantage_count
        total_stats.topk_overlap_count += group_stats.topk_overlap_count
        total_stats.topk_total_count += group_stats.topk_total_count
        for key, value in group_stats.env_metric_sums.items():
            total_stats.env_metric_sums[key] = total_stats.env_metric_sums.get(key, 0.0) + value
        for key, value in group_stats.env_metric_counts.items():
            total_stats.env_metric_counts[key] = total_stats.env_metric_counts.get(key, 0) + value

    metrics: dict[str, Any] = {
        "sdpo/num_datums": len(all_data),
        "sdpo/num_groups": len(states_by_group),
        "sdpo/num_samples": total_stats.num_samples,
        "sdpo/num_skipped_samples": total_stats.skipped_samples,
        "sdpo/grpo_mix_lambda": config.grpo_mix_lambda,
        "sdpo/full_logit_distillation": float(config.full_logit_distillation),
        "sdpo/updates_per_batch": float(config.updates_per_batch),
        "sdpo/student_max_thinking_tokens": float(config.student_max_thinking_tokens),
        "sdpo/stockfish_hints_enabled": float(config.enable_stockfish_hints),
        "sdpo/stockfish_move_verification_enabled": float(config.enable_stockfish_move_verification),
        "sdpo/stockfish_verification_sample_rate": float(config.stockfish_verification_sample_rate),
        "sdpo/stockfish_verification_depth": float(config.stockfish_verification_depth),
        "sdpo/stockfish_verification_multipv": float(config.stockfish_verification_multipv),
        "sdpo/stockfish_shared_eval_enabled": float(
            config.stockfish_shared_hint_and_verification_eval
        ),
        "sdpo/stockfish_shared_eval_mode_two_pass": float(
            config.stockfish_shared_eval_mode == "two_pass"
        ),
    }

    if len(states_by_group) > 0:
        metrics["sdpo/success_group_fraction"] = total_stats.success_groups / len(states_by_group)
    else:
        metrics["sdpo/success_group_fraction"] = 0.0

    if total_stats.num_samples > 0:
        metrics["sdpo/success_sample_fraction"] = total_stats.success_samples / total_stats.num_samples
        metrics["sdpo/feedback_available_fraction"] = (
            total_stats.feedback_available_samples / total_stats.num_samples
        )
        metrics["sdpo/feedback_used_fraction"] = (
            total_stats.feedback_used_samples / total_stats.num_samples
        )
        metrics["sdpo/stockfish_hint_available_fraction"] = (
            total_stats.stockfish_hint_available_samples / total_stats.num_samples
        )
        metrics["sdpo/stockfish_hint_used_fraction"] = (
            total_stats.stockfish_hint_used_samples / total_stats.num_samples
        )
        metrics["sdpo/stockfish_verification_candidate_fraction"] = (
            total_stats.stockfish_verification_candidate_samples / total_stats.num_samples
        )
        metrics["sdpo/stockfish_verification_scheduled_fraction"] = (
            total_stats.stockfish_verification_scheduled_samples / total_stats.num_samples
        )
        metrics["sdpo/stockfish_verified_fraction"] = (
            total_stats.stockfish_verified_samples / total_stats.num_samples
        )
        metrics["sdpo/stockfish_legal_move_fraction"] = (
            total_stats.stockfish_legal_move_samples / total_stats.num_samples
        )
        metrics["sdpo/stockfish_best_move_fraction"] = (
            total_stats.stockfish_best_move_samples / total_stats.num_samples
        )
        metrics["sdpo/stockfish_feedback_fraction"] = (
            total_stats.stockfish_feedback_samples / total_stats.num_samples
        )
        metrics["sdpo/stockfish_estimated_cp_loss_fraction"] = (
            total_stats.stockfish_estimated_cp_loss_samples / total_stats.num_samples
        )
        metrics["sdpo/reprompt_sample_fraction"] = total_stats.reprompt_samples / total_stats.num_samples
    else:
        metrics["sdpo/success_sample_fraction"] = 0.0
        metrics["sdpo/feedback_available_fraction"] = 0.0
        metrics["sdpo/feedback_used_fraction"] = 0.0
        metrics["sdpo/stockfish_hint_available_fraction"] = 0.0
        metrics["sdpo/stockfish_hint_used_fraction"] = 0.0
        metrics["sdpo/stockfish_verification_candidate_fraction"] = 0.0
        metrics["sdpo/stockfish_verification_scheduled_fraction"] = 0.0
        metrics["sdpo/stockfish_verified_fraction"] = 0.0
        metrics["sdpo/stockfish_legal_move_fraction"] = 0.0
        metrics["sdpo/stockfish_best_move_fraction"] = 0.0
        metrics["sdpo/stockfish_feedback_fraction"] = 0.0
        metrics["sdpo/stockfish_estimated_cp_loss_fraction"] = 0.0
        metrics["sdpo/reprompt_sample_fraction"] = 0.0

    metrics["sdpo/num_zero_adv_samples"] = total_stats.zero_adv_samples
    if total_stats.stockfish_verified_samples > 0:
        metrics["sdpo/stockfish_avg_cp_loss"] = (
            total_stats.stockfish_cp_loss_sum / total_stats.stockfish_verified_samples
        )
        metrics["sdpo/stockfish_accuracy"] = (
            total_stats.stockfish_best_move_samples / total_stats.stockfish_verified_samples
        )
    else:
        metrics["sdpo/stockfish_avg_cp_loss"] = 0.0
        metrics["sdpo/stockfish_accuracy"] = 0.0
    metrics["sdpo/stockfish_acpl"] = metrics["sdpo/stockfish_avg_cp_loss"]
    metrics["chess/acc"] = metrics["sdpo/stockfish_accuracy"]
    metrics["chess/acpl"] = metrics["sdpo/stockfish_acpl"]

    if total_stats.advantage_count > 0:
        metrics["sdpo/mean_advantage"] = total_stats.advantage_sum / total_stats.advantage_count
        metrics["sdpo/mean_abs_advantage"] = (
            total_stats.advantage_abs_sum / total_stats.advantage_count
        )
    else:
        metrics["sdpo/mean_advantage"] = 0.0
        metrics["sdpo/mean_abs_advantage"] = 0.0
    if total_stats.topk_total_count > 0:
        metrics["sdpo/topk_overlap_fraction"] = total_stats.topk_overlap_count / total_stats.topk_total_count
    else:
        metrics["sdpo/topk_overlap_fraction"] = 0.0

    for key, total_value in sorted(total_stats.env_metric_sums.items()):
        count = total_stats.env_metric_counts.get(key, 0)
        if count > 0:
            metrics[f"env/{key}"] = total_value / count

    if all_data:
        with timed("train", metrics):
            update_metrics: dict[str, Any] = {}
            _ = await rl_train.train_step(
                data_D=all_data,
                training_client=training_client,
                learning_rate=config.learning_rate,
                num_substeps=config.num_substeps,
                loss_fn=config.loss_fn,
                loss_fn_config=config.loss_fn_config,
                metrics=update_metrics,
            )
            metrics.update(update_metrics)

    return metrics


async def _run_verifiers_group_rollout(
    builder: Any,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    tokenizer: Tokenizer,
    group_size: int,
    max_tokens: int,
    student_max_thinking_tokens: int,
    temperature: float,
    max_concurrent_generation: int,
    max_concurrent_scoring: int,
) -> list[Mapping[str, Any]]:
    if not hasattr(builder, "vf_env") or not hasattr(builder, "get_rollout_inputs"):
        raise TypeError(
            "SDPO verifiers recipe requires builders with vf_env and get_rollout_inputs()"
        )

    # Import lazily so importing this module does not require verifiers/openai extras.
    from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient

    rollout_inputs = builder.get_rollout_inputs(group_size)
    # Some verifiers versions expect semaphores (not None) for run_group.
    # For non-positive limits, default to group_size-bound concurrency.
    effective_max_concurrent_generation = (
        max_concurrent_generation if max_concurrent_generation > 0 else group_size
    )
    effective_max_concurrent_scoring = (
        max_concurrent_scoring if max_concurrent_scoring > 0 else group_size
    )
    gen_sem = asyncio.Semaphore(max(1, effective_max_concurrent_generation))
    score_sem = asyncio.Semaphore(max(1, effective_max_concurrent_scoring))

    client = TinkerAsyncOpenAIClient(sampling_client, renderer, tokenizer)
    gen_sampling_args: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if student_max_thinking_tokens > 0:
        gen_sampling_args["max_thinking_tokens"] = student_max_thinking_tokens

    run_group_fn = builder.vf_env.run_group
    run_group_sig = inspect.signature(run_group_fn)
    run_group_params = run_group_sig.parameters
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in run_group_params.values()
    )

    run_group_kwargs: dict[str, Any] = {
        "group_inputs": rollout_inputs,
        "client": client,
        "model": "tinker",
    }
    if "gen_sem" in run_group_params or accepts_var_kwargs:
        run_group_kwargs["gen_sem"] = gen_sem
    if "score_sem" in run_group_params or accepts_var_kwargs:
        run_group_kwargs["score_sem"] = score_sem
    if "gen_sampling_args" in run_group_params or accepts_var_kwargs:
        run_group_kwargs["gen_sampling_args"] = gen_sampling_args
    if "sampling_args" in run_group_params or accepts_var_kwargs:
        run_group_kwargs["sampling_args"] = gen_sampling_args

    states = await run_group_fn(**run_group_kwargs)

    if not isinstance(states, list):
        raise ValueError("vf_env.run_group did not return a list of states")
    return [state for state in states if isinstance(state, Mapping)]


async def _get_or_create_reference_sampling_client(
    config: Config,
    service_client: tinker.ServiceClient,
    training_client: tinker.TrainingClient,
    start_batch: int,
) -> tuple[tinker.SamplingClient | None, str | None]:
    if config.teacher_regularization != "trust_region":
        return None, None

    path_file = os.path.join(config.log_path, _REFERENCE_SAMPLER_FILE)
    if os.path.exists(path_file):
        with open(path_file, "r") as f:
            existing_path = f.read().strip()
        if existing_path:
            logger.info("Reusing SDPO reference sampler path from %s", path_file)
            return (
                service_client.create_sampling_client(
                    base_model=config.model_name,
                    model_path=existing_path,
                ),
                existing_path,
            )

    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="sdpo_reference_sampler",
        log_path=config.log_path,
        loop_state={"batch": start_batch, "tag": "sdpo_reference_sampler"},
        kind="sampler",
        ttl_seconds=config.ttl_seconds,
    )
    sampler_path = path_dict["sampler_path"]

    with open(path_file, "w") as f:
        f.write(sampler_path)

    logger.info("Created SDPO reference sampler at %s", sampler_path)
    return (
        service_client.create_sampling_client(
            base_model=config.model_name,
            model_path=sampler_path,
        ),
        sampler_path,
    )


async def main(config: Config):
    if config.teacher_regularization == "ema" and config.ema_teacher_history <= 0:
        raise ValueError(
            f"ema_teacher_history must be >= 1 when teacher_regularization='ema' (got {config.ema_teacher_history})"
        )
    if config.stockfish_num_workers <= 0:
        raise ValueError(f"stockfish_num_workers must be >= 1, got {config.stockfish_num_workers}")
    if config.stockfish_verification_multipv <= 0:
        raise ValueError(
            "stockfish_verification_multipv must be >= 1, got "
            f"{config.stockfish_verification_multipv}"
        )
    if config.stockfish_shared_eval_mode not in {"single", "two_pass"}:
        raise ValueError(
            "stockfish_shared_eval_mode must be 'single' or 'two_pass', got "
            f"{config.stockfish_shared_eval_mode}"
        )
    if config.student_max_thinking_tokens < 0:
        raise ValueError(
            "student_max_thinking_tokens must be >= 0, got "
            f"{config.student_max_thinking_tokens}"
        )
    if config.debug_examples_every_n_steps < 0:
        raise ValueError(
            f"debug_examples_every_n_steps must be >= 0, got {config.debug_examples_every_n_steps}"
        )
    if config.debug_examples_per_step <= 0:
        raise ValueError(f"debug_examples_per_step must be >= 1, got {config.debug_examples_per_step}")
    if config.debug_examples_max_text_chars <= 0:
        raise ValueError(
            "debug_examples_max_text_chars must be >= 1, got "
            f"{config.debug_examples_max_text_chars}"
        )
    if not config.debug_examples_file_name.strip():
        raise ValueError("debug_examples_file_name must be non-empty")
    if config.updates_per_batch != 1:
        raise ValueError(
            "SDPO training is on-policy only; updates_per_batch must be exactly 1 "
            f"(got {config.updates_per_batch})"
        )

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
    )

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    start_batch = resume_info["batch"] if resume_info else 0

    service_client = tinker.ServiceClient(base_url=config.base_url)
    if resume_info:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume_info["state_path"]
        )
        logger.info("Resumed SDPO training from %s", resume_info["state_path"])
    elif config.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint_path
        )
        logger.info("Loaded SDPO weights from %s", config.load_checkpoint_path)
    else:
        training_client = await service_client.create_lora_training_client_async(
            config.model_name,
            rank=config.lora_rank,
        )

    tokenizer = training_client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    dataset, _maybe_test_dataset = await config.dataset_builder()
    num_batches = len(dataset)
    logger.info("Will run SDPO for %d batches", num_batches)

    reference_sampling_client, reference_sampler_path = await _get_or_create_reference_sampling_client(
        config=config,
        service_client=service_client,
        training_client=training_client,
        start_batch=start_batch,
    )
    if reference_sampler_path:
        logger.info("SDPO reference sampler path: %s", reference_sampler_path)

    ema_teacher_clients: deque[tinker.SamplingClient] = deque(
        maxlen=max(1, config.ema_teacher_history)
    )
    stockfish_hint_extractor: Any | None = None
    persistent_stockfish_cache_dir = config.stockfish_persistent_cache_dir
    enable_stockfish_service = (
        config.enable_stockfish_hints or config.enable_stockfish_move_verification
    )
    if enable_stockfish_service and persistent_stockfish_cache_dir is None:
        persistent_stockfish_cache_dir = os.path.join(config.log_path, "stockfish_cache")
    try:
        if enable_stockfish_service:
            stockfish_hint_config = StockfishHintConfig(
                stockfish_path=config.stockfish_path,
                depth=config.stockfish_depth,
                multipv=config.stockfish_multipv,
                threads=config.stockfish_threads,
                hash_mb=config.stockfish_hash_mb,
                wdl_model=config.stockfish_wdl_model,
                max_pv_plies=config.stockfish_max_pv_plies,
                max_good_moves=config.stockfish_hint_max_good_moves,
                max_bad_moves=config.stockfish_hint_max_bad_moves,
                bad_move_threshold=config.stockfish_hint_bad_move_threshold,
                include_fen_decode=config.stockfish_include_fen_decode,
                include_ascii_board=config.stockfish_include_ascii_board,
                include_search_stats=config.stockfish_include_search_stats,
                analysis_time_limit_sec=config.stockfish_analysis_time_limit_sec,
                engine_max_retries=config.stockfish_engine_max_retries,
                max_root_cache_entries=config.stockfish_max_root_cache_entries,
                max_move_cache_entries=config.stockfish_max_move_cache_entries,
                max_verification_cache_entries=config.stockfish_max_verification_cache_entries,
                max_piece_pressure_items=config.stockfish_max_piece_pressure_items,
                max_weak_square_items=config.stockfish_max_weak_square_items,
                syzygy_path=config.stockfish_syzygy_path,
                syzygy_max_pieces=config.stockfish_syzygy_max_pieces,
                unknown_score_cp_loss=config.stockfish_unknown_score_cp_loss,
                persistent_cache_dir=persistent_stockfish_cache_dir,
            )
            stockfish_hint_extractor = StockfishHintPool(
                stockfish_hint_config,
                num_workers=config.stockfish_num_workers,
            )
            if config.enable_stockfish_hints and config.enable_stockfish_move_verification:
                logger.info(
                    "Stockfish hints and move verification enabled via %s "
                    "(%d worker(s), %d thread(s) per worker)",
                    config.stockfish_path,
                    config.stockfish_num_workers,
                    config.stockfish_threads,
                )
            elif config.enable_stockfish_hints:
                logger.info(
                    "Stockfish hints enabled via %s (%d worker(s), %d thread(s) per worker)",
                    config.stockfish_path,
                    config.stockfish_num_workers,
                    config.stockfish_threads,
                )
            else:
                logger.info(
                    "Stockfish move verification enabled via %s "
                    "(%d worker(s), %d thread(s) per worker)",
                    config.stockfish_path,
                    config.stockfish_num_workers,
                    config.stockfish_threads,
                )

        for i_batch in range(start_batch, num_batches):
            metrics: dict[str, Any] = {
                "progress/batch": i_batch,
                "progress/done_frac": (i_batch + 1) / num_batches,
                "optim/lr": config.learning_rate,
            }
            t_start = time.time()

            with timed("save_weights_and_get_sampling_client", metrics):
                current_sampling_client = await training_client.save_weights_and_get_sampling_client_async()
            if config.teacher_regularization == "ema":
                ema_teacher_clients.appendleft(current_sampling_client)

            env_group_builders = dataset.get_batch(i_batch)
            with timed("rollout", metrics):
                states_by_group = await rl_train.gather_with_progress(
                    (
                        _run_verifiers_group_rollout(
                            builder=builder,
                            sampling_client=current_sampling_client,
                            renderer=renderer,
                            tokenizer=tokenizer,
                            group_size=config.group_size,
                            max_tokens=config.max_tokens,
                            student_max_thinking_tokens=config.student_max_thinking_tokens,
                            temperature=config.temperature,
                            max_concurrent_generation=config.max_concurrent_generation,
                            max_concurrent_scoring=config.max_concurrent_scoring,
                        )
                        for builder in env_group_builders
                    ),
                    desc=f"SDPO sampling batch {i_batch}",
                )

            should_log_debug_examples = (
                config.debug_examples_every_n_steps > 0
                and (i_batch + 1) % config.debug_examples_every_n_steps == 0
            )
            debug_examples: list[dict[str, Any]] = []
            sdpo_metrics = await run_sdpo_batch_update(
                config=config,
                training_client=training_client,
                current_sampling_client=current_sampling_client,
                reference_sampling_client=reference_sampling_client,
                ema_teacher_sampling_clients=list(ema_teacher_clients),
                renderer=renderer,
                tokenizer=tokenizer,
                states_by_group=states_by_group,
                stockfish_hint_extractor=stockfish_hint_extractor,
                debug_examples_sink=debug_examples if should_log_debug_examples else None,
                debug_examples_limit=config.debug_examples_per_step,
                debug_examples_max_text_chars=config.debug_examples_max_text_chars,
            )
            metrics.update(sdpo_metrics)
            metrics["sdpo/debug_examples_logged"] = 0.0

            if should_log_debug_examples:
                debug_path = _append_debug_examples_jsonl(
                    log_path=config.log_path,
                    file_name=config.debug_examples_file_name,
                    batch_index=i_batch,
                    debug_examples=debug_examples,
                )
                metrics["sdpo/debug_examples_logged"] = float(len(debug_examples))
                logger.info(
                    "Wrote %d SDPO debug examples for batch %d to %s",
                    len(debug_examples),
                    i_batch,
                    debug_path,
                )
                if debug_examples:
                    long_text = _format_debug_examples_long_text(
                        batch_index=i_batch,
                        debug_examples=debug_examples,
                    )
                    ml_logger.log_long_text(
                        key=f"sdpo/debug_examples/batch_{i_batch:06d}",
                        text=long_text,
                        step=i_batch,
                    )

            if config.save_every > 0 and (i_batch + 1) % config.save_every == 0:
                with timed("save_checkpoint", metrics):
                    checkpoint_paths = await checkpoint_utils.save_checkpoint_async(
                        training_client=training_client,
                        name=f"{i_batch + 1:06d}",
                        log_path=config.log_path,
                        loop_state={"batch": i_batch + 1},
                        kind="both",
                        ttl_seconds=config.ttl_seconds,
                    )
                metrics.update(checkpoint_paths)

            metrics["time/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)

        if start_batch < num_batches:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name="final",
                log_path=config.log_path,
                loop_state={"batch": num_batches},
                kind="both",
                ttl_seconds=config.ttl_seconds,
            )
        logger.info("SDPO training completed successfully")
    finally:
        if stockfish_hint_extractor is not None:
            stockfish_hint_extractor.close()
        try:
            from tinker_cookbook.recipes.verifiers_rl.verifiers_env import get_vf_env
        except Exception:
            get_vf_env = None
        if get_vf_env is not None:
            try:
                vf_env = get_vf_env()
                teardown = getattr(vf_env, "_teardown", None) if vf_env is not None else None
                if callable(teardown):
                    await teardown()
            except Exception as exc:
                logger.warning("Failed to teardown verifiers environment: %s", exc)
        ml_logger.close()
