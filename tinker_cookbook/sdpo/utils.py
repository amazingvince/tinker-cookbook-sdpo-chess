from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import tinker
import torch
from tinker import TensorData

from tinker_cookbook import renderers

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def maybe_strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model responses."""
    return _THINK_RE.sub("", text).strip()


def _normalize_feedback_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (list, tuple)):
        parts = [_normalize_feedback_value(v) for v in value]
        non_empty_parts = [p for p in parts if p]
        if not non_empty_parts:
            return None
        return "\n".join(non_empty_parts)
    if isinstance(value, Mapping):
        preferred_keys = (
            "feedback",
            "error",
            "errors",
            "judge_feedback",
            "message",
            "messages",
            "detail",
            "details",
            "text",
        )
        for key in preferred_keys:
            if key in value:
                maybe = _normalize_feedback_value(value[key])
                if maybe:
                    return maybe
        try:
            as_json = json.dumps(value, sort_keys=True, ensure_ascii=True)
        except TypeError:
            as_json = str(value)
        as_json = as_json.strip()
        return as_json or None

    as_text = str(value).strip()
    return as_text or None


def extract_feedback_text(state: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    """
    Extract feedback using deterministic key and container precedence.

    Precedence order per key:
    1) state top-level
    2) state[metrics], state[info], state[extra_info], state[reward_extra_info], state[metadata]
    3) trajectory steps in reverse order, checking step, step[logs], step[metrics], step[info]
    """
    containers: list[Mapping[str, Any]] = [state]

    for nested_key in ("metrics", "info", "extra_info", "reward_extra_info", "metadata"):
        nested = state.get(nested_key)
        if isinstance(nested, Mapping):
            containers.append(nested)

    trajectory = state.get("trajectory")
    if isinstance(trajectory, list):
        for step in reversed(trajectory):
            if not isinstance(step, Mapping):
                continue
            containers.append(step)
            for nested_key in ("logs", "metrics", "info"):
                nested = step.get(nested_key)
                if isinstance(nested, Mapping):
                    containers.append(nested)

    for key in keys:
        for container in containers:
            if key in container:
                maybe = _normalize_feedback_value(container[key])
                if maybe:
                    return maybe

    return None


def select_solution_idx(
    rewards: Sequence[float],
    sample_idx: int,
    success_reward_threshold: float,
    dont_reprompt_on_self_success: bool,
) -> int | None:
    """Select a successful rollout index from the same group, if one exists."""
    successful_indices = [i for i, reward in enumerate(rewards) if reward >= success_reward_threshold]
    if dont_reprompt_on_self_success:
        successful_indices = [i for i in successful_indices if i != sample_idx]
    if not successful_indices:
        return None
    return successful_indices[0]


def build_teacher_messages(
    prompt_messages: Sequence[renderers.Message],
    solution_text: str | None,
    feedback_text: str | None,
    reprompt_template: str,
    solution_template: str,
    feedback_template: str,
    include_environment_feedback: bool,
    environment_feedback_only_without_solution: bool,
    hints_text: str | None = None,
    hints_template: str = "\nPosition hints:\n\n{stockfish_hints}\n\n",
    include_hints: bool = False,
    hints_only_without_solution: bool = False,
) -> tuple[list[renderers.Message], bool, bool]:
    """
    Build teacher messages from the original prompt and optional solution/feedback.

    Returns:
        (messages, feedback_used, used_reprompt)
    """
    if not prompt_messages:
        raise ValueError("prompt_messages must be non-empty")

    original_messages = [dict(message) for message in prompt_messages]
    last_message = original_messages[-1]
    prompt_text = renderers.format_content_as_string(last_message.get("content", ""))

    has_solution = bool(solution_text)
    has_feedback = bool(feedback_text)
    has_hints = bool(hints_text)
    use_feedback = (
        include_environment_feedback
        and has_feedback
        and (not environment_feedback_only_without_solution or not has_solution)
    )
    use_hints = (
        include_hints
        and has_hints
        and (not hints_only_without_solution or not has_solution)
    )

    solution_section = ""
    if has_solution:
        solution_section = solution_template.format(successful_previous_attempt=solution_text)

    feedback_section = ""
    if use_feedback:
        feedback_section = feedback_template.format(feedback_raw=feedback_text)

    hints_section = ""
    if use_hints:
        hints_section = hints_template.format(stockfish_hints=hints_text)

    if has_solution or use_feedback or use_hints:
        reprompt_text = reprompt_template.format(
            prompt=prompt_text,
            solution=solution_section,
            feedback=feedback_section,
            hints=hints_section,
        )
        system_messages = original_messages[:-1]
        teacher_messages = system_messages + [{"role": "user", "content": reprompt_text}]
        return teacher_messages, use_feedback, True

    return original_messages, False, False


def trust_region_mix_logprob(
    student_logprob: torch.Tensor,
    reference_logprob: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Mix student/reference logprobs with log-sum-exp for numerical stability."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if alpha == 0.0:
        return reference_logprob
    if alpha == 1.0:
        return student_logprob

    alpha_t = torch.tensor(alpha, dtype=student_logprob.dtype, device=student_logprob.device)
    one_minus_alpha_t = torch.tensor(
        1.0 - alpha,
        dtype=student_logprob.dtype,
        device=student_logprob.device,
    )

    return torch.logsumexp(
        torch.stack(
            [
                student_logprob + torch.log(alpha_t),
                reference_logprob + torch.log(one_minus_alpha_t),
            ]
        ),
        dim=0,
    )


def build_sdpo_datum(
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    rollout_logprobs: Sequence[float],
    advantages: Sequence[float],
) -> tinker.Datum:
    """
    Build a training datum for token-level SDPO with completion-only masking.

    If prompt length is P and completion length is C:
    - model_input length = P + C - 1
    - completion supervision starts at index P - 1
    """
    if len(completion_tokens) != len(rollout_logprobs):
        raise ValueError(
            "completion_tokens and rollout_logprobs must have same length "
            f"({len(completion_tokens)} != {len(rollout_logprobs)})"
        )
    if len(completion_tokens) != len(advantages):
        raise ValueError(
            "completion_tokens and advantages must have same length "
            f"({len(completion_tokens)} != {len(advantages)})"
        )
    if len(prompt_tokens) == 0:
        raise ValueError("prompt_tokens must be non-empty")

    all_tokens = list(prompt_tokens) + list(completion_tokens)
    if len(all_tokens) < 2:
        raise ValueError("Need at least 2 tokens to construct right-shifted training data")

    model_input = tinker.ModelInput.from_ints(all_tokens[:-1])
    target_tokens = torch.tensor(all_tokens[1:], dtype=torch.int64)

    token_count = len(target_tokens)
    target_logprobs = torch.zeros(token_count, dtype=torch.float32)
    target_advantages = torch.zeros(token_count, dtype=torch.float32)
    mask = torch.zeros(token_count, dtype=torch.float32)

    completion_start = len(prompt_tokens) - 1
    completion_end = completion_start + len(completion_tokens)

    if len(completion_tokens) > 0:
        target_logprobs[completion_start:completion_end] = torch.tensor(
            rollout_logprobs,
            dtype=torch.float32,
        )
        target_advantages[completion_start:completion_end] = torch.tensor(
            advantages,
            dtype=torch.float32,
        )
        mask[completion_start:completion_end] = 1.0

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(target_tokens),
            "logprobs": TensorData.from_torch(target_logprobs),
            "advantages": TensorData.from_torch(target_advantages),
            "mask": TensorData.from_torch(mask),
        },
    )
