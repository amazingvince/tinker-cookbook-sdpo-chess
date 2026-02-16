from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tinker_cookbook.sdpo import train as sdpo_train
from tinker_cookbook.sdpo.utils import (
    build_sdpo_datum,
    build_teacher_messages,
    extract_feedback_text,
    select_solution_idx,
    trust_region_mix_logprob,
)


class _DummyDatasetBuilder:
    async def __call__(self):
        raise RuntimeError("Dataset builder should not be called in unit tests")


class _FakeTokenizer:
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeRenderer:
    def build_generation_prompt(self, messages: list[dict[str, Any]]):
        text = " ".join(str(m.get("content", "")) for m in messages)
        n_tokens = max(1, len(text.split()))
        return sdpo_train.tinker.ModelInput.from_ints([1000 + i for i in range(n_tokens)])


class _FakeSamplingClient:
    def __init__(self, base_logprob: float):
        self.base_logprob = base_logprob

    async def compute_logprobs_async(
        self, model_input: sdpo_train.tinker.ModelInput
    ) -> list[float | None]:
        return [self.base_logprob - (0.01 * i) for i in range(model_input.length)]

    async def sample_async(
        self,
        prompt: sdpo_train.tinker.ModelInput,
        num_samples: int,
        sampling_params: sdpo_train.tinker.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ):
        _ = (num_samples, sampling_params)
        if not include_prompt_logprobs:
            raise ValueError("tests expect include_prompt_logprobs=True")
        tokens = prompt.to_ints()
        prompt_logprobs: list[float | None] = [None]
        topk: list[list[tuple[int, float]] | None] = [None]
        for i, token_id in enumerate(tokens[1:], start=1):
            lp = self.base_logprob - (0.01 * i)
            prompt_logprobs.append(lp)
            if topk_prompt_logprobs > 0:
                topk.append(
                    [
                        (int(token_id), lp),
                        (int(token_id + 1000), lp - 0.4),
                    ][:topk_prompt_logprobs]
                )
            else:
                topk.append([])
        return SimpleNamespace(
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=topk if topk_prompt_logprobs > 0 else None,
            sequences=[],
        )


def _make_state(
    *,
    prompt_messages: list[dict[str, Any]],
    prompt_ids: list[int],
    completion_ids: list[int],
    completion_logprobs: list[float],
    reward: float,
    feedback_text: str | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "prompt": prompt_messages,
        "reward": reward,
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": prompt_ids,
                    "completion_ids": completion_ids,
                    "completion_logprobs": completion_logprobs,
                }
            }
        ],
    }
    if feedback_text is not None:
        state["info"] = {"feedback": feedback_text}
    return state


def _make_config(**overrides: Any) -> sdpo_train.Config:
    cfg_kwargs: dict[str, Any] = {
        "learning_rate": 1e-4,
        "dataset_builder": _DummyDatasetBuilder(),
        "model_name": "dummy/model",
        "max_tokens": 128,
        "log_path": "/tmp/sdpo-tests",
    }
    cfg_kwargs.update(overrides)
    return sdpo_train.Config(**cfg_kwargs)


def test_extract_feedback_text_precedence_and_shapes():
    keys = ["feedback", "error", "errors", "judge_feedback"]

    state_top_level = {
        "feedback": "  top-level  ",
        "info": {"feedback": "nested"},
    }
    assert extract_feedback_text(state_top_level, keys) == "top-level"

    state_list_and_nested = {
        "feedback": "  ",
        "metrics": {"feedback": ["", "first issue", {"message": "second issue"}]},
    }
    assert extract_feedback_text(state_list_and_nested, keys) == "first issue\nsecond issue"

    state_dict_value = {
        "reward_extra_info": {
            "error": {
                "message": "division by zero",
                "code": 123,
            }
        }
    }
    assert extract_feedback_text(state_dict_value, keys) == "division by zero"

    state_trajectory = {
        "trajectory": [
            {"logs": {"judge_feedback": ["ignored"]}},
            {"logs": {"judge_feedback": ["late error", {"message": "late detail"}]}},
        ]
    }
    assert extract_feedback_text(state_trajectory, keys) == "late error\nlate detail"

    state_empty = {"info": {"feedback": []}}
    assert extract_feedback_text(state_empty, keys) is None


def test_select_solution_idx_group_logic():
    assert select_solution_idx(
        rewards=[0.1, 0.2, 0.3],
        sample_idx=0,
        success_reward_threshold=0.5,
        dont_reprompt_on_self_success=True,
    ) is None

    assert select_solution_idx(
        rewards=[0.0, 0.8, 0.4],
        sample_idx=0,
        success_reward_threshold=0.5,
        dont_reprompt_on_self_success=True,
    ) == 1

    assert select_solution_idx(
        rewards=[0.9, 0.2, 0.1],
        sample_idx=0,
        success_reward_threshold=0.5,
        dont_reprompt_on_self_success=True,
    ) is None

    assert select_solution_idx(
        rewards=[0.9, 0.2, 0.1],
        sample_idx=0,
        success_reward_threshold=0.5,
        dont_reprompt_on_self_success=False,
    ) == 0


def test_build_teacher_messages_variants():
    prompt_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Solve 2 + 2."},
    ]

    solution_only, feedback_used, used_reprompt = build_teacher_messages(
        prompt_messages=prompt_messages,
        solution_text="4",
        feedback_text=None,
        reprompt_template="{prompt}{solution}{feedback}",
        solution_template=" SOLUTION={successful_previous_attempt}",
        feedback_template=" FEEDBACK={feedback_raw}",
        include_environment_feedback=True,
        environment_feedback_only_without_solution=True,
    )
    assert used_reprompt is True
    assert feedback_used is False
    assert "SOLUTION=4" in str(solution_only[-1]["content"])
    assert "FEEDBACK=" not in str(solution_only[-1]["content"])

    feedback_only, feedback_used, used_reprompt = build_teacher_messages(
        prompt_messages=prompt_messages,
        solution_text=None,
        feedback_text="wrong sign",
        reprompt_template="{prompt}{solution}{feedback}",
        solution_template=" SOLUTION={successful_previous_attempt}",
        feedback_template=" FEEDBACK={feedback_raw}",
        include_environment_feedback=True,
        environment_feedback_only_without_solution=True,
    )
    assert used_reprompt is True
    assert feedback_used is True
    assert "FEEDBACK=wrong sign" in str(feedback_only[-1]["content"])

    both_sections, feedback_used, used_reprompt = build_teacher_messages(
        prompt_messages=prompt_messages,
        solution_text="4",
        feedback_text="wrong sign",
        reprompt_template="{prompt}{solution}{feedback}",
        solution_template=" SOLUTION={successful_previous_attempt}",
        feedback_template=" FEEDBACK={feedback_raw}",
        include_environment_feedback=True,
        environment_feedback_only_without_solution=False,
    )
    assert used_reprompt is True
    assert feedback_used is True
    assert "SOLUTION=4" in str(both_sections[-1]["content"])
    assert "FEEDBACK=wrong sign" in str(both_sections[-1]["content"])

    no_sections, feedback_used, used_reprompt = build_teacher_messages(
        prompt_messages=prompt_messages,
        solution_text=None,
        feedback_text=None,
        reprompt_template="{prompt}{solution}{feedback}",
        solution_template=" SOLUTION={successful_previous_attempt}",
        feedback_template=" FEEDBACK={feedback_raw}",
        include_environment_feedback=True,
        environment_feedback_only_without_solution=True,
    )
    assert used_reprompt is False
    assert feedback_used is False
    assert no_sections == prompt_messages


def test_trust_region_mix_logprob_matches_manual_logsumexp():
    student = torch.tensor([-0.2, -0.7], dtype=torch.float32)
    reference = torch.tensor([-1.1, -0.3], dtype=torch.float32)
    alpha = 0.05

    mixed = trust_region_mix_logprob(student, reference, alpha)
    expected = torch.log((alpha * torch.exp(student)) + ((1.0 - alpha) * torch.exp(reference)))
    assert torch.allclose(mixed, expected, atol=1e-6)


def test_build_sdpo_datum_alignment():
    prompt_tokens = [1, 2, 3]
    completion_tokens = [10, 11]
    rollout_logprobs = [-1.5, -1.0]
    advantages = [0.3, -0.2]

    datum = build_sdpo_datum(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        rollout_logprobs=rollout_logprobs,
        advantages=advantages,
    )
    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
    mask = datum.loss_fn_inputs["mask"].to_torch()
    logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
    adv = datum.loss_fn_inputs["advantages"].to_torch()

    assert len(target_tokens) == len(prompt_tokens) + len(completion_tokens) - 1
    assert target_tokens.tolist() == [2, 3, 10, 11]
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert logprobs.tolist() == [0.0, 0.0, -1.5, -1.0]
    assert torch.allclose(adv, torch.tensor([0.0, 0.0, 0.3, -0.2], dtype=torch.float32))


def test_zero_advantage_when_no_reprompt_source():
    async def _inner():
        config = _make_config(teacher_regularization="none")
        state = _make_state(
            prompt_messages=[{"role": "user", "content": "Solve 2+2"}],
            prompt_ids=[1, 2, 3],
            completion_ids=[4, 5],
            completion_logprobs=[-1.0, -0.9],
            reward=0.1,
            feedback_text=None,
        )
        data, stats = await sdpo_train.build_group_sdpo_datums(
            states=[state],
            config=config,
            current_sampling_client=_FakeSamplingClient(base_logprob=-0.2),
            reference_sampling_client=None,
            ema_teacher_sampling_clients=None,
            renderer=_FakeRenderer(),
            tokenizer=_FakeTokenizer(),
            feedback_keys=["feedback"],
            teacher_logprob_semaphore=None,
        )
        assert len(data) == 1
        assert stats.zero_adv_samples == 1
        assert stats.reprompt_samples == 0
        advantages = data[0].loss_fn_inputs["advantages"].to_torch()
        mask = data[0].loss_fn_inputs["mask"].to_torch()
        assert torch.allclose(advantages[mask > 0], torch.zeros_like(advantages[mask > 0]))

    asyncio.run(_inner())


def test_truncate_teacher_prompt_tokens_modes():
    tokens = [1, 2, 3, 4, 5]
    assert sdpo_train._truncate_teacher_prompt_tokens(tokens, 3, "right") == [1, 2, 3]
    assert sdpo_train._truncate_teacher_prompt_tokens(tokens, 3, "left") == [3, 4, 5]
    with pytest.raises(ValueError, match="reprompt_truncation='error'"):
        sdpo_train._truncate_teacher_prompt_tokens(tokens, 3, "error")


def test_grpo_mix_lambda_without_reprompt(monkeypatch):
    captured_advantages: list[list[float]] = []

    async def fake_train_step(
        data_D: list[sdpo_train.tinker.Datum],
        training_client: Any,
        learning_rate: float,
        num_substeps: int,
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ):
        _ = (training_client, learning_rate, num_substeps, loss_fn_config, metrics, loss_fn)
        for datum in data_D:
            adv = datum.loss_fn_inputs["advantages"].to_torch()
            mask = datum.loss_fn_inputs["mask"].to_torch()
            captured_advantages.append(adv[mask > 0].tolist())
        return []

    monkeypatch.setattr(sdpo_train.rl_train, "train_step", fake_train_step)

    async def _inner():
        config = _make_config(
            teacher_regularization="none",
            success_reward_threshold=2.0,
            grpo_mix_lambda=1.0,
        )
        state_high_reward = _make_state(
            prompt_messages=[{"role": "user", "content": "Task"}],
            prompt_ids=[1, 2],
            completion_ids=[3, 4],
            completion_logprobs=[-1.0, -1.0],
            reward=1.0,
            feedback_text=None,
        )
        state_low_reward = _make_state(
            prompt_messages=[{"role": "user", "content": "Task"}],
            prompt_ids=[1, 2],
            completion_ids=[5, 6],
            completion_logprobs=[-1.0, -1.0],
            reward=0.0,
            feedback_text=None,
        )

        metrics = await sdpo_train.run_sdpo_batch_update(
            config=config,
            training_client=object(),
            current_sampling_client=_FakeSamplingClient(base_logprob=-0.3),
            reference_sampling_client=None,
            ema_teacher_sampling_clients=None,
            renderer=_FakeRenderer(),
            tokenizer=_FakeTokenizer(),
            states_by_group=[[state_high_reward, state_low_reward]],
        )
        assert metrics["sdpo/num_zero_adv_samples"] == 0
        assert metrics["sdpo/mean_abs_advantage"] > 0.0
        assert metrics["sdpo/grpo_mix_lambda"] == 1.0

    asyncio.run(_inner())
    assert len(captured_advantages) == 2
    flattened = [round(v, 6) for seq in captured_advantages for v in seq]
    assert set(flattened) == {0.5, -0.5}


def test_run_sdpo_batch_update_mocked_path(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_train_step(
        data_D: list[sdpo_train.tinker.Datum],
        training_client: Any,
        learning_rate: float,
        num_substeps: int,
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ):
        _ = (training_client, learning_rate, num_substeps, loss_fn_config)
        if metrics is not None:
            metrics["optim/mock_called"] = 1.0
        captured["loss_fn"] = loss_fn
        captured["num_datums"] = len(data_D)
        captured["all_have_mask"] = all("mask" in datum.loss_fn_inputs for datum in data_D)
        return []

    monkeypatch.setattr(sdpo_train.rl_train, "train_step", fake_train_step)

    async def _inner():
        config = _make_config(teacher_regularization="none")
        success_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Solve 2+2"}],
            prompt_ids=[10, 11],
            completion_ids=[12, 13],
            completion_logprobs=[-1.2, -1.1],
            reward=1.0,
            feedback_text=None,
        )
        failed_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Solve 2+2"}],
            prompt_ids=[10, 11],
            completion_ids=[14, 15],
            completion_logprobs=[-1.4, -1.3],
            reward=0.0,
            feedback_text="Try again with arithmetic.",
        )

        metrics = await sdpo_train.run_sdpo_batch_update(
            config=config,
            training_client=object(),
            current_sampling_client=_FakeSamplingClient(base_logprob=-0.2),
            reference_sampling_client=None,
            ema_teacher_sampling_clients=None,
            renderer=_FakeRenderer(),
            tokenizer=_FakeTokenizer(),
            states_by_group=[[success_state, failed_state]],
        )

        assert captured["loss_fn"] == "importance_sampling"
        assert captured["num_datums"] == 2
        assert captured["all_have_mask"] is True

        assert metrics["sdpo/num_datums"] == 2
        assert metrics["sdpo/success_group_fraction"] == 1.0
        assert metrics["sdpo/success_sample_fraction"] == 0.5
        assert metrics["sdpo/feedback_available_fraction"] == 0.5
        assert metrics["sdpo/feedback_used_fraction"] == 0.0
        assert metrics["sdpo/reprompt_sample_fraction"] == 0.5
        assert metrics["sdpo/num_zero_adv_samples"] == 1
        assert metrics["sdpo/num_skipped_samples"] == 0
        assert metrics["sdpo/mean_abs_advantage"] > 0.0
        assert metrics["optim/mock_called"] == 1.0

    asyncio.run(_inner())


def test_ema_distribution_weights_properties():
    weights = sdpo_train._ema_distribution_weights(alpha=0.2, num_components=4)
    assert len(weights) == 4
    assert math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert weights[0] > weights[1] > weights[2]


def test_topk_tail_advantage_matches_manual_computation():
    student_topk = [(10, math.log(0.6)), (11, math.log(0.3))]
    teacher_topk = [(10, math.log(0.5)), (11, math.log(0.2))]
    adv, overlap_count, total_count = sdpo_train._compute_topk_tail_advantage(
        student_topk=student_topk,
        teacher_topk=teacher_topk,
        add_tail=True,
    )
    assert overlap_count == 2
    assert total_count == 2
    expected = (
        0.6 * (math.log(0.5) - math.log(0.6))
        + 0.3 * (math.log(0.2) - math.log(0.3))
        + 0.1 * (math.log(0.3) - math.log(0.1))
    )
    assert adv is not None
    assert math.isclose(adv, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_full_logit_distillation_topk_path(monkeypatch):
    async def fake_train_step(
        data_D: list[sdpo_train.tinker.Datum],
        training_client: Any,
        learning_rate: float,
        num_substeps: int,
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ):
        _ = (data_D, training_client, learning_rate, num_substeps, loss_fn, loss_fn_config, metrics)
        return []

    monkeypatch.setattr(sdpo_train.rl_train, "train_step", fake_train_step)

    async def _inner():
        config = _make_config(
            teacher_regularization="none",
            full_logit_distillation=True,
            distillation_topk=2,
        )
        success_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Solve 2+2"}],
            prompt_ids=[10, 11],
            completion_ids=[12, 13],
            completion_logprobs=[-1.2, -1.1],
            reward=1.0,
            feedback_text=None,
        )
        failed_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Solve 2+2"}],
            prompt_ids=[10, 11],
            completion_ids=[14, 15],
            completion_logprobs=[-1.4, -1.3],
            reward=0.0,
            feedback_text="Try again with arithmetic.",
        )

        metrics = await sdpo_train.run_sdpo_batch_update(
            config=config,
            training_client=object(),
            current_sampling_client=_FakeSamplingClient(base_logprob=-0.2),
            reference_sampling_client=None,
            ema_teacher_sampling_clients=None,
            renderer=_FakeRenderer(),
            tokenizer=_FakeTokenizer(),
            states_by_group=[[success_state, failed_state]],
        )
        assert metrics["sdpo/full_logit_distillation"] == 1.0
        assert metrics["sdpo/topk_overlap_fraction"] > 0.0

    asyncio.run(_inner())


def test_updates_per_batch_and_loss_fn(monkeypatch):
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_train_step(
        data_D: list[sdpo_train.tinker.Datum],
        training_client: Any,
        learning_rate: float,
        num_substeps: int,
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ):
        _ = (data_D, training_client, learning_rate, num_substeps, metrics)
        calls.append((loss_fn, loss_fn_config))
        return []

    monkeypatch.setattr(sdpo_train.rl_train, "train_step", fake_train_step)

    async def _inner():
        config = _make_config(
            teacher_regularization="none",
            updates_per_batch=3,
            loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
        )
        success_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Task"}],
            prompt_ids=[1, 2],
            completion_ids=[3, 4],
            completion_logprobs=[-1.0, -1.0],
            reward=1.0,
            feedback_text=None,
        )
        failed_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Task"}],
            prompt_ids=[1, 2],
            completion_ids=[5, 6],
            completion_logprobs=[-1.1, -1.1],
            reward=0.0,
            feedback_text="wrong answer",
        )

        metrics = await sdpo_train.run_sdpo_batch_update(
            config=config,
            training_client=object(),
            current_sampling_client=_FakeSamplingClient(base_logprob=-0.2),
            reference_sampling_client=None,
            ema_teacher_sampling_clients=None,
            renderer=_FakeRenderer(),
            tokenizer=_FakeTokenizer(),
            states_by_group=[[success_state, failed_state]],
        )
        assert metrics["sdpo/updates_per_batch"] == 3.0

    asyncio.run(_inner())
    assert len(calls) == 3
    assert all(call[0] == "ppo" for call in calls)
    assert all(call[1] == {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1} for call in calls)


def test_ema_teacher_regularization_path(monkeypatch):
    async def fake_train_step(
        data_D: list[sdpo_train.tinker.Datum],
        training_client: Any,
        learning_rate: float,
        num_substeps: int,
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ):
        _ = (data_D, training_client, learning_rate, num_substeps, loss_fn, loss_fn_config, metrics)
        return []

    monkeypatch.setattr(sdpo_train.rl_train, "train_step", fake_train_step)

    async def _inner():
        config = _make_config(
            teacher_regularization="ema",
            teacher_mix_alpha=0.2,
        )
        success_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Task"}],
            prompt_ids=[1, 2],
            completion_ids=[3, 4],
            completion_logprobs=[-1.0, -1.0],
            reward=1.0,
            feedback_text=None,
        )
        failed_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Task"}],
            prompt_ids=[1, 2],
            completion_ids=[5, 6],
            completion_logprobs=[-1.1, -1.1],
            reward=0.0,
            feedback_text="wrong answer",
        )

        metrics = await sdpo_train.run_sdpo_batch_update(
            config=config,
            training_client=object(),
            current_sampling_client=_FakeSamplingClient(base_logprob=-0.2),
            reference_sampling_client=None,
            ema_teacher_sampling_clients=[
                _FakeSamplingClient(base_logprob=-0.2),
                _FakeSamplingClient(base_logprob=-0.8),
            ],
            renderer=_FakeRenderer(),
            tokenizer=_FakeTokenizer(),
            states_by_group=[[success_state, failed_state]],
        )
        assert metrics["sdpo/mean_abs_advantage"] > 0.0

    asyncio.run(_inner())


def test_full_logit_distillation_with_ema_teacher(monkeypatch):
    async def fake_train_step(
        data_D: list[sdpo_train.tinker.Datum],
        training_client: Any,
        learning_rate: float,
        num_substeps: int,
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ):
        _ = (data_D, training_client, learning_rate, num_substeps, loss_fn, loss_fn_config, metrics)
        return []

    monkeypatch.setattr(sdpo_train.rl_train, "train_step", fake_train_step)

    async def _inner():
        config = _make_config(
            teacher_regularization="ema",
            teacher_mix_alpha=0.2,
            full_logit_distillation=True,
            distillation_topk=2,
        )
        success_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Task"}],
            prompt_ids=[1, 2],
            completion_ids=[3, 4],
            completion_logprobs=[-1.0, -1.0],
            reward=1.0,
            feedback_text=None,
        )
        failed_state = _make_state(
            prompt_messages=[{"role": "user", "content": "Task"}],
            prompt_ids=[1, 2],
            completion_ids=[5, 6],
            completion_logprobs=[-1.1, -1.1],
            reward=0.0,
            feedback_text="wrong answer",
        )

        metrics = await sdpo_train.run_sdpo_batch_update(
            config=config,
            training_client=object(),
            current_sampling_client=_FakeSamplingClient(base_logprob=-0.2),
            reference_sampling_client=None,
            ema_teacher_sampling_clients=[
                _FakeSamplingClient(base_logprob=-0.2),
                _FakeSamplingClient(base_logprob=-0.8),
            ],
            renderer=_FakeRenderer(),
            tokenizer=_FakeTokenizer(),
            states_by_group=[[success_state, failed_state]],
        )
        assert metrics["sdpo/full_logit_distillation"] == 1.0
        assert metrics["sdpo/topk_overlap_fraction"] > 0.0

    asyncio.run(_inner())
