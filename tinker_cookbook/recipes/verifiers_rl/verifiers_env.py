from __future__ import annotations

from contextvars import ContextVar
import random
from typing import Sequence

import chz
import tinker
import verifiers as vf

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

_vf_env_ctx: ContextVar[vf.Environment | None] = ContextVar("vf_env", default=None)


def set_vf_env(env: vf.Environment) -> None:
    """Set the verifiers environment for the current context."""
    _vf_env_ctx.set(env)


def get_vf_env() -> vf.Environment | None:
    """Get the verifiers environment from the current context."""
    return _vf_env_ctx.get()


def convert_states_to_trajectory_group(states: list[vf.State]) -> TrajectoryGroup:
    """Convert verifiers States to tinker TrajectoryGroup."""
    trajectories_G: list[Trajectory] = []
    final_rewards_G: list[float] = []
    metrics_G: list[dict[str, float | int]] = []

    for state in states:
        transitions: list[Transition] = []
        trajectory_steps = state.get("trajectory", [])

        for i, step in enumerate(trajectory_steps):
            tokens_data = step.get("tokens")
            if tokens_data is not None:
                prompt_ids = tokens_data.get("prompt_ids", [])
                ob = tinker.ModelInput.from_ints(prompt_ids)
                completion_ids = tokens_data.get("completion_ids", [])
                completion_logprobs = tokens_data.get("completion_logprobs", [])
                ac = TokensWithLogprobs(
                    tokens=completion_ids,
                    maybe_logprobs=completion_logprobs,
                )
            else:
                ob = tinker.ModelInput.empty()
                ac = TokensWithLogprobs(tokens=[], maybe_logprobs=[])

            is_last = i == len(trajectory_steps) - 1
            transition = Transition(
                ob=ob,
                ac=ac,
                reward=0.0,
                episode_done=is_last,
                metrics={},
            )
            transitions.append(transition)

        trajectory = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())
        trajectories_G.append(trajectory)
        final_rewards_G.append(state.get("reward") or 0.0)
        metrics_G.append(state.get("metrics") or {})

    return TrajectoryGroup(
        trajectories_G=trajectories_G,
        final_rewards_G=final_rewards_G,
        metrics_G=metrics_G,
    )


class VerifiersRLDataset(RLDataset):
    def __init__(
        self,
        rows: list[dict],
        vf_env: vf.Environment,
        groups_per_batch: int,
        *,
        source_rows: list[dict] | None = None,
        sample_with_replacement: bool = False,
        num_batches: int = -1,
        refresh_rows_per_batch: int = 0,
        rng_seed: int | None = None,
    ):
        if groups_per_batch <= 0:
            raise ValueError(f"groups_per_batch must be >= 1, got {groups_per_batch}")
        if num_batches == 0 or num_batches < -1:
            raise ValueError(f"num_batches must be -1 or >= 1, got {num_batches}")
        if refresh_rows_per_batch < 0:
            raise ValueError(f"refresh_rows_per_batch must be >= 0, got {refresh_rows_per_batch}")

        self._buffer_rows = list(rows)
        self._source_rows = list(source_rows) if source_rows is not None else list(rows)
        self.vf_env = vf_env
        self.groups_per_batch = groups_per_batch
        self.sample_with_replacement = bool(sample_with_replacement)
        self.num_batches = int(num_batches)
        self.refresh_rows_per_batch = int(refresh_rows_per_batch)
        self._rng = random.Random(rng_seed)

    def __len__(self) -> int:
        if self.num_batches > 0:
            return self.num_batches
        return (len(self._buffer_rows) + self.groups_per_batch - 1) // self.groups_per_batch

    def _refresh_buffer(self) -> None:
        if self.refresh_rows_per_batch <= 0:
            return
        if not self._buffer_rows:
            return
        if len(self._source_rows) <= len(self._buffer_rows):
            return

        num_to_refresh = min(self.refresh_rows_per_batch, len(self._buffer_rows))
        for _ in range(num_to_refresh):
            dst_idx = self._rng.randrange(len(self._buffer_rows))
            src_row = self._source_rows[self._rng.randrange(len(self._source_rows))]
            self._buffer_rows[dst_idx] = src_row

    @staticmethod
    def _make_builder_from_row(vf_env: vf.Environment, row: dict) -> EnvGroupBuilder:
        info = row.get("info", {})
        info_payload = dict(info) if isinstance(info, dict) else info
        return VerifiersEnvGroupBuilder(
            vf_env=vf_env,
            prompt=row["prompt"],
            example_id=row["example_id"],
            task=row["task"],
            answer=row.get("answer", ""),
            info=info_payload,
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if not self._buffer_rows:
            return []

        if self.sample_with_replacement:
            selected_rows = [
                self._buffer_rows[self._rng.randrange(len(self._buffer_rows))]
                for _ in range(self.groups_per_batch)
            ]
            self._refresh_buffer()
            return [self._make_builder_from_row(self.vf_env, row) for row in selected_rows]

        start = index * self.groups_per_batch
        end = min(len(self._buffer_rows), start + self.groups_per_batch)
        selected_rows = self._buffer_rows[start:end]
        self._refresh_buffer()
        return [self._make_builder_from_row(self.vf_env, row) for row in selected_rows]


@chz.chz
class VerifiersRLDatasetBuilder(RLDatasetBuilder):
    vf_env_id: str
    vf_env_args: dict = chz.field(default_factory=dict)
    groups_per_batch: int = 32
    dataset_n: int = -1
    dataset_seed: int | None = None
    dataset_buffer_size: int = -1
    dataset_num_batches: int = -1
    dataset_sample_with_replacement: bool = False
    dataset_refresh_rows_per_batch: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        if self.dataset_buffer_size == 0 or self.dataset_buffer_size < -1:
            raise ValueError(
                f"dataset_buffer_size must be -1 or >= 1, got {self.dataset_buffer_size}"
            )
        if self.dataset_num_batches == 0 or self.dataset_num_batches < -1:
            raise ValueError(
                f"dataset_num_batches must be -1 or >= 1, got {self.dataset_num_batches}"
            )
        if self.dataset_refresh_rows_per_batch < 0:
            raise ValueError(
                "dataset_refresh_rows_per_batch must be >= 0, got "
                f"{self.dataset_refresh_rows_per_batch}"
            )

        vf_env = get_vf_env()
        if vf_env is None:
            vf_env = vf.load_environment(self.vf_env_id, **self.vf_env_args)
            set_vf_env(vf_env)
        ds = vf_env.get_dataset(n=self.dataset_n, seed=self.dataset_seed)
        source_rows = [
            {
                "prompt": ds["prompt"][i],
                "example_id": ds["example_id"][i],
                "task": ds["task"][i],
                **({"answer": ds["answer"][i]} if "answer" in ds.column_names else {}),
                **({"info": ds["info"][i]} if "info" in ds.column_names else {}),
            }
            for i in range(len(ds))
        ]
        if self.dataset_buffer_size > 0:
            rows = source_rows[: min(self.dataset_buffer_size, len(source_rows))]
        else:
            rows = source_rows

        return (
            VerifiersRLDataset(
                rows=rows,
                source_rows=source_rows,
                vf_env=vf_env,
                groups_per_batch=self.groups_per_batch,
                sample_with_replacement=self.dataset_sample_with_replacement,
                num_batches=self.dataset_num_batches,
                refresh_rows_per_batch=self.dataset_refresh_rows_per_batch,
                rng_seed=self.dataset_seed,
            ),
            None,
        )


class VerifiersEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        vf_env: vf.Environment,
        prompt: vf.Messages,
        example_id: int,
        task: str,
        answer: str = "",
        info: dict | None = None,
    ):
        self.vf_env = vf_env
        self.prompt = prompt
        self.example_id = example_id
        self.task = task
        self.answer = answer
        self.info = info or {}

    def get_rollout_inputs(self, group_size: int) -> list[vf.RolloutInput]:
        return [
            vf.RolloutInput(
                prompt=self.prompt,
                answer=self.answer,
                task=self.task,
                info=self.info,
                example_id=self.example_id,
            )
            for _ in range(group_size)
        ]

    async def make_envs(self):
        return []  # unused when using custom_do_group_rollout

    def logging_tags(self) -> list[str]:
        return [self.task] if self.task else []
