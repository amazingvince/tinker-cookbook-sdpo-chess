from __future__ import annotations

import asyncio
import random
from typing import Any

from tinker_cookbook.recipes.verifiers_rl import verifiers_env as ve


def _make_row(i: int) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": f"prompt-{i}"}],
        "example_id": i,
        "task": "chess_next_move",
        "answer": "e2e4",
        "info": {"fen": f"fen-{i}"},
    }


def test_verifiers_rl_dataset_sequential_batches():
    rows = [_make_row(i) for i in range(5)]
    dataset = ve.VerifiersRLDataset(rows=rows, vf_env=object(), groups_per_batch=2)

    assert len(dataset) == 3
    batch0 = dataset.get_batch(0)
    batch1 = dataset.get_batch(1)
    batch2 = dataset.get_batch(2)

    assert [builder.example_id for builder in batch0] == [0, 1]
    assert [builder.example_id for builder in batch1] == [2, 3]
    assert [builder.example_id for builder in batch2] == [4]


def test_verifiers_rl_dataset_buffer_sampling_with_replacement():
    rows = [_make_row(i) for i in range(4)]
    dataset = ve.VerifiersRLDataset(
        rows=rows,
        vf_env=object(),
        groups_per_batch=3,
        sample_with_replacement=True,
        num_batches=7,
        rng_seed=0,
    )

    assert len(dataset) == 7
    batch = dataset.get_batch(0)
    assert len(batch) == 3
    assert set(builder.example_id for builder in batch).issubset({0, 1, 2, 3})


def test_verifiers_rl_dataset_refreshes_buffer_from_source():
    source_rows = [_make_row(i) for i in range(12)]
    buffer_rows = source_rows[:3]
    dataset = ve.VerifiersRLDataset(
        rows=buffer_rows,
        source_rows=source_rows,
        vf_env=object(),
        groups_per_batch=2,
        sample_with_replacement=True,
        num_batches=10,
        refresh_rows_per_batch=3,
        rng_seed=0,
    )

    initial_ids = {row["example_id"] for row in dataset._buffer_rows}
    assert initial_ids == {0, 1, 2}

    for i in range(len(dataset)):
        _ = dataset.get_batch(i)

    refreshed_ids = {row["example_id"] for row in dataset._buffer_rows}
    assert any(example_id >= 3 for example_id in refreshed_ids)


class _FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows
        self.column_names = ["prompt", "example_id", "task", "answer", "info"]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: str) -> list[Any]:
        return [row[key] for row in self._rows]

    def shuffle(self, seed: int | None = None) -> "_FakeDataset":
        rng = random.Random(seed)
        rows = list(self._rows)
        rng.shuffle(rows)
        return _FakeDataset(rows)

    def select(self, indices: Any) -> "_FakeDataset":
        if hasattr(indices, "__iter__"):
            idxs = list(indices)
        else:
            idxs = [int(indices)]
        return _FakeDataset([self._rows[i] for i in idxs])


class _FakeEnv:
    def __init__(self, rows: list[dict[str, Any]]):
        self._dataset = _FakeDataset(rows)

    def get_dataset(self, n: int = -1, seed: int | None = None) -> _FakeDataset:
        ds = self._dataset.shuffle(seed=seed) if seed is not None else self._dataset
        if n > 0:
            return ds.select(range(min(n, len(ds))))
        return ds


def test_dataset_builder_rolling_buffer_mode(monkeypatch):
    source_rows = [_make_row(i) for i in range(10)]
    fake_env = _FakeEnv(source_rows)

    monkeypatch.setattr(ve, "get_vf_env", lambda: fake_env)

    builder = ve.VerifiersRLDatasetBuilder(
        vf_env_id="hf-chess-mix",
        groups_per_batch=4,
        dataset_n=-1,
        dataset_seed=5,
        dataset_buffer_size=3,
        dataset_num_batches=8,
        dataset_sample_with_replacement=True,
        dataset_refresh_rows_per_batch=1,
    )

    dataset, _ = asyncio.run(builder())
    assert isinstance(dataset, ve.VerifiersRLDataset)
    assert len(dataset) == 8
    assert len(dataset._buffer_rows) == 3

    batch = dataset.get_batch(0)
    assert len(batch) == 4
    assert set(builder_row.example_id for builder_row in batch).issubset(set(range(10)))
