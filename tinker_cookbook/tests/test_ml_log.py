from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tinker_cookbook.utils import ml_log


class _FakeTable:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.rows: list[tuple[Any, ...]] = []

    def add_data(self, *row: Any) -> None:
        self.rows.append(row)


class _FakeRun:
    url = "https://wandb.fake/run/123"


class _FakeWandb:
    def __init__(self):
        self.logged: list[tuple[dict[str, Any], int | None]] = []
        self.tables: list[_FakeTable] = []
        self.config = SimpleNamespace(update=lambda *_args, **_kwargs: None)

    def init(self, **_kwargs: Any) -> _FakeRun:
        return _FakeRun()

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self.logged.append((payload, step))

    def Table(self, columns: list[str]) -> _FakeTable:
        table = _FakeTable(columns)
        self.tables.append(table)
        return table

    def finish(self) -> None:
        return None


def test_wandb_logger_logs_sdpo_debug_examples_in_table(monkeypatch, tmp_path):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(ml_log, "_wandb_available", True)
    monkeypatch.setattr(ml_log, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")

    logger = ml_log.WandbLogger(
        project="unit-test",
        config={"hello": "world"},
        log_dir=tmp_path,
        wandb_name="run",
    )

    key = "sdpo/debug_examples/batch_000007"
    text = "debug rollout snapshot"
    logger.log_long_text(key=key, text=text, step=7)

    assert len(fake_wandb.logged) == 2
    assert fake_wandb.logged[0] == ({key: text}, 7)

    second_payload, second_step = fake_wandb.logged[1]
    assert second_step == 7
    assert "sdpo/debug_examples/table" in second_payload
    assert second_payload["sdpo/debug_examples/latest"] == text

    assert logger._sdpo_debug_examples_table.rows == [(7, key, text)]
