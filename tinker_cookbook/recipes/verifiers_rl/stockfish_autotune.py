from __future__ import annotations

import json
from typing import Literal

import chz

from tinker_cookbook.recipes.verifiers_rl.stockfish_helpers import (
    SystemInfo,
    detect_system_info,
    rank_stockfish_asset_suffixes,
)


@chz.chz
class CLIConfig:
    reserve_cpu_fraction: float = 0.15
    reserve_cpu_min: int = 8
    max_workers: int = 64
    preferred_threads_per_worker: int = 3
    hash_budget_fraction: float = 0.35
    min_hash_mb: int = 256
    max_hash_mb: int = 4096
    output_format: Literal["kv", "json"] = "kv"


def _pick_threads_per_worker(cpu_count: int, preferred: int) -> int:
    if cpu_count >= 96:
        return max(1, preferred)
    if cpu_count >= 48:
        return min(max(1, preferred), 2)
    return 1


def _autotune(system: SystemInfo, config: CLIConfig) -> dict[str, float | int | str]:
    reserve = max(config.reserve_cpu_min, int(system.cpu_count * config.reserve_cpu_fraction))
    usable_cpus = max(1, system.cpu_count - reserve)
    threads_per_worker = _pick_threads_per_worker(system.cpu_count, config.preferred_threads_per_worker)
    workers = max(1, usable_cpus // threads_per_worker)
    workers = min(workers, config.max_workers)

    hash_budget_mb = int(system.memory_mb * config.hash_budget_fraction)
    if workers > 0:
        hash_mb = hash_budget_mb // workers
    else:
        hash_mb = config.min_hash_mb
    hash_mb = max(config.min_hash_mb, min(config.max_hash_mb, hash_mb))

    worker_thread_product = workers * threads_per_worker
    if worker_thread_product >= 96:
        verification_sample_rate = 1.0
        analysis_time_limit_sec = 0.12
    elif worker_thread_product >= 48:
        verification_sample_rate = 0.5
        analysis_time_limit_sec = 0.15
    else:
        verification_sample_rate = 0.25
        analysis_time_limit_sec = 0.20

    return {
        "system_os": system.os_name,
        "system_machine": system.machine,
        "system_cpu_count": system.cpu_count,
        "system_memory_mb": system.memory_mb,
        "stockfish_num_workers": workers,
        "stockfish_threads": threads_per_worker,
        "stockfish_hash_mb": hash_mb,
        "stockfish_analysis_time_limit_sec": analysis_time_limit_sec,
        "stockfish_engine_max_retries": 1,
        "stockfish_verification_sample_rate": verification_sample_rate,
        "stockfish_asset_priority": ",".join(rank_stockfish_asset_suffixes(system)),
    }


def main(config: CLIConfig) -> None:
    if not 0.0 <= config.reserve_cpu_fraction < 1.0:
        raise ValueError(
            f"reserve_cpu_fraction must be in [0, 1), got {config.reserve_cpu_fraction}"
        )
    if config.max_workers <= 0:
        raise ValueError(f"max_workers must be >= 1, got {config.max_workers}")
    if not 0.0 < config.hash_budget_fraction <= 0.9:
        raise ValueError(
            f"hash_budget_fraction must be in (0, 0.9], got {config.hash_budget_fraction}"
        )
    if config.min_hash_mb <= 0 or config.max_hash_mb < config.min_hash_mb:
        raise ValueError(
            f"Require 0 < min_hash_mb <= max_hash_mb, got {config.min_hash_mb}, {config.max_hash_mb}"
        )

    system = detect_system_info()
    tuned = _autotune(system, config)

    if config.output_format == "json":
        print(json.dumps(tuned, ensure_ascii=True, indent=2))
        return

    for key, value in tuned.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    main(cfg)
