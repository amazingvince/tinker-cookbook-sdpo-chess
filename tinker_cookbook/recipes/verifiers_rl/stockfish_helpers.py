from __future__ import annotations

import json
import os
import platform
import subprocess
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_STOCKFISH_RELEASES_API = "https://api.github.com/repos/official-stockfish/Stockfish/releases"


@dataclass(frozen=True)
class SystemInfo:
    os_name: str
    machine: str
    cpu_count: int
    memory_mb: int
    cpu_flags: frozenset[str]


def detect_cpu_flags() -> set[str]:
    flags: set[str] = set()
    os_name = platform.system().lower()
    if os_name == "linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("flags"):
                        _, value = line.split(":", 1)
                        flags.update(value.strip().lower().split())
        except Exception:
            pass
    elif os_name == "darwin":
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", key],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                continue
            flags.update(out.lower().split())
    return flags


def _detect_memory_mb() -> int:
    os_name = platform.system().lower()
    if os_name == "linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return max(1, kb // 1024)
        except Exception:
            pass
    elif os_name == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return max(1, int(out) // (1024 * 1024))
        except Exception:
            pass

    # Generic fallback.
    try:
        pages = int(subprocess.check_output(["getconf", "PHYS_PAGES"], text=True).strip())
        page_size = int(subprocess.check_output(["getconf", "PAGESIZE"], text=True).strip())
        return max(1, (pages * page_size) // (1024 * 1024))
    except Exception:
        return 16384


def detect_system_info() -> SystemInfo:
    return SystemInfo(
        os_name=platform.system().lower(),
        machine=platform.machine().lower(),
        cpu_count=max(1, int(os.cpu_count() or 1)),
        memory_mb=_detect_memory_mb(),
        cpu_flags=frozenset(detect_cpu_flags()),
    )


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "tinker-cookbook-stockfish-helper"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def fetch_stockfish_release(tag: str = "latest") -> Mapping[str, Any]:
    if tag == "latest":
        return _get_json(f"{_STOCKFISH_RELEASES_API}/latest")
    return _get_json(f"{_STOCKFISH_RELEASES_API}/tags/{tag}")


def _has(flags: set[str] | frozenset[str], *keys: str) -> bool:
    return all(key in flags for key in keys)


def rank_stockfish_asset_suffixes(system: SystemInfo) -> list[str]:
    flags = set(system.cpu_flags)
    if system.os_name == "linux" and system.machine in {"x86_64", "amd64"}:
        order: list[str] = []
        if _has(flags, "avx512vnni") or _has(flags, "avx512_vnni"):
            order.append("x86-64-vnni512")
        if _has(flags, "avxvnni"):
            order.append("x86-64-avxvnni")
        if _has(flags, "avx512f", "avx512bw", "avx512dq", "avx512vl"):
            order.append("x86-64-avx512icl")
        if _has(flags, "avx512f"):
            order.append("x86-64-avx512")
        if _has(flags, "avx2"):
            order.append("x86-64-avx2")
        if _has(flags, "bmi2"):
            order.append("x86-64-bmi2")
        if _has(flags, "sse4_1", "popcnt") or _has(flags, "sse4.1", "popcnt"):
            order.append("x86-64-sse41-popcnt")
        order.append("x86-64")
        return order

    if system.os_name == "darwin" and system.machine in {"x86_64", "amd64"}:
        order = []
        if _has(flags, "avx2"):
            order.append("x86-64-avx2")
        if _has(flags, "bmi2"):
            order.append("x86-64-bmi2")
        if _has(flags, "sse4_1", "popcnt") or _has(flags, "sse4.1", "popcnt"):
            order.append("x86-64-sse41-popcnt")
        order.append("x86-64")
        return order

    if system.os_name == "darwin" and system.machine in {"arm64", "aarch64"}:
        return ["m1-apple-silicon"]

    return []


def select_stockfish_asset(
    release: Mapping[str, Any],
    system: SystemInfo,
) -> tuple[Mapping[str, Any], str]:
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise ValueError("GitHub release payload has no assets list")

    name_to_asset: dict[str, Mapping[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, Mapping):
            continue
        name = asset.get("name")
        if isinstance(name, str):
            name_to_asset[name] = asset

    if system.os_name == "linux" and system.machine in {"x86_64", "amd64"}:
        prefix = "stockfish-ubuntu-"
        ext = ".tar"
    elif system.os_name == "darwin" and system.machine in {"x86_64", "amd64"}:
        prefix = "stockfish-macos-"
        ext = ".tar"
    elif system.os_name == "darwin" and system.machine in {"arm64", "aarch64"}:
        candidate = "stockfish-macos-m1-apple-silicon.tar"
        asset = name_to_asset.get(candidate)
        if asset is None:
            raise ValueError(f"No matching Stockfish asset found for {system.os_name}/{system.machine}")
        return asset, candidate
    else:
        raise ValueError(
            f"Unsupported platform for auto Stockfish install: {system.os_name}/{system.machine}"
        )

    for suffix in rank_stockfish_asset_suffixes(system):
        candidate = f"{prefix}{suffix}{ext}"
        asset = name_to_asset.get(candidate)
        if asset is not None:
            return asset, candidate

    raise ValueError(f"No matching Stockfish asset found for {system.os_name}/{system.machine}")
