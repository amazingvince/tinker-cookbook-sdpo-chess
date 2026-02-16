from __future__ import annotations

import os
import shutil
import stat
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import chz

from tinker_cookbook.recipes.verifiers_rl.stockfish_helpers import (
    detect_system_info,
    fetch_stockfish_release,
    rank_stockfish_asset_suffixes,
    select_stockfish_asset,
)


@chz.chz
class CLIConfig:
    stockfish_tag: str = "latest"
    install_root: str = "~/.local/stockfish"
    symlink_path: str = "~/.local/bin/stockfish"
    overwrite: bool = False
    dry_run: bool = False


def _download_file(url: str, destination: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "tinker-cookbook-stockfish-installer"})
    with urllib.request.urlopen(req) as resp, destination.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _find_binary(extracted_root: Path) -> Path:
    candidates: list[Path] = []
    archive_suffixes = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".zip", ".bz2", ".7z")
    for path in extracted_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(archive_suffixes):
            continue
        if name.startswith("stockfish-") or name == "stockfish":
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError("Could not locate extracted Stockfish binary")

    # Prefer canonical executable name, then executable bit, then shorter paths.
    candidates.sort(
        key=lambda p: (
            0 if p.name == "stockfish" else 1,
            0 if os.access(p, os.X_OK) else 1,
            len(str(p)),
        )
    )
    return candidates[0]


def _set_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main(config: CLIConfig) -> None:
    system = detect_system_info()
    release = fetch_stockfish_release(config.stockfish_tag)
    asset, asset_name = select_stockfish_asset(release, system)

    tag_name = release.get("tag_name")
    if not isinstance(tag_name, str):
        tag_name = config.stockfish_tag
    download_url = asset.get("browser_download_url")
    if not isinstance(download_url, str):
        raise ValueError(f"Malformed asset payload for {asset_name}: missing browser_download_url")

    install_root = Path(config.install_root).expanduser().resolve()
    version_dir = install_root / tag_name
    binary_target = version_dir / asset_name.replace(".tar", "")
    # Avoid resolving symlink_path; we want to manage the link path itself, not its current target.
    symlink_path = Path(config.symlink_path).expanduser()
    if not symlink_path.is_absolute():
        symlink_path = (Path.cwd() / symlink_path).resolve()

    if binary_target.exists() and not config.overwrite:
        print(f"Stockfish already installed at {binary_target} (set overwrite=true to replace)")
        print(f"symlink_path={symlink_path}")
        print(f"stockfish_path={binary_target}")
        return

    if config.dry_run:
        print("Dry run:")
        print(f"  os={system.os_name} machine={system.machine}")
        print(f"  cpu_count={system.cpu_count} memory_mb={system.memory_mb}")
        print(f"  ranked_suffixes={rank_stockfish_asset_suffixes(system)}")
        print(f"  selected_asset={asset_name}")
        print(f"  download_url={download_url}")
        print(f"  install_target={binary_target}")
        print(f"  symlink_path={symlink_path}")
        return

    version_dir.mkdir(parents=True, exist_ok=True)
    symlink_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="stockfish_install_") as tmp:
        tmpdir = Path(tmp)
        archive_path = tmpdir / asset_name
        _download_file(download_url, archive_path)
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(path=tmpdir)
        extracted_binary = _find_binary(tmpdir)
        shutil.copy2(extracted_binary, binary_target)
        _set_executable(binary_target)

    tmp_link = symlink_path.with_suffix(".tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(binary_target)
    os.replace(tmp_link, symlink_path)

    print(f"Installed {asset_name} to {binary_target}")
    print(f"Updated symlink: {symlink_path} -> {binary_target}")
    print(f"Use in SDPO config: stockfish_path={binary_target}")


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    main(cfg)
