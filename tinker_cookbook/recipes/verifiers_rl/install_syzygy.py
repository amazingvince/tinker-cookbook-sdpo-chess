from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chz

_BASE_URL = "https://tablebase.lichess.ovh/tables/standard/"


@chz.chz
class CLIConfig:
    output_dir: str = "~/.local/share/syzygy/standard"
    pieces: str = "345"  # one of: 345, 6, 7, 3456, all
    include_wdl: bool = True
    include_dtz: bool = True
    parallel_downloads: int = 8
    max_files: int = -1
    verify_sha256: bool = True
    skip_existing: bool = True
    dry_run: bool = False
    continue_on_error: bool = True


def _fetch_lines(url: str) -> list[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "tinker-cookbook-syzygy-installer"})
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_sha256_map(lines: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            digest = parts[0].strip().lower()
            name = parts[-1].strip()
            mapping[name] = digest
    return mapping


def _selected_prefixes(config: CLIConfig) -> list[str]:
    pieces = config.pieces.lower()
    include_345 = pieces in {"345", "3456", "all"}
    include_6 = pieces in {"6", "3456", "all"}
    include_7 = pieces in {"7", "all"}

    prefixes: list[str] = []
    if include_345:
        if config.include_wdl:
            prefixes.append("/3-4-5-wdl/")
        if config.include_dtz:
            prefixes.append("/3-4-5-dtz/")
    if include_6:
        if config.include_wdl:
            prefixes.append("/6-wdl/")
        if config.include_dtz:
            prefixes.append("/6-dtz/")
    if include_7:
        prefixes.append("/7/")
    return prefixes


def _filter_urls(urls: list[str], prefixes: list[str]) -> list[str]:
    return [url for url in urls if any(prefix in url for prefix in prefixes)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().lower()


def _download_url_to_path(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        req = urllib.request.Request(url, headers={"User-Agent": "tinker-cookbook-syzygy-installer"})
        with urllib.request.urlopen(req) as resp:
            shutil.copyfileobj(resp, tmp)
    tmp_path.replace(target)


def main(config: CLIConfig) -> None:
    if config.parallel_downloads <= 0:
        raise ValueError(f"parallel_downloads must be >= 1, got {config.parallel_downloads}")

    prefixes = _selected_prefixes(config)
    if not prefixes:
        raise ValueError(
            f"No tablebase prefixes selected for pieces={config.pieces!r}, "
            f"include_wdl={config.include_wdl}, include_dtz={config.include_dtz}"
        )

    download_urls = _fetch_lines(_BASE_URL + "download.txt")
    selected_urls = _filter_urls(download_urls, prefixes)
    if config.max_files > 0:
        selected_urls = selected_urls[: config.max_files]

    sha256_map: dict[str, str] = {}
    if config.verify_sha256:
        sha256_map = _parse_sha256_map(_fetch_lines(_BASE_URL + "sha256"))

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.dry_run:
        estimated = {
            "output_dir": str(output_dir),
            "num_urls": len(selected_urls),
            "prefixes": prefixes,
            "examples": selected_urls[:10],
        }
        print(json.dumps(estimated, ensure_ascii=True, indent=2))
        return

    num_downloaded = 0
    num_skipped = 0
    num_failed = 0
    failures: list[str] = []

    def _job(url: str) -> tuple[str, str]:
        filename = url.rsplit("/", 1)[-1]
        target = output_dir / filename
        if target.exists() and config.skip_existing:
            if config.verify_sha256 and filename in sha256_map:
                current_digest = _sha256_file(target)
                expected_digest = sha256_map[filename]
                if current_digest == expected_digest:
                    return "skipped", filename
            else:
                return "skipped", filename

        _download_url_to_path(url, target)

        if config.verify_sha256 and filename in sha256_map:
            current_digest = _sha256_file(target)
            expected_digest = sha256_map[filename]
            if current_digest != expected_digest:
                target.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA256 mismatch for {filename}: expected {expected_digest}, got {current_digest}"
                )

        return "downloaded", filename

    with ThreadPoolExecutor(max_workers=config.parallel_downloads) as executor:
        futures = [executor.submit(_job, url) for url in selected_urls]
        for future in as_completed(futures):
            try:
                status, _name = future.result()
                if status == "downloaded":
                    num_downloaded += 1
                else:
                    num_skipped += 1
            except Exception as exc:
                num_failed += 1
                failures.append(str(exc))
                if not config.continue_on_error:
                    raise

    summary = {
        "output_dir": str(output_dir),
        "requested_urls": len(selected_urls),
        "downloaded": num_downloaded,
        "skipped": num_skipped,
        "failed": num_failed,
        "syzygy_path_for_sdpo": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if failures:
        print("Failures:")
        for item in failures[:20]:
            print(f"- {item}")
        if len(failures) > 20:
            print(f"- ... and {len(failures) - 20} more")


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    main(cfg)
