from __future__ import annotations

import json
import random
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chz
from datasets import load_dataset

from tinker_cookbook.sdpo.chess_hints import (
    StockfishHintConfig,
    StockfishHintExtractor,
    extract_fen_from_text,
    pick_random_game_fen,
    render_hint_text,
)


def _reservoir_sample_rows(
    dataset_name: str,
    sample_size: int,
    seed: int,
    max_scan_rows: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    stream = load_dataset(dataset_name, split="train", streaming=True)

    for idx, row in enumerate(stream):
        if idx >= max_scan_rows:
            break
        if not isinstance(row, Mapping):
            continue
        row_dict = dict(row)
        if len(sampled) < sample_size:
            sampled.append(row_dict)
            continue
        replace_idx = rng.randint(0, idx)
        if replace_idx < sample_size:
            sampled[replace_idx] = row_dict

    return sampled


def _make_prompt_from_fen(fen: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Given this chess position as FEN, choose the best next move.\n\n"
                f"FEN: {fen}\n\n"
                "Return only one legal UCI move."
            ),
        }
    ]


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@chz.chz
class CLIConfig:
    output_path: str = "/tmp/chess_positions.jsonl"
    num_positions: int = 1000
    puzzles_fraction: float = 0.5
    seed: int = 0
    max_scan_rows_per_source: int = 200_000
    oversample_factor: int = 4

    puzzles_dataset: str = "Lichess/chess-puzzles"
    games_dataset: str = "Lichess/standard-chess-games"

    include_stockfish_hints: bool = False
    stockfish_path: str = "stockfish"
    stockfish_depth: int = 14
    stockfish_multipv: int = 5
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 128
    stockfish_wdl_model: str = "sf"
    stockfish_max_pv_plies: int = 6
    stockfish_hint_max_good_moves: int = 3
    stockfish_hint_max_bad_moves: int = 3
    stockfish_hint_bad_move_threshold: float = 0.05


def main(config: CLIConfig) -> None:
    if config.num_positions <= 0:
        raise ValueError(f"num_positions must be > 0, got {config.num_positions}")
    if not 0.0 <= config.puzzles_fraction <= 1.0:
        raise ValueError(f"puzzles_fraction must be in [0, 1], got {config.puzzles_fraction}")
    if config.max_scan_rows_per_source <= 0:
        raise ValueError(
            f"max_scan_rows_per_source must be > 0, got {config.max_scan_rows_per_source}"
        )
    if config.oversample_factor <= 0:
        raise ValueError(f"oversample_factor must be > 0, got {config.oversample_factor}")

    rng = random.Random(config.seed)
    num_puzzles = int(round(config.num_positions * config.puzzles_fraction))
    num_games = max(0, config.num_positions - num_puzzles)

    puzzle_rows = _reservoir_sample_rows(
        dataset_name=config.puzzles_dataset,
        sample_size=max(num_puzzles * config.oversample_factor, num_puzzles),
        seed=rng.randint(0, 2**31 - 1),
        max_scan_rows=config.max_scan_rows_per_source,
    )
    game_rows = _reservoir_sample_rows(
        dataset_name=config.games_dataset,
        sample_size=max(num_games * config.oversample_factor, num_games),
        seed=rng.randint(0, 2**31 - 1),
        max_scan_rows=config.max_scan_rows_per_source,
    )

    stockfish_extractor: StockfishHintExtractor | None = None
    if config.include_stockfish_hints:
        stockfish_extractor = StockfishHintExtractor(
            StockfishHintConfig(
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
            )
        )

    output_rows: list[dict[str, Any]] = []
    try:
        for row in puzzle_rows:
            if len(output_rows) >= num_puzzles:
                break
            fen_value = row.get("FEN")
            fen = fen_value.strip() if isinstance(fen_value, str) else None
            if not fen:
                continue
            first_solution_move = None
            moves_raw = row.get("Moves")
            if isinstance(moves_raw, str):
                split_moves = [part.strip() for part in moves_raw.split() if part.strip()]
                if split_moves:
                    first_solution_move = split_moves[0]
            out_row = {
                "example_id": len(output_rows),
                "task": "chess_next_move",
                "source": "lichess_puzzle",
                "prompt": _make_prompt_from_fen(fen),
                "answer": first_solution_move or "",
                "info": {
                    "fen": fen,
                    "puzzle_id": row.get("PuzzleId"),
                    "themes": row.get("Themes") if isinstance(row.get("Themes"), list) else [],
                    "rating": _safe_int(row.get("Rating")),
                },
            }
            if stockfish_extractor is not None:
                try:
                    pack = stockfish_extractor.analyze_fen(fen)
                    out_row["info"]["stockfish_hint_text"] = render_hint_text(
                        pack, stockfish_extractor.config
                    )
                    if pack.candidate_moves:
                        out_row["info"]["stockfish_best_move"] = pack.candidate_moves[0].uci
                        out_row["info"]["stockfish_best_expected_score"] = (
                            pack.candidate_moves[0].expected_score
                        )
                except Exception:
                    pass
            output_rows.append(out_row)

        for row in game_rows:
            if len(output_rows) >= config.num_positions:
                break
            movetext_value = row.get("movetext")
            if not isinstance(movetext_value, str):
                continue
            fen = pick_random_game_fen(
                movetext_value,
                seed=rng.randint(0, 2**31 - 1),
            )
            if fen is None:
                maybe_fen = extract_fen_from_text(movetext_value)
                if maybe_fen is None:
                    continue
                fen = maybe_fen

            out_row = {
                "example_id": len(output_rows),
                "task": "chess_next_move",
                "source": "lichess_game",
                "prompt": _make_prompt_from_fen(fen),
                "answer": "",
                "info": {
                    "fen": fen,
                    "site": row.get("Site"),
                    "opening": row.get("Opening"),
                    "result": row.get("Result"),
                    "white_elo": _safe_int(row.get("WhiteElo")),
                    "black_elo": _safe_int(row.get("BlackElo")),
                },
            }
            if stockfish_extractor is not None:
                try:
                    pack = stockfish_extractor.analyze_fen(fen)
                    out_row["info"]["stockfish_hint_text"] = render_hint_text(
                        pack, stockfish_extractor.config
                    )
                    if pack.candidate_moves:
                        out_row["info"]["stockfish_best_move"] = pack.candidate_moves[0].uci
                        out_row["info"]["stockfish_best_expected_score"] = (
                            pack.candidate_moves[0].expected_score
                        )
                except Exception:
                    pass
            output_rows.append(out_row)
    finally:
        if stockfish_extractor is not None:
            stockfish_extractor.close()

    rng.shuffle(output_rows)
    for idx, row in enumerate(output_rows):
        row["example_id"] = idx

    output_path = Path(config.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "requested_positions": config.num_positions,
        "generated_positions": len(output_rows),
        "puzzle_rows": sum(1 for row in output_rows if row.get("source") == "lichess_puzzle"),
        "game_rows": sum(1 for row in output_rows if row.get("source") == "lichess_game"),
        "hints_enabled": config.include_stockfish_hints,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    main(config)
