from __future__ import annotations

import asyncio
import io
import itertools
import logging
import math
import random
import re
import threading
from collections.abc import Mapping
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

import verifiers as vf
from tinker_cookbook.sdpo.chess_hints import (
    MoveVerification,
    StockfishHintConfig,
    StockfishHintExtractor,
)

try:
    import chess
    import chess.pgn
except ImportError:  # pragma: no cover - guarded at runtime
    chess = None

_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", flags=re.IGNORECASE)
_UCI_FULL_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", flags=re.IGNORECASE)
_GAME_VERIFICATION_CACHE_KEY = "_hf_chess_mix_stockfish_verification"
logger = logging.getLogger(__name__)


def _build_stockfish_config(
    *,
    stockfish_path: str,
    stockfish_depth: int,
    stockfish_multipv: int,
    stockfish_threads: int,
    stockfish_hash_mb: int,
    stockfish_wdl_model: str,
    stockfish_syzygy_path: str | None,
    stockfish_syzygy_max_pieces: int,
    stockfish_persistent_cache_dir: str | None,
) -> StockfishHintConfig:
    return StockfishHintConfig(
        stockfish_path=stockfish_path,
        depth=max(1, stockfish_depth),
        multipv=max(1, stockfish_multipv),
        threads=max(1, stockfish_threads),
        hash_mb=max(1, stockfish_hash_mb),
        wdl_model=stockfish_wdl_model,
        include_fen_decode=False,
        include_ascii_board=False,
        include_search_stats=False,
        max_good_moves=1,
        max_bad_moves=0,
        bad_move_threshold=1.0,
        syzygy_path=stockfish_syzygy_path,
        syzygy_max_pieces=max(1, stockfish_syzygy_max_pieces),
        persistent_cache_dir=stockfish_persistent_cache_dir,
    )


class _StockfishBestMoveOracle:
    def __init__(self, config: StockfishHintConfig):
        self._extractor = StockfishHintExtractor(config)
        self._cache: dict[str, tuple[str, float | None]] = {}

    def get_best_move(self, fen: str) -> tuple[str, float | None] | None:
        cached = self._cache.get(fen)
        if cached is not None:
            return cached
        pack = self._extractor.analyze_fen(fen, multipv=1)
        if not pack.candidate_moves:
            return None
        best = pack.candidate_moves[0]
        result = (best.uci, best.expected_score)
        self._cache[fen] = result
        return result

    def close(self) -> None:
        self._extractor.close()


class _AsyncStockfishVerifierPool:
    def __init__(self, config: StockfishHintConfig, num_workers: int, verification_multipv: int):
        if num_workers <= 0:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self._config = config
        self._num_workers = int(num_workers)
        self._verification_multipv = max(1, int(verification_multipv))
        self._workers: list[StockfishHintExtractor] = []
        self._locks: list[threading.Lock] = []
        self._init_lock = threading.Lock()
        self._rr_counter = itertools.count()
        self._closed = False

    def _ensure_workers(self) -> None:
        if self._workers:
            return
        with self._init_lock:
            if self._workers:
                return
            self._workers = [StockfishHintExtractor(self._config) for _ in range(self._num_workers)]
            self._locks = [threading.Lock() for _ in range(self._num_workers)]

    async def verify_predicted_move_async(
        self,
        *,
        fen: str,
        predicted_text: str,
        depth: int,
        illegal_move_cp_loss: float,
    ) -> MoveVerification:
        if self._closed:
            raise RuntimeError("Async stockfish verifier pool is closed")
        self._ensure_workers()
        idx = next(self._rr_counter) % len(self._workers)
        worker = self._workers[idx]
        lock = self._locks[idx]

        def _run() -> MoveVerification:
            with lock:
                return worker.verify_predicted_move(
                    fen=fen,
                    predicted_text=predicted_text,
                    depth=depth,
                    illegal_move_cp_loss=illegal_move_cp_loss,
                    verification_multipv=self._verification_multipv,
                )

        return await asyncio.to_thread(_run)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass
        self._workers = []
        self._locks = []


def _as_stripped(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass
    return str(value)


def _extract_first_uci(value: Any) -> str | None:
    if isinstance(value, str):
        match = _UCI_RE.search(value.lower())
        return match.group(1) if match else None
    if isinstance(value, list):
        for item in value:
            found = _extract_first_uci(item)
            if found:
                return found
    return None


def _extract_uci_moves(value: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(value, str):
        tokens = [part.strip().lower() for part in value.split() if part.strip()]
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                tokens.extend(part.strip().lower() for part in item.split() if part.strip())
    return [token for token in tokens if _UCI_FULL_RE.fullmatch(token) is not None]


def _is_valid_fen(fen: str) -> bool:
    if chess is None:
        return False
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False


def _make_prompt_from_fen(fen: str) -> list[dict[str, str]]:
    side_to_move = "Unknown"
    if chess is not None:
        try:
            board = chess.Board(fen)
            side_to_move = "White" if board.turn == chess.WHITE else "Black"
        except ValueError:
            pass

    return [
        {
            "role": "user",
            "content": (
                "You are analyzing a chess position.\n"
                "Choose the strongest next move for the side to move.\n"
                "Return only one legal move in UCI format (for example, e2e4 or e7e8q).\n\n"
                f"FEN: {fen}\n"
                f"Side to move: {side_to_move}"
            ),
        }
    ]


def _parse_puzzle_row(
    row: Mapping[str, Any],
    min_puzzle_rating: int,
    puzzle_solver_moves_only: bool,
    max_puzzle_solver_moves_per_puzzle: int,
) -> list[dict[str, Any]]:
    if chess is None:
        raise RuntimeError("python-chess is required for hf-chess-mix")

    fen = _as_stripped(row.get("FEN") or row.get("fen"))
    if fen is None or not _is_valid_fen(fen):
        return []

    rating = _safe_int(row.get("Rating") or row.get("rating"))
    if min_puzzle_rating > 0 and rating is not None and rating < min_puzzle_rating:
        return []

    moves = _extract_uci_moves(row.get("Moves") or row.get("moves"))
    if not moves:
        return []

    themes = row.get("Themes")
    if isinstance(themes, list):
        normalized_themes = [str(x) for x in themes]
    else:
        normalized_themes = []

    board = chess.Board(fen)
    solver_color = board.turn
    selected_count = 0
    examples: list[dict[str, Any]] = []

    for ply_index, answer in enumerate(moves):
        try:
            move = chess.Move.from_uci(answer)
        except ValueError:
            break
        if move not in board.legal_moves:
            break

        is_solver_turn = board.turn == solver_color
        should_include = is_solver_turn or not puzzle_solver_moves_only
        if should_include:
            examples.append(
                {
                    "task": "chess_next_move",
                    "prompt": _make_prompt_from_fen(board.fen()),
                    "answer": answer,
                    "info": {
                        "fen": board.fen(),
                        "source": "lichess_puzzle",
                        "puzzle_id": _metadata_value(row.get("PuzzleId")),
                        "game_id": _metadata_value(row.get("GameId")),
                        "rating": rating,
                        "themes": normalized_themes,
                        "puzzle_ply_index": ply_index,
                        "puzzle_selected_move_index": selected_count,
                        "puzzle_total_line_plies": len(moves),
                        "puzzle_solver_side": "white" if solver_color == chess.WHITE else "black",
                        "puzzle_solver_move": bool(is_solver_turn),
                    },
                }
            )
            selected_count += 1
            if (
                max_puzzle_solver_moves_per_puzzle > 0
                and selected_count >= max_puzzle_solver_moves_per_puzzle
            ):
                break

        board.push(move)

    return examples


def _parse_game_row(
    row: Mapping[str, Any],
    rng: random.Random,
    min_game_ply: int,
    max_game_ply: int,
    min_game_average_elo: int,
    game_positions_per_game: int,
    game_answer_mode: str,
    best_move_oracle: _StockfishBestMoveOracle | None,
) -> list[dict[str, Any]]:
    if chess is None:
        raise RuntimeError("python-chess is required for hf-chess-mix")

    movetext = _as_stripped(row.get("movetext") or row.get("pgn") or row.get("moves"))
    if movetext is None:
        return []

    white_elo = _safe_int(row.get("WhiteElo") or row.get("white_elo"))
    black_elo = _safe_int(row.get("BlackElo") or row.get("black_elo"))
    if white_elo is not None and black_elo is not None:
        average_elo = (white_elo + black_elo) / 2.0
    else:
        average_elo = None

    if (
        min_game_average_elo > 0
        and average_elo is not None
        and average_elo < float(min_game_average_elo)
    ):
        return []

    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None:
        return []
    moves = list(game.mainline_moves())
    if not moves:
        return []

    lower = max(0, min_game_ply)
    upper = len(moves) - 1 if max_game_ply < 0 else min(max_game_ply, len(moves) - 1)
    if lower > upper:
        return []

    board = game.board()
    candidates: list[tuple[int, str, str]] = []
    for ply_index, move in enumerate(moves):
        if lower <= ply_index <= upper:
            candidates.append((ply_index, board.fen(), move.uci()))
        board.push(move)
    if not candidates:
        return []

    if game_positions_per_game < 0:
        selected = list(candidates)
    else:
        k = min(max(1, game_positions_per_game), len(candidates))
        selected = rng.sample(candidates, k=k)
    selected.sort(key=lambda item: item[0])

    examples: list[dict[str, Any]] = []
    for local_index, (ply_index, fen, pgn_move) in enumerate(selected):
        answer = pgn_move
        stockfish_best_move: str | None = None
        stockfish_best_expected_score: float | None = None
        if game_answer_mode == "stockfish":
            if best_move_oracle is None:
                raise ValueError("best_move_oracle is required when game_answer_mode='stockfish'")
            best = best_move_oracle.get_best_move(fen)
            if best is None:
                continue
            stockfish_best_move, stockfish_best_expected_score = best
            answer = stockfish_best_move

        examples.append(
            {
                "task": "chess_next_move",
                "prompt": _make_prompt_from_fen(fen),
                "answer": answer,
                "info": {
                    "fen": fen,
                    "source": "lichess_game",
                    "ply_index": ply_index,
                    "sampled_move_index_in_game": local_index,
                    "sampled_positions_from_game": len(selected),
                    "total_plies": len(moves),
                    "game_answer_mode": game_answer_mode,
                    "pgn_next_move": pgn_move,
                    "stockfish_best_move": stockfish_best_move,
                    "stockfish_best_expected_score": stockfish_best_expected_score,
                    "site": _metadata_value(row.get("Site")),
                    "opening": _metadata_value(row.get("Opening")),
                    "result": _metadata_value(row.get("Result")),
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "average_elo": average_elo,
                    "event": _metadata_value(row.get("Event")),
                    "eco": _metadata_value(row.get("ECO")),
                    "time_control": _metadata_value(row.get("TimeControl")),
                    "utc_date": _metadata_value(row.get("UTCDate")),
                    "utc_time": _metadata_value(row.get("UTCTime")),
                },
            }
        )
    return examples


def _iter_dataset_rows(
    dataset_name: str,
    *,
    seed: int,
    shuffle_buffer_size: int,
) -> IterableDataset:
    stream = load_dataset(dataset_name, split="train", streaming=True)
    if shuffle_buffer_size > 1:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return stream


def _collect_examples(
    dataset_name: str,
    target_count: int,
    max_scan_rows: int,
    seed: int,
    shuffle_buffer_size: int,
    parser: Any,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []

    logger.info(
        "hf-chess-mix: begin streaming dataset=%s target_count=%d max_scan_rows=%d seed=%d shuffle_buffer_size=%d",
        dataset_name,
        target_count,
        max_scan_rows,
        seed,
        shuffle_buffer_size,
    )
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    scanned = 0
    log_every = max(1000, min(50000, max_scan_rows // 20 if max_scan_rows > 0 else 10000))
    stream = _iter_dataset_rows(
        dataset_name=dataset_name,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    for row in stream:
        scanned += 1
        if scanned % log_every == 0:
            logger.info(
                "hf-chess-mix: dataset=%s scanned=%d collected=%d/%d",
                dataset_name,
                scanned,
                len(rows),
                target_count,
            )
        if scanned > max_scan_rows:
            logger.info(
                "hf-chess-mix: dataset=%s reached max_scan_rows=%d collected=%d/%d",
                dataset_name,
                max_scan_rows,
                len(rows),
                target_count,
            )
            break
        if not isinstance(row, Mapping):
            continue
        parsed = parser(row, rng)
        if not parsed:
            continue
        remaining = target_count - len(rows)
        if remaining <= 0:
            break
        if len(parsed) > remaining:
            parsed_copy = list(parsed)
            rng.shuffle(parsed_copy)
            rows.extend(parsed_copy[:remaining])
            break
        rows.extend(parsed)
        if len(rows) >= target_count:
            break
    logger.info(
        "hf-chess-mix: completed dataset=%s scanned=%d collected=%d/%d",
        dataset_name,
        scanned,
        len(rows),
        target_count,
    )
    return rows


def _assemble_mixed_examples(
    puzzle_rows: list[dict[str, Any]],
    game_rows: list[dict[str, Any]],
    max_examples: int,
    puzzles_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []

    rng = random.Random(seed)
    rng.shuffle(puzzle_rows)
    rng.shuffle(game_rows)

    target_puzzles = int(round(max_examples * puzzles_fraction))
    target_games = max_examples - target_puzzles
    selected = puzzle_rows[:target_puzzles] + game_rows[:target_games]

    if len(selected) < max_examples:
        leftovers = puzzle_rows[target_puzzles:] + game_rows[target_games:]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_examples - len(selected)])

    rng.shuffle(selected)
    for idx, row in enumerate(selected):
        row["example_id"] = idx
    return selected


def _build_dataset(
    *,
    max_examples: int,
    puzzles_fraction: float,
    seed: int,
    puzzles_dataset: str,
    games_dataset: str,
    max_scan_rows_per_source: int,
    oversample_factor: int,
    shuffle_buffer_size: int,
    min_puzzle_rating: int,
    puzzle_solver_moves_only: bool,
    max_puzzle_solver_moves_per_puzzle: int,
    min_game_ply: int,
    max_game_ply: int,
    min_game_average_elo: int,
    game_positions_per_game: int,
    game_answer_mode: str,
    stockfish_config: StockfishHintConfig,
    shuffle: bool,
) -> Dataset:
    pool_size = max(max_examples, max_examples * oversample_factor)
    puzzle_pool_target = int(round(pool_size * puzzles_fraction))
    game_pool_target = max(0, pool_size - puzzle_pool_target)
    logger.info(
        "hf-chess-mix: building dataset max_examples=%d pool_size=%d puzzles_fraction=%.3f puzzle_pool_target=%d game_pool_target=%d",
        max_examples,
        pool_size,
        puzzles_fraction,
        puzzle_pool_target,
        game_pool_target,
    )

    puzzle_rows = _collect_examples(
        dataset_name=puzzles_dataset,
        target_count=puzzle_pool_target,
        max_scan_rows=max_scan_rows_per_source,
        seed=seed + 11,
        shuffle_buffer_size=shuffle_buffer_size,
        parser=lambda row, _rng: _parse_puzzle_row(
            row,
            min_puzzle_rating=min_puzzle_rating,
            puzzle_solver_moves_only=puzzle_solver_moves_only,
            max_puzzle_solver_moves_per_puzzle=max_puzzle_solver_moves_per_puzzle,
        ),
    )
    logger.info(
        "hf-chess-mix: collected puzzle candidate rows=%d from %s",
        len(puzzle_rows),
        puzzles_dataset,
    )
    best_move_oracle: _StockfishBestMoveOracle | None = None
    try:
        if game_answer_mode == "stockfish" and game_pool_target > 0:
            logger.info("hf-chess-mix: initializing Stockfish best-move oracle for game labeling")
            best_move_oracle = _StockfishBestMoveOracle(stockfish_config)

        game_rows = _collect_examples(
            dataset_name=games_dataset,
            target_count=game_pool_target,
            max_scan_rows=max_scan_rows_per_source,
            seed=seed + 29,
            shuffle_buffer_size=shuffle_buffer_size,
            parser=lambda row, rng: _parse_game_row(
                row=row,
                rng=rng,
                min_game_ply=min_game_ply,
                max_game_ply=max_game_ply,
                min_game_average_elo=min_game_average_elo,
                game_positions_per_game=game_positions_per_game,
                game_answer_mode=game_answer_mode,
                best_move_oracle=best_move_oracle,
            ),
        )
        logger.info(
            "hf-chess-mix: collected game candidate rows=%d from %s",
            len(game_rows),
            games_dataset,
        )
    finally:
        if best_move_oracle is not None:
            best_move_oracle.close()

    mixed_rows = _assemble_mixed_examples(
        puzzle_rows=puzzle_rows,
        game_rows=game_rows,
        max_examples=max_examples,
        puzzles_fraction=puzzles_fraction,
        seed=seed,
    )
    logger.info("hf-chess-mix: mixed final dataset rows=%d", len(mixed_rows))
    dataset = Dataset.from_list(mixed_rows)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    logger.info("hf-chess-mix: dataset construction complete rows=%d", len(dataset))
    return dataset


def _normalize_uci(text: str) -> str:
    cleaned = text.strip().lower()
    match = _UCI_RE.search(cleaned)
    return match.group(1) if match else cleaned


def _game_quality_from_verification(
    verification: MoveVerification,
    expected_score_temperature: float,
    cp_loss_scale: float,
) -> float:
    if not verification.move_is_legal:
        return 0.0

    if (
        verification.best_expected_score is not None
        and verification.predicted_expected_score is not None
    ):
        delta = max(0.0, verification.best_expected_score - verification.predicted_expected_score)
        return math.exp(-delta / max(expected_score_temperature, 1e-6))

    cp_loss = max(0.0, float(verification.cp_loss))
    return math.exp(-cp_loss / max(cp_loss_scale, 1e-6))


def _syzygy_wdl_penalty(verification: MoveVerification) -> float:
    if verification.syzygy_best_wdl is None or verification.syzygy_predicted_wdl is None:
        return 0.0
    return max(0.0, float(verification.syzygy_best_wdl - verification.syzygy_predicted_wdl))


def _syzygy_dtz_penalty(verification: MoveVerification) -> float:
    if (
        verification.syzygy_best_dtz is None
        or verification.syzygy_predicted_dtz is None
        or verification.syzygy_best_wdl != verification.syzygy_predicted_wdl
    ):
        return 0.0
    return max(
        0.0,
        float(abs(verification.syzygy_predicted_dtz) - abs(verification.syzygy_best_dtz)),
    )


def _pv_motif_overlap(verification: MoveVerification, motif_plies: int) -> float:
    if motif_plies <= 0:
        return 0.0
    best_tail = tuple(verification.best_pv_san[1 : 1 + motif_plies])
    pred_tail = tuple(verification.predicted_pv_san[1 : 1 + motif_plies])
    if not best_tail or not pred_tail:
        return 0.0
    best_set = set(best_tail)
    pred_set = set(pred_tail)
    union = best_set | pred_set
    if not union:
        return 0.0
    return len(best_set & pred_set) / len(union)


def _search_confidence(
    verification: MoveVerification,
    target_depth: int,
    nodes_reference: int,
    seldepth_factor: float,
) -> float:
    target = max(1, int(target_depth))
    depth_candidates = [
        v for v in (verification.best_search_depth, verification.predicted_search_depth) if v is not None
    ]
    seldepth_candidates = [
        v
        for v in (verification.best_selective_depth, verification.predicted_selective_depth)
        if v is not None
    ]
    node_candidates = [v for v in (verification.best_nodes, verification.predicted_nodes) if v is not None]

    components: list[float] = []
    if depth_candidates:
        components.append(min(depth_candidates) / float(target))
    if seldepth_candidates:
        seldepth_target = max(1.0, target * max(seldepth_factor, 1e-6))
        components.append(min(seldepth_candidates) / seldepth_target)
    if node_candidates:
        node_ref = max(1, int(nodes_reference))
        node_component = math.log10(max(1, min(node_candidates)) + 1.0) / math.log10(
            float(node_ref) + 1.0
        )
        components.append(node_component)

    if not components:
        return 1.0
    return max(0.0, min(1.0, sum(components) / len(components)))


class ChessMoveRubric(vf.Rubric):
    def __init__(
        self,
        *,
        game_stockfish_pool: _AsyncStockfishVerifierPool | None,
        game_stockfish_depth: int,
        game_reward_illegal_move_cp_loss: float,
        game_reward_legal_floor: float,
        game_reward_best_move_bonus: float,
        game_reward_expected_score_temperature: float,
        game_reward_cp_loss_scale: float,
        game_reward_syzygy_wdl_scale: float,
        game_reward_syzygy_dtz_scale: float,
        game_reward_pv_overlap_bonus: float,
        game_reward_pv_motif_plies: int,
        game_reward_use_confidence_weighting: bool,
        game_reward_confidence_neutral: float,
        game_reward_confidence_nodes_reference: int,
        game_reward_confidence_seldepth_factor: float,
    ):
        super().__init__()
        self.game_stockfish_pool = game_stockfish_pool
        self.game_stockfish_depth = int(game_stockfish_depth)
        self.game_reward_illegal_move_cp_loss = float(game_reward_illegal_move_cp_loss)
        self.game_reward_legal_floor = float(game_reward_legal_floor)
        self.game_reward_best_move_bonus = float(game_reward_best_move_bonus)
        self.game_reward_expected_score_temperature = float(game_reward_expected_score_temperature)
        self.game_reward_cp_loss_scale = float(game_reward_cp_loss_scale)
        self.game_reward_syzygy_wdl_scale = float(game_reward_syzygy_wdl_scale)
        self.game_reward_syzygy_dtz_scale = float(game_reward_syzygy_dtz_scale)
        self.game_reward_pv_overlap_bonus = float(game_reward_pv_overlap_bonus)
        self.game_reward_pv_motif_plies = int(game_reward_pv_motif_plies)
        self.game_reward_use_confidence_weighting = bool(game_reward_use_confidence_weighting)
        self.game_reward_confidence_neutral = float(game_reward_confidence_neutral)
        self.game_reward_confidence_nodes_reference = int(game_reward_confidence_nodes_reference)
        self.game_reward_confidence_seldepth_factor = float(
            game_reward_confidence_seldepth_factor
        )

        self.add_reward_func(self.chess_move_reward)
        self.add_metric(self.game_move_legal_metric)
        self.add_metric(self.game_stockfish_quality_metric)
        self.add_metric(self.game_best_move_metric)
        self.add_metric(self.game_cp_loss_metric)
        self.add_metric(self.game_pv_overlap_metric)
        self.add_metric(self.game_syzygy_wdl_penalty_metric)
        self.add_metric(self.game_syzygy_dtz_penalty_metric)
        self.add_metric(self.game_search_confidence_metric)

    @staticmethod
    def _source(info: Mapping[str, Any]) -> str:
        return str(info.get("source", "")).strip().lower()

    @staticmethod
    def _is_game_source(info: Mapping[str, Any]) -> bool:
        return ChessMoveRubric._source(info) == "lichess_game"

    async def _get_game_verification(
        self,
        *,
        parser: vf.Parser,
        completion: vf.Messages,
        info: Mapping[str, Any],
        state: vf.State,
    ) -> MoveVerification | None:
        if self.game_stockfish_pool is None:
            return None
        cached = state.get(_GAME_VERIFICATION_CACHE_KEY)
        if isinstance(cached, MoveVerification):
            return cached

        fen = _as_stripped(info.get("fen"))
        if fen is None:
            return None
        predicted_text = parser.parse_answer(completion) or ""
        verification = await self.game_stockfish_pool.verify_predicted_move_async(
            fen=fen,
            predicted_text=predicted_text,
            depth=self.game_stockfish_depth,
            illegal_move_cp_loss=self.game_reward_illegal_move_cp_loss,
        )
        state[_GAME_VERIFICATION_CACHE_KEY] = verification
        return verification

    async def chess_move_reward(
        self,
        parser: vf.Parser,
        completion: vf.Messages,
        answer: str,
        info: Mapping[str, Any],
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        predicted = _normalize_uci(parser.parse_answer(completion) or "")
        expected = _normalize_uci(answer)
        if not self._is_game_source(info):
            return 1.0 if predicted == expected else 0.0

        verification = await self._get_game_verification(
            parser=parser,
            completion=completion,
            info=info,
            state=state,
        )
        if verification is None:
            return 1.0 if predicted == expected else 0.0
        if not verification.move_is_legal:
            return 0.0

        quality = _game_quality_from_verification(
            verification=verification,
            expected_score_temperature=self.game_reward_expected_score_temperature,
            cp_loss_scale=self.game_reward_cp_loss_scale,
        )
        syzygy_wdl_penalty = _syzygy_wdl_penalty(verification)
        if self.game_reward_syzygy_wdl_scale > 0.0:
            quality *= math.exp(-syzygy_wdl_penalty / self.game_reward_syzygy_wdl_scale)

        syzygy_dtz_penalty = _syzygy_dtz_penalty(verification)
        if self.game_reward_syzygy_dtz_scale > 0.0:
            quality *= math.exp(-syzygy_dtz_penalty / self.game_reward_syzygy_dtz_scale)

        pv_overlap = _pv_motif_overlap(verification, motif_plies=self.game_reward_pv_motif_plies)
        quality += self.game_reward_pv_overlap_bonus * pv_overlap
        quality = max(0.0, min(1.0, quality))

        if self.game_reward_use_confidence_weighting:
            confidence = _search_confidence(
                verification=verification,
                target_depth=self.game_stockfish_depth,
                nodes_reference=self.game_reward_confidence_nodes_reference,
                seldepth_factor=self.game_reward_confidence_seldepth_factor,
            )
            neutral = max(0.0, min(1.0, self.game_reward_confidence_neutral))
            quality = neutral + (confidence * (quality - neutral))
            quality = max(0.0, min(1.0, quality))

        reward = self.game_reward_legal_floor + ((1.0 - self.game_reward_legal_floor) * quality)
        if (
            verification.best_move_uci is not None
            and verification.predicted_move_uci == verification.best_move_uci
        ):
            reward += self.game_reward_best_move_bonus
        return max(0.0, min(1.0, reward))

    async def game_move_legal_metric(
        self,
        parser: vf.Parser,
        completion: vf.Messages,
        info: Mapping[str, Any],
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if not self._is_game_source(info):
            return 0.0
        verification = await self._get_game_verification(
            parser=parser,
            completion=completion,
            info=info,
            state=state,
        )
        return 1.0 if verification is not None and verification.move_is_legal else 0.0

    async def game_stockfish_quality_metric(
        self,
        parser: vf.Parser,
        completion: vf.Messages,
        info: Mapping[str, Any],
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if not self._is_game_source(info):
            return 0.0
        verification = await self._get_game_verification(
            parser=parser,
            completion=completion,
            info=info,
            state=state,
        )
        if verification is None:
            return 0.0
        return _game_quality_from_verification(
            verification=verification,
            expected_score_temperature=self.game_reward_expected_score_temperature,
            cp_loss_scale=self.game_reward_cp_loss_scale,
        )

    async def game_best_move_metric(
        self,
        parser: vf.Parser,
        completion: vf.Messages,
        info: Mapping[str, Any],
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if not self._is_game_source(info):
            return 0.0
        verification = await self._get_game_verification(
            parser=parser,
            completion=completion,
            info=info,
            state=state,
        )
        if verification is None or verification.best_move_uci is None:
            return 0.0
        return 1.0 if verification.predicted_move_uci == verification.best_move_uci else 0.0

    async def game_cp_loss_metric(
        self,
        parser: vf.Parser,
        completion: vf.Messages,
        info: Mapping[str, Any],
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if not self._is_game_source(info):
            return 0.0
        verification = await self._get_game_verification(
            parser=parser,
            completion=completion,
            info=info,
            state=state,
        )
        return float(verification.cp_loss) if verification is not None else 0.0

    async def game_pv_overlap_metric(
        self,
        parser: vf.Parser,
        completion: vf.Messages,
        info: Mapping[str, Any],
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if not self._is_game_source(info):
            return 0.0
        verification = await self._get_game_verification(
            parser=parser,
            completion=completion,
            info=info,
            state=state,
        )
        if verification is None:
            return 0.0
        return _pv_motif_overlap(verification, motif_plies=self.game_reward_pv_motif_plies)

    async def game_syzygy_wdl_penalty_metric(
        self,
        parser: vf.Parser,
        completion: vf.Messages,
        info: Mapping[str, Any],
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if not self._is_game_source(info):
            return 0.0
        verification = await self._get_game_verification(
            parser=parser,
            completion=completion,
            info=info,
            state=state,
        )
        if verification is None:
            return 0.0
        return _syzygy_wdl_penalty(verification)

    async def game_syzygy_dtz_penalty_metric(
        self,
        parser: vf.Parser,
        completion: vf.Messages,
        info: Mapping[str, Any],
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if not self._is_game_source(info):
            return 0.0
        verification = await self._get_game_verification(
            parser=parser,
            completion=completion,
            info=info,
            state=state,
        )
        if verification is None:
            return 0.0
        return _syzygy_dtz_penalty(verification)

    async def game_search_confidence_metric(
        self,
        parser: vf.Parser,
        completion: vf.Messages,
        info: Mapping[str, Any],
        state: vf.State,
        **kwargs: Any,
    ) -> float:
        if not self._is_game_source(info):
            return 0.0
        verification = await self._get_game_verification(
            parser=parser,
            completion=completion,
            info=info,
            state=state,
        )
        if verification is None:
            return 0.0
        return _search_confidence(
            verification=verification,
            target_depth=self.game_stockfish_depth,
            nodes_reference=self.game_reward_confidence_nodes_reference,
            seldepth_factor=self.game_reward_confidence_seldepth_factor,
        )


class HFChessMixEnv(vf.SingleTurnEnv):
    def __init__(
        self,
        *args: Any,
        game_stockfish_pool: _AsyncStockfishVerifierPool | None,
        **kwargs: Any,
    ):
        self._game_stockfish_pool = game_stockfish_pool
        super().__init__(*args, **kwargs)

    @vf.teardown
    async def close_game_stockfish_pool(self) -> None:
        if self._game_stockfish_pool is None:
            return
        self._game_stockfish_pool.close()
        self._game_stockfish_pool = None


def load_environment(
    max_examples: int = 10000,
    seed: int = 0,
    shuffle: bool = True,
    puzzles_fraction: float = 0.5,
    puzzles_dataset: str = "Lichess/chess-puzzles",
    games_dataset: str = "Lichess/standard-chess-games",
    max_scan_rows_per_source: int = 200000,
    oversample_factor: int = 4,
    shuffle_buffer_size: int = 10000,
    min_puzzle_rating: int = 0,
    puzzle_solver_moves_only: bool = True,
    max_puzzle_solver_moves_per_puzzle: int = -1,
    min_game_ply: int = 0,
    max_game_ply: int = 120,
    min_game_average_elo: int = 0,
    game_positions_per_game: int = 3,
    game_answer_mode: str = "stockfish",
    stockfish_path: str = "stockfish",
    stockfish_depth: int = 20,
    stockfish_multipv: int = 8,
    stockfish_threads: int = 1,
    stockfish_hash_mb: int = 128,
    stockfish_wdl_model: str = "sf",
    stockfish_syzygy_path: str | None = None,
    stockfish_syzygy_max_pieces: int = 5,
    stockfish_persistent_cache_dir: str | None = None,
    stockfish_num_workers: int = 4,
    stockfish_verification_multipv: int = 1,
    use_stockfish_game_reward: bool = True,
    game_reward_legal_floor: float = 0.2,
    game_reward_best_move_bonus: float = 0.05,
    game_reward_expected_score_temperature: float = 0.08,
    game_reward_cp_loss_scale: float = 120.0,
    game_reward_syzygy_wdl_scale: float = 1.0,
    game_reward_syzygy_dtz_scale: float = 20.0,
    game_reward_pv_overlap_bonus: float = 0.05,
    game_reward_pv_motif_plies: int = 6,
    game_reward_use_confidence_weighting: bool = True,
    game_reward_confidence_neutral: float = 0.5,
    game_reward_confidence_nodes_reference: int = 500000,
    game_reward_confidence_seldepth_factor: float = 1.5,
    game_reward_illegal_move_cp_loss: float = 1000.0,
    **kwargs: Any,
) -> vf.Environment:
    if chess is None:
        raise RuntimeError("python-chess is required for hf-chess-mix")

    logger.info(
        "hf-chess-mix: load_environment start max_examples=%d puzzles_fraction=%.3f game_answer_mode=%s stockfish_depth=%d stockfish_num_workers=%d",
        max_examples,
        puzzles_fraction,
        game_answer_mode,
        stockfish_depth,
        stockfish_num_workers,
    )

    if max_examples <= 0:
        raise ValueError(f"max_examples must be > 0, got {max_examples}")
    if not 0.0 <= puzzles_fraction <= 1.0:
        raise ValueError(f"puzzles_fraction must be in [0, 1], got {puzzles_fraction}")
    if max_scan_rows_per_source <= 0:
        raise ValueError(
            f"max_scan_rows_per_source must be > 0, got {max_scan_rows_per_source}"
        )
    if oversample_factor <= 0:
        raise ValueError(f"oversample_factor must be > 0, got {oversample_factor}")
    if shuffle_buffer_size <= 0:
        raise ValueError(f"shuffle_buffer_size must be > 0, got {shuffle_buffer_size}")
    if max_puzzle_solver_moves_per_puzzle == 0 or max_puzzle_solver_moves_per_puzzle < -1:
        raise ValueError(
            "max_puzzle_solver_moves_per_puzzle must be -1 or >= 1, "
            f"got {max_puzzle_solver_moves_per_puzzle}"
        )
    if max_game_ply >= 0 and max_game_ply < min_game_ply:
        raise ValueError(
            "max_game_ply must be >= min_game_ply when non-negative "
            f"({max_game_ply} < {min_game_ply})"
        )
    if game_positions_per_game == 0 or game_positions_per_game < -1:
        raise ValueError(
            "game_positions_per_game must be -1 (all) or >= 1, "
            f"got {game_positions_per_game}"
        )
    if game_answer_mode not in {"stockfish", "pgn"}:
        raise ValueError(
            f"game_answer_mode must be 'stockfish' or 'pgn', got {game_answer_mode}"
        )
    if stockfish_num_workers <= 0:
        raise ValueError(f"stockfish_num_workers must be >= 1, got {stockfish_num_workers}")
    if stockfish_verification_multipv <= 0:
        raise ValueError(
            "stockfish_verification_multipv must be >= 1, "
            f"got {stockfish_verification_multipv}"
        )
    if not 0.0 <= game_reward_legal_floor <= 1.0:
        raise ValueError(
            f"game_reward_legal_floor must be in [0, 1], got {game_reward_legal_floor}"
        )
    if game_reward_best_move_bonus < 0.0:
        raise ValueError(
            f"game_reward_best_move_bonus must be >= 0, got {game_reward_best_move_bonus}"
        )
    if game_reward_expected_score_temperature <= 0.0:
        raise ValueError(
            "game_reward_expected_score_temperature must be > 0, "
            f"got {game_reward_expected_score_temperature}"
        )
    if game_reward_cp_loss_scale <= 0.0:
        raise ValueError(
            f"game_reward_cp_loss_scale must be > 0, got {game_reward_cp_loss_scale}"
        )
    if game_reward_syzygy_wdl_scale <= 0.0:
        raise ValueError(
            f"game_reward_syzygy_wdl_scale must be > 0, got {game_reward_syzygy_wdl_scale}"
        )
    if game_reward_syzygy_dtz_scale <= 0.0:
        raise ValueError(
            f"game_reward_syzygy_dtz_scale must be > 0, got {game_reward_syzygy_dtz_scale}"
        )
    if game_reward_pv_overlap_bonus < 0.0:
        raise ValueError(
            f"game_reward_pv_overlap_bonus must be >= 0, got {game_reward_pv_overlap_bonus}"
        )
    if game_reward_pv_motif_plies < 0:
        raise ValueError(
            f"game_reward_pv_motif_plies must be >= 0, got {game_reward_pv_motif_plies}"
        )
    if not 0.0 <= game_reward_confidence_neutral <= 1.0:
        raise ValueError(
            "game_reward_confidence_neutral must be in [0, 1], "
            f"got {game_reward_confidence_neutral}"
        )
    if game_reward_confidence_nodes_reference <= 0:
        raise ValueError(
            "game_reward_confidence_nodes_reference must be > 0, "
            f"got {game_reward_confidence_nodes_reference}"
        )
    if game_reward_confidence_seldepth_factor <= 0.0:
        raise ValueError(
            "game_reward_confidence_seldepth_factor must be > 0, "
            f"got {game_reward_confidence_seldepth_factor}"
        )
    if game_reward_illegal_move_cp_loss < 0.0:
        raise ValueError(
            "game_reward_illegal_move_cp_loss must be >= 0, "
            f"got {game_reward_illegal_move_cp_loss}"
        )

    stockfish_config = _build_stockfish_config(
        stockfish_path=stockfish_path,
        stockfish_depth=stockfish_depth,
        stockfish_multipv=stockfish_multipv,
        stockfish_threads=stockfish_threads,
        stockfish_hash_mb=stockfish_hash_mb,
        stockfish_wdl_model=stockfish_wdl_model,
        stockfish_syzygy_path=stockfish_syzygy_path,
        stockfish_syzygy_max_pieces=stockfish_syzygy_max_pieces,
        stockfish_persistent_cache_dir=stockfish_persistent_cache_dir,
    )
    logger.info(
        "hf-chess-mix: stockfish config path=%s threads=%d hash_mb=%d depth=%d multipv=%d syzygy_max_pieces=%d",
        stockfish_path,
        stockfish_threads,
        stockfish_hash_mb,
        stockfish_depth,
        stockfish_multipv,
        stockfish_syzygy_max_pieces,
    )

    dataset = _build_dataset(
        max_examples=max_examples,
        puzzles_fraction=puzzles_fraction,
        seed=seed,
        puzzles_dataset=puzzles_dataset,
        games_dataset=games_dataset,
        max_scan_rows_per_source=max_scan_rows_per_source,
        oversample_factor=oversample_factor,
        shuffle_buffer_size=shuffle_buffer_size,
        min_puzzle_rating=min_puzzle_rating,
        puzzle_solver_moves_only=puzzle_solver_moves_only,
        max_puzzle_solver_moves_per_puzzle=max_puzzle_solver_moves_per_puzzle,
        min_game_ply=min_game_ply,
        max_game_ply=max_game_ply,
        min_game_average_elo=min_game_average_elo,
        game_positions_per_game=game_positions_per_game,
        game_answer_mode=game_answer_mode,
        stockfish_config=stockfish_config,
        shuffle=shuffle,
    )

    game_stockfish_pool: _AsyncStockfishVerifierPool | None = None
    if use_stockfish_game_reward:
        game_stockfish_pool = _AsyncStockfishVerifierPool(
            stockfish_config,
            num_workers=stockfish_num_workers,
            verification_multipv=stockfish_verification_multipv,
        )
        logger.info(
            "hf-chess-mix: enabled async Stockfish game verifier pool workers=%d verification_multipv=%d",
            stockfish_num_workers,
            stockfish_verification_multipv,
        )

    rubric = ChessMoveRubric(
        game_stockfish_pool=game_stockfish_pool,
        game_stockfish_depth=stockfish_depth,
        game_reward_illegal_move_cp_loss=game_reward_illegal_move_cp_loss,
        game_reward_legal_floor=game_reward_legal_floor,
        game_reward_best_move_bonus=game_reward_best_move_bonus,
        game_reward_expected_score_temperature=game_reward_expected_score_temperature,
        game_reward_cp_loss_scale=game_reward_cp_loss_scale,
        game_reward_syzygy_wdl_scale=game_reward_syzygy_wdl_scale,
        game_reward_syzygy_dtz_scale=game_reward_syzygy_dtz_scale,
        game_reward_pv_overlap_bonus=game_reward_pv_overlap_bonus,
        game_reward_pv_motif_plies=game_reward_pv_motif_plies,
        game_reward_use_confidence_weighting=game_reward_use_confidence_weighting,
        game_reward_confidence_neutral=game_reward_confidence_neutral,
        game_reward_confidence_nodes_reference=game_reward_confidence_nodes_reference,
        game_reward_confidence_seldepth_factor=game_reward_confidence_seldepth_factor,
    )
    logger.info("hf-chess-mix: load_environment complete")
    return HFChessMixEnv(
        dataset=dataset,
        eval_dataset=dataset,
        rubric=rubric,
        game_stockfish_pool=game_stockfish_pool,
        env_id="hf-chess-mix",
        env_args={
            "max_examples": max_examples,
            "seed": seed,
            "shuffle": shuffle,
            "puzzles_fraction": puzzles_fraction,
            "puzzles_dataset": puzzles_dataset,
            "games_dataset": games_dataset,
            "max_scan_rows_per_source": max_scan_rows_per_source,
            "oversample_factor": oversample_factor,
            "shuffle_buffer_size": shuffle_buffer_size,
            "min_puzzle_rating": min_puzzle_rating,
            "puzzle_solver_moves_only": puzzle_solver_moves_only,
            "max_puzzle_solver_moves_per_puzzle": max_puzzle_solver_moves_per_puzzle,
            "min_game_ply": min_game_ply,
            "max_game_ply": max_game_ply,
            "min_game_average_elo": min_game_average_elo,
            "game_positions_per_game": game_positions_per_game,
            "game_answer_mode": game_answer_mode,
            "stockfish_path": stockfish_path,
            "stockfish_depth": stockfish_depth,
            "stockfish_multipv": stockfish_multipv,
            "stockfish_threads": stockfish_threads,
            "stockfish_hash_mb": stockfish_hash_mb,
            "stockfish_wdl_model": stockfish_wdl_model,
            "stockfish_syzygy_path": stockfish_syzygy_path,
            "stockfish_syzygy_max_pieces": stockfish_syzygy_max_pieces,
            "stockfish_persistent_cache_dir": stockfish_persistent_cache_dir,
            "stockfish_num_workers": stockfish_num_workers,
            "stockfish_verification_multipv": stockfish_verification_multipv,
            "use_stockfish_game_reward": use_stockfish_game_reward,
            "game_reward_legal_floor": game_reward_legal_floor,
            "game_reward_best_move_bonus": game_reward_best_move_bonus,
            "game_reward_expected_score_temperature": game_reward_expected_score_temperature,
            "game_reward_cp_loss_scale": game_reward_cp_loss_scale,
            "game_reward_syzygy_wdl_scale": game_reward_syzygy_wdl_scale,
            "game_reward_syzygy_dtz_scale": game_reward_syzygy_dtz_scale,
            "game_reward_pv_overlap_bonus": game_reward_pv_overlap_bonus,
            "game_reward_pv_motif_plies": game_reward_pv_motif_plies,
            "game_reward_use_confidence_weighting": game_reward_use_confidence_weighting,
            "game_reward_confidence_neutral": game_reward_confidence_neutral,
            "game_reward_confidence_nodes_reference": game_reward_confidence_nodes_reference,
            "game_reward_confidence_seldepth_factor": game_reward_confidence_seldepth_factor,
            "game_reward_illegal_move_cp_loss": game_reward_illegal_move_cp_loss,
        },
        **kwargs,
    )


__all__ = ["load_environment"]
