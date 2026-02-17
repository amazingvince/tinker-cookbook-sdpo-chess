from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import pickle
import queue
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

try:
    import chess
    import chess.engine
    import chess.pgn
except ImportError:  # pragma: no cover - guarded at runtime
    chess = None

if chess is not None:  # pragma: no branch
    try:
        import chess.syzygy
    except ImportError:  # pragma: no cover - optional dependency
        pass

logger = logging.getLogger(__name__)
_INIT_LOG_LOCK = threading.Lock()
_LOGGED_PERSISTENT_CACHE_PATHS: set[str] = set()
_LOGGED_SYZYGY_PATH_SETS: set[tuple[str, ...]] = set()

_FEN_RE = re.compile(
    r"((?:[pnbrqkPNBRQK1-8]{1,8}/){7}[pnbrqkPNBRQK1-8]{1,8}\s[wb]\s(?:-|[KQkq]{1,4})\s(?:-|[a-h][36])\s\d+\s\d+)"
)
_UCI_MOVE_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", flags=re.IGNORECASE)
_PIECE_TYPE_VALUES = {
    1: 1,  # pawn
    2: 3,  # knight
    3: 3,  # bishop
    4: 5,  # rook
    5: 9,  # queen
    6: 0,  # king
}
_WDL_CP_LOSS_SCALE = 1000.0


@dataclass(frozen=True)
class WdlStats:
    win_probability: float
    draw_probability: float
    loss_probability: float
    expected_score: float


@dataclass(frozen=True)
class ThreatSummary:
    side_to_move_hanging: tuple[str, ...]
    opponent_hanging: tuple[str, ...]
    side_to_move_threatened_count: int
    opponent_threatened_count: int
    legal_checking_moves: int


@dataclass(frozen=True)
class MoveHint:
    uci: str
    san: str
    expected_score: float
    delta_expected_score: float
    pv_san: tuple[str, ...]
    refutation_san: str | None
    is_capture: bool
    gives_check: bool
    is_promotion: bool
    hangs_moved_piece: bool
    centipawn_score: float | None = None
    search_depth: int | None = None
    selective_depth: int | None = None
    nodes: int | None = None
    nps: int | None = None
    tablebase_hits: int | None = None


@dataclass(frozen=True)
class PositionHintPack:
    fen: str
    side_to_move: str
    root_wdl: WdlStats
    threat_summary: ThreatSummary
    candidate_moves: tuple[MoveHint, ...]
    root_centipawn_score: float | None = None
    root_search_depth: int | None = None
    root_selective_depth: int | None = None
    root_nodes: int | None = None
    root_nps: int | None = None
    root_tablebase_hits: int | None = None
    syzygy_root_wdl: int | None = None
    syzygy_root_dtz: int | None = None


@dataclass(frozen=True)
class StockfishHintConfig:
    stockfish_path: str = "stockfish"
    depth: int = 14
    multipv: int = 5
    threads: int = 1
    hash_mb: int = 128
    wdl_model: str = "sf"
    max_pv_plies: int = 6
    max_good_moves: int = 3
    max_bad_moves: int = 3
    bad_move_threshold: float = 0.05
    include_fen_decode: bool = True
    include_ascii_board: bool = True
    include_search_stats: bool = True
    max_piece_pressure_items: int = 8
    max_weak_square_items: int = 8
    syzygy_path: str | None = None
    syzygy_max_pieces: int = 7
    unknown_score_cp_loss: float = 80.0
    analysis_time_limit_sec: float | None = None
    engine_max_retries: int = 1
    max_root_cache_entries: int = 8192
    max_move_cache_entries: int = 32768
    max_verification_cache_entries: int = 65536
    persistent_cache_dir: str | None = None


@dataclass(frozen=True)
class MoveVerification:
    fen: str
    depth: int
    predicted_move_uci: str | None
    predicted_move_san: str | None
    move_is_legal: bool
    best_move_uci: str | None
    best_move_san: str | None
    predicted_centipawn: float | None
    best_centipawn: float | None
    cp_loss: float
    cp_loss_source: str
    predicted_expected_score: float | None
    best_expected_score: float | None
    predicted_pv_san: tuple[str, ...]
    best_pv_san: tuple[str, ...]
    predicted_search_depth: int | None
    predicted_selective_depth: int | None
    predicted_nodes: int | None
    predicted_nps: int | None
    best_search_depth: int | None
    best_selective_depth: int | None
    best_nodes: int | None
    best_nps: int | None
    syzygy_root_wdl: int | None
    syzygy_root_dtz: int | None
    syzygy_predicted_wdl: int | None
    syzygy_predicted_dtz: int | None
    syzygy_best_wdl: int | None
    syzygy_best_dtz: int | None
    feedback_text: str


def wdl_to_stats(wins: int, draws: int, losses: int) -> WdlStats:
    total = wins + draws + losses
    if total <= 0:
        raise ValueError(f"WDL total must be > 0, got {total}")
    win_probability = wins / total
    draw_probability = draws / total
    loss_probability = losses / total
    expected_score = win_probability + (0.5 * draw_probability)
    return WdlStats(
        win_probability=win_probability,
        draw_probability=draw_probability,
        loss_probability=loss_probability,
        expected_score=expected_score,
    )


def extract_fen_from_text(text: str) -> str | None:
    match = _FEN_RE.search(text)
    if not match:
        return None
    return match.group(1)


def _as_stripped_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _is_valid_fen(fen: str) -> bool:
    if chess is None:
        return bool(_FEN_RE.fullmatch(fen))
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False


def extract_fen_from_state(
    state: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, Any]] | None = None,
) -> str | None:
    candidate_keys = ("fen", "FEN", "position_fen", "board_fen", "start_fen")

    for key in candidate_keys:
        value = _as_stripped_string(state.get(key))
        if value and _is_valid_fen(value):
            return value

    for nested_key in ("info", "metadata", "extra_info", "reward_extra_info"):
        nested = state.get(nested_key)
        if not isinstance(nested, Mapping):
            continue
        for key in candidate_keys:
            value = _as_stripped_string(nested.get(key))
            if value and _is_valid_fen(value):
                return value

    if prompt_messages:
        for message in reversed(prompt_messages):
            content = message.get("content")
            if isinstance(content, list):
                message_text = " ".join(str(item) for item in content)
            else:
                message_text = str(content)
            maybe_fen = extract_fen_from_text(message_text)
            if maybe_fen and _is_valid_fen(maybe_fen):
                return maybe_fen

    return None


def _piece_symbol(piece: Any) -> str:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")
    symbol = piece.symbol().upper()
    return "P" if piece.piece_type == chess.PAWN else symbol


def _piece_label(piece: Any, square: int) -> str:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")
    color_prefix = "w" if piece.color == chess.WHITE else "b"
    return f"{color_prefix}{_piece_symbol(piece)}@{chess.square_name(square)}"


def _effective_attacker_count(board: Any, attacker_color: bool, target_square: int) -> int:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")

    count = 0
    for attacker_square in board.attackers(attacker_color, target_square):
        if board.is_pinned(attacker_color, attacker_square):
            if target_square not in board.pin(attacker_color, attacker_square):
                continue
        count += 1
    return count


def summarize_threats(board: Any) -> ThreatSummary:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")

    side = board.turn
    side_hanging: list[str] = []
    opponent_hanging: list[str] = []
    side_threatened_count = 0
    opponent_threatened_count = 0

    for square, piece in board.piece_map().items():
        num_attackers = _effective_attacker_count(board, not piece.color, square)
        num_defenders = _effective_attacker_count(board, piece.color, square)
        if num_attackers > 0:
            if piece.color == side:
                side_threatened_count += 1
            else:
                opponent_threatened_count += 1
        if num_attackers > 0 and num_defenders == 0:
            label = _piece_label(piece, square)
            if piece.color == side:
                side_hanging.append(label)
            else:
                opponent_hanging.append(label)

    legal_checking_moves = sum(1 for move in board.legal_moves if board.gives_check(move))
    return ThreatSummary(
        side_to_move_hanging=tuple(sorted(side_hanging)),
        opponent_hanging=tuple(sorted(opponent_hanging)),
        side_to_move_threatened_count=side_threatened_count,
        opponent_threatened_count=opponent_threatened_count,
        legal_checking_moves=legal_checking_moves,
    )


def _material_signature(board: Any, color: bool) -> str:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")

    counts = {
        chess.KING: 0,
        chess.QUEEN: 0,
        chess.ROOK: 0,
        chess.BISHOP: 0,
        chess.KNIGHT: 0,
        chess.PAWN: 0,
    }
    for piece in board.piece_map().values():
        if piece.color == color:
            counts[piece.piece_type] += 1

    material_points = sum(
        counts[piece_type] * _PIECE_TYPE_VALUES[piece_type]
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    return (
        f"K{counts[chess.KING]} Q{counts[chess.QUEEN]} R{counts[chess.ROOK]} "
        f"B{counts[chess.BISHOP]} N{counts[chess.KNIGHT]} P{counts[chess.PAWN]} "
        f"(points={material_points})"
    )


def _board_ascii(board: Any) -> str:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")

    lines: list[str] = []
    for rank in range(7, -1, -1):
        row: list[str] = []
        for file_idx in range(8):
            square = chess.square(file_idx, rank)
            piece = board.piece_at(square)
            row.append(piece.symbol() if piece else ".")
        lines.append(f"{rank + 1} " + " ".join(row))
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


def _piece_pressure_lines(board: Any, color: bool, max_items: int) -> tuple[str, ...]:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")

    scored_lines: list[tuple[tuple[int, int, int], str]] = []
    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue
        num_attackers = _effective_attacker_count(board, not color, square)
        if num_attackers == 0:
            continue
        num_defenders = _effective_attacker_count(board, color, square)
        if num_defenders == 0:
            status = "hanging"
            severity = 2
        elif num_attackers > num_defenders:
            status = "underdefended"
            severity = 1
        else:
            status = "contested"
            severity = 0

        piece_value = _PIECE_TYPE_VALUES[piece.piece_type]
        sort_key = (
            severity,
            num_attackers - num_defenders,
            piece_value,
        )
        scored_lines.append(
            (
                sort_key,
                (
                    f"{_piece_label(piece, square)} attacked_by={num_attackers} "
                    f"defended_by={num_defenders} ({status})"
                ),
            )
        )

    scored_lines.sort(key=lambda item: item[0], reverse=True)
    return tuple(line for _score, line in scored_lines[: max(0, max_items)])


def _weak_king_zone_squares(board: Any, color: bool, max_items: int) -> tuple[str, ...]:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")

    king_square = board.king(color)
    if king_square is None:
        return ()

    ring = chess.SquareSet(chess.BB_KING_ATTACKS[king_square] | chess.BB_SQUARES[king_square])
    scored_squares: list[tuple[tuple[int, int], str]] = []
    for square in ring:
        num_attackers = _effective_attacker_count(board, not color, square)
        if num_attackers == 0:
            continue
        num_defenders = _effective_attacker_count(board, color, square)
        if num_defenders > 0:
            continue
        sort_key = (
            num_attackers,
            _PIECE_TYPE_VALUES.get(board.piece_type_at(square) or chess.PAWN, 0),
        )
        scored_squares.append(
            (
                sort_key,
                f"{chess.square_name(square)}(atk={num_attackers})",
            )
        )
    scored_squares.sort(key=lambda item: item[0], reverse=True)
    return tuple(name for _score, name in scored_squares[: max(0, max_items)])


def _fen_decode_lines(pack: PositionHintPack, config: StockfishHintConfig) -> list[str]:
    if chess is None:
        return []

    board = chess.Board(pack.fen)
    side_to_move = board.turn
    opponent = not side_to_move
    side_name = "white" if side_to_move == chess.WHITE else "black"
    opponent_name = "white" if opponent == chess.WHITE else "black"

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    white_king_square = chess.square_name(white_king) if white_king is not None else "missing"
    black_king_square = chess.square_name(black_king) if black_king is not None else "missing"

    side_pressure = _piece_pressure_lines(board, side_to_move, config.max_piece_pressure_items)
    opponent_pressure = _piece_pressure_lines(board, opponent, config.max_piece_pressure_items)
    side_weak_king_zone = _weak_king_zone_squares(
        board, side_to_move, config.max_weak_square_items
    )
    opponent_weak_king_zone = _weak_king_zone_squares(
        board, opponent, config.max_weak_square_items
    )

    lines: list[str] = ["Position decode from FEN:"]
    if config.include_ascii_board:
        lines.append("Board (white uppercase, black lowercase):")
        lines.extend(_board_ascii(board).splitlines())

    lines.append(
        "Material: "
        f"white[{_material_signature(board, chess.WHITE)}], "
        f"black[{_material_signature(board, chess.BLACK)}]"
    )
    lines.append(f"Kings: white@{white_king_square}, black@{black_king_square}")

    lines.append(
        f"{side_name} pieces under pressure: "
        + (", ".join(side_pressure) if side_pressure else "none")
    )
    lines.append(
        f"{opponent_name} pieces under pressure: "
        + (", ".join(opponent_pressure) if opponent_pressure else "none")
    )

    lines.append(
        f"Weak king-zone squares for {side_name}: "
        + (", ".join(side_weak_king_zone) if side_weak_king_zone else "none")
    )
    lines.append(
        f"Weak king-zone squares for {opponent_name}: "
        + (", ".join(opponent_weak_king_zone) if opponent_weak_king_zone else "none")
    )
    return lines


def _score_to_wdl_stats(score: Any, board: Any, wdl_model: str) -> WdlStats:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")
    if score is None:
        return WdlStats(0.0, 1.0, 0.0, 0.5)

    if isinstance(score, chess.engine.PovScore):
        pov_score = score.pov(board.turn)
    else:
        pov_score = score
    try:
        wdl = pov_score.wdl(model=wdl_model, ply=max(1, board.ply()))
    except Exception:
        wdl = pov_score.wdl(model="sf", ply=max(1, board.ply()))

    return wdl_to_stats(int(wdl.wins), int(wdl.draws), int(wdl.losses))


def _score_to_centipawn(score: Any, board: Any, mate_score: int = 10000) -> float | None:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")
    if score is None:
        return None

    if isinstance(score, chess.engine.PovScore):
        pov_score = score.pov(board.turn)
    else:
        pov_score = score

    centipawn = pov_score.score(mate_score=mate_score)
    if centipawn is None:
        return None
    return float(centipawn)


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_large_int(value: int | None) -> str | None:
    if value is None:
        return None
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _format_search_stats(
    depth: int | None,
    selective_depth: int | None,
    nodes: int | None,
    nps: int | None,
    tablebase_hits: int | None,
) -> str | None:
    parts: list[str] = []
    if depth is not None:
        parts.append(f"d={depth}")
    if selective_depth is not None:
        parts.append(f"sd={selective_depth}")
    formatted_nodes = _format_large_int(nodes)
    if formatted_nodes is not None:
        parts.append(f"nodes={formatted_nodes}")
    formatted_nps = _format_large_int(nps)
    if formatted_nps is not None:
        parts.append(f"nps={formatted_nps}")
    formatted_tbhits = _format_large_int(tablebase_hits)
    if formatted_tbhits is not None:
        parts.append(f"tbhits={formatted_tbhits}")
    if not parts:
        return None
    return ", ".join(parts)


def _syzygy_wdl_label(wdl: int) -> str:
    if wdl >= 2:
        return "tablebase win"
    if wdl == 1:
        return "cursed win"
    if wdl == 0:
        return "draw"
    if wdl == -1:
        return "blessed loss"
    return "tablebase loss"


def _compute_cp_loss(
    best_cp: float | None,
    predicted_cp: float | None,
    best_expected_score: float | None,
    predicted_expected_score: float | None,
    best_move_uci: str | None,
    predicted_move_uci: str | None,
    unknown_score_cp_loss: float,
) -> tuple[float, str]:
    if best_cp is not None and predicted_cp is not None:
        return max(0.0, float(best_cp - predicted_cp)), "centipawn"

    if best_expected_score is not None and predicted_expected_score is not None:
        scaled_loss = max(
            0.0,
            float(best_expected_score - predicted_expected_score) * _WDL_CP_LOSS_SCALE,
        )
        return scaled_loss, "wdl_scaled"

    if (
        best_move_uci is not None
        and predicted_move_uci is not None
        and best_move_uci == predicted_move_uci
    ):
        return 0.0, "same_move"

    if best_move_uci is not None and predicted_move_uci is not None:
        return max(0.0, float(unknown_score_cp_loss)), "fallback_penalty"

    return 0.0, "unavailable"


def extract_predicted_move(board: Any, predicted_text: str) -> Any | None:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")

    for match in _UCI_MOVE_RE.finditer(predicted_text.lower()):
        candidate = match.group(1)
        try:
            move = chess.Move.from_uci(candidate)
        except ValueError:
            continue
        if move in board.legal_moves:
            return move

    for raw in predicted_text.split():
        candidate = raw.strip().strip("`\"'.,;:!?()[]{}")
        if not candidate:
            continue
        try:
            move = board.parse_san(candidate)
        except Exception:
            continue
        if move in board.legal_moves:
            return move

    return None


def _move_hangs_piece(board: Any, move: Any) -> bool:
    moved_board = board.copy(stack=False)
    moved_board.push(move)
    moved_square = move.to_square
    num_attackers = _effective_attacker_count(moved_board, moved_board.turn, moved_square)
    num_defenders = _effective_attacker_count(moved_board, not moved_board.turn, moved_square)
    return num_attackers > 0 and num_defenders == 0


def _pv_to_san(board: Any, pv: Sequence[Any], max_pv_plies: int) -> tuple[str, ...]:
    pv_board = board.copy(stack=False)
    san_moves: list[str] = []
    for move in pv[:max_pv_plies]:
        if move not in pv_board.legal_moves:
            break
        san_moves.append(pv_board.san(move))
        pv_board.push(move)
    return tuple(san_moves)


def _extract_move_hint_from_info(
    board: Any,
    info: Mapping[str, Any],
    config: StockfishHintConfig,
) -> MoveHint | None:
    pv = info.get("pv")
    if not isinstance(pv, list) or not pv:
        return None

    move = pv[0]
    if move not in board.legal_moves:
        return None

    move_score = _score_to_wdl_stats(info.get("score"), board, config.wdl_model)
    centipawn_score = _score_to_centipawn(info.get("score"), board)
    pv_san = _pv_to_san(board, pv, config.max_pv_plies)
    search_depth = _to_optional_int(info.get("depth"))
    selective_depth = _to_optional_int(info.get("seldepth"))
    nodes = _to_optional_int(info.get("nodes"))
    nps = _to_optional_int(info.get("nps"))
    tablebase_hits = _to_optional_int(info.get("tbhits"))

    return MoveHint(
        uci=move.uci(),
        san=board.san(move),
        expected_score=move_score.expected_score,
        delta_expected_score=0.0,
        centipawn_score=centipawn_score,
        pv_san=pv_san,
        refutation_san=pv_san[1] if len(pv_san) > 1 else None,
        is_capture=board.is_capture(move),
        gives_check=board.gives_check(move),
        is_promotion=move.promotion is not None,
        hangs_moved_piece=_move_hangs_piece(board, move),
        search_depth=search_depth,
        selective_depth=selective_depth,
        nodes=nodes,
        nps=nps,
        tablebase_hits=tablebase_hits,
    )


def _trap_severity_label(delta_expected_score: float) -> str:
    if delta_expected_score >= 0.30:
        return "critical"
    if delta_expected_score >= 0.18:
        return "major"
    if delta_expected_score >= 0.10:
        return "moderate"
    return "minor"


def _trap_motifs(move: MoveHint) -> tuple[str, ...]:
    motifs: list[str] = []
    refutation_tail = move.pv_san[1:]
    if move.hangs_moved_piece:
        motifs.append("material drop")
    if any(("+" in san or "#" in san) for san in refutation_tail):
        motifs.append("king safety tactic")
    if any("x" in san for san in refutation_tail):
        motifs.append("tactical capture sequence")
    if move.gives_check and move.delta_expected_score >= 0.10:
        motifs.append("unsound checking idea")
    if move.is_promotion and move.delta_expected_score >= 0.10:
        motifs.append("premature promotion race")
    if not motifs and move.refutation_san:
        motifs.append("concrete refutation line")
    if not motifs:
        motifs.append("strategic deterioration")
    return tuple(motifs)


def _render_trap_analysis_lines(
    pack: PositionHintPack,
    config: StockfishHintConfig,
) -> list[str]:
    trap_threshold = max(float(config.bad_move_threshold), 0.05)
    trap_candidates = [
        move for move in pack.candidate_moves if move.delta_expected_score >= trap_threshold
    ]
    if not trap_candidates:
        return []

    lines = ["Trap analysis (future-state refutations):"]
    for move in trap_candidates[: config.max_bad_moves]:
        severity = _trap_severity_label(move.delta_expected_score)
        refutation_tail = move.pv_san[1:5]
        refutation_text = " ".join(refutation_tail) if refutation_tail else "n/a"
        motifs_text = ", ".join(_trap_motifs(move))
        lines.append(
            f"- {move.uci} ({move.san}) [{severity}] "
            f"delta_E={move.delta_expected_score:+.3f}; "
            f"refutation: {refutation_text}; motifs: {motifs_text}"
        )
    return lines


def render_hint_text(pack: PositionHintPack, config: StockfishHintConfig) -> str:
    side = "white" if pack.side_to_move == "w" else "black"
    lines: list[str] = [
        f"FEN: {pack.fen}",
        f"Side to move: {side}",
        (
            "Root expected score (WDL): "
            f"{pack.root_wdl.expected_score:.3f} "
            f"(W={pack.root_wdl.win_probability:.3f}, "
            f"D={pack.root_wdl.draw_probability:.3f}, "
            f"L={pack.root_wdl.loss_probability:.3f})"
        ),
        (
            "Threat summary: "
            f"threatened_own={pack.threat_summary.side_to_move_threatened_count}, "
            f"threatened_opp={pack.threat_summary.opponent_threatened_count}, "
            f"checking_moves={pack.threat_summary.legal_checking_moves}"
        ),
    ]
    if pack.root_centipawn_score is not None:
        lines.append(f"Root centipawn score (side to move): {pack.root_centipawn_score:+.1f}")
    if config.include_search_stats:
        root_search_stats = _format_search_stats(
            depth=pack.root_search_depth,
            selective_depth=pack.root_selective_depth,
            nodes=pack.root_nodes,
            nps=pack.root_nps,
            tablebase_hits=pack.root_tablebase_hits,
        )
        if root_search_stats:
            lines.append("Root search stats: " + root_search_stats)
    if pack.syzygy_root_wdl is not None:
        syzygy_text = _syzygy_wdl_label(pack.syzygy_root_wdl)
        if pack.syzygy_root_dtz is not None:
            syzygy_text += f", DTZ={pack.syzygy_root_dtz}"
        lines.append("Syzygy root: " + syzygy_text)
    if pack.threat_summary.side_to_move_hanging:
        lines.append(
            "Hanging own pieces: " + ", ".join(pack.threat_summary.side_to_move_hanging)
        )
    if pack.threat_summary.opponent_hanging:
        lines.append("Hanging opponent pieces: " + ", ".join(pack.threat_summary.opponent_hanging))
    if config.include_fen_decode:
        lines.extend(_fen_decode_lines(pack, config))

    if pack.candidate_moves:
        lines.append("Top candidate moves by expected score:")
    for move in pack.candidate_moves[: config.max_good_moves]:
        pv_text = " ".join(move.pv_san) if move.pv_san else "(no pv)"
        cp_text = f"{move.centipawn_score:+.1f}" if move.centipawn_score is not None else "n/a"
        search_stats_text = None
        if config.include_search_stats:
            search_stats_text = _format_search_stats(
                depth=move.search_depth,
                selective_depth=move.selective_depth,
                nodes=move.nodes,
                nps=move.nps,
                tablebase_hits=move.tablebase_hits,
            )
        search_suffix = f", search={search_stats_text}" if search_stats_text else ""
        lines.append(
            f"- {move.uci} ({move.san}): E={move.expected_score:.3f}, "
            f"delta_E={move.delta_expected_score:+.3f}, cp={cp_text}, pv={pv_text}{search_suffix}"
        )

    bad_moves = [
        move
        for move in pack.candidate_moves
        if move.delta_expected_score >= config.bad_move_threshold
    ][: config.max_bad_moves]
    if bad_moves:
        lines.append("Moves likely to be bad:")
    for move in bad_moves:
        reasons: list[str] = []
        if move.refutation_san:
            reasons.append(f"allows {move.refutation_san}")
        if move.hangs_moved_piece:
            reasons.append("hangs moved piece")
        if move.is_capture:
            reasons.append("forcing capture line")
        if move.gives_check:
            reasons.append("checking try not best")
        if move.is_promotion:
            reasons.append("promotion line")
        reason_text = "; ".join(reasons) if reasons else "worse than best line"
        lines.append(
            f"- {move.uci} ({move.san}): delta_E={move.delta_expected_score:+.3f}; {reason_text}"
        )

    lines.extend(_render_trap_analysis_lines(pack, config))

    return "\n".join(lines).strip()


class StockfishHintExtractor:
    _PERSISTENT_ROOT_TABLE = "root_pack"
    _PERSISTENT_VERIFICATION_TABLE = "verification"

    def __init__(self, config: StockfishHintConfig):
        if chess is None:
            raise ImportError(
                "python-chess is required for Stockfish hint extraction. "
                "Install `python-chess` to enable enable_stockfish_hints."
            )
        self.config = config
        self._persistent_key_prefix = self._build_persistent_key_prefix(config)
        self._engine = self._create_engine()
        self._root_analysis_cache: OrderedDict[
            tuple[str, int, int], list[Mapping[str, Any]]
        ] = OrderedDict()
        self._position_hint_cache: OrderedDict[
            tuple[str, int, int], PositionHintPack
        ] = OrderedDict()
        self._move_analysis_cache: OrderedDict[
            tuple[str, int, str], Mapping[str, Any]
        ] = OrderedDict()
        self._verification_cache: OrderedDict[
            tuple[str, int, int, str | None, float], MoveVerification
        ] = OrderedDict()
        self._persistent_cache_lock = threading.Lock()
        self._persistent_cache_conn = self._open_persistent_cache(config.persistent_cache_dir)
        self._tablebase: Any | None = None
        self._configure_engine()
        self._tablebase = self._open_tablebase(config.syzygy_path)

    def _create_engine(self) -> Any:
        if chess is None:
            raise RuntimeError("python-chess is required for Stockfish hint extraction")
        return chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)

    def _restart_engine(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass
        self._engine = self._create_engine()
        self._configure_engine()

    def _configure_engine(self) -> None:
        options = {
            "Threads": self.config.threads,
            "Hash": self.config.hash_mb,
            "UCI_ShowWDL": True,
        }
        for name, value in options.items():
            if value is None:
                continue
            try:
                self._engine.configure({name: value})
            except Exception as exc:
                logger.warning("Could not set Stockfish option %s=%s: %s", name, value, exc)

    def close(self) -> None:
        self._engine.quit()
        if self._persistent_cache_conn is not None:
            try:
                self._persistent_cache_conn.close()
            except Exception:
                pass
            self._persistent_cache_conn = None
        if self._tablebase is not None:
            self._tablebase.close()
            self._tablebase = None

    @staticmethod
    def _log_once_persistent_cache_enabled(db_path: str) -> None:
        normalized = os.path.abspath(db_path)
        with _INIT_LOG_LOCK:
            if normalized in _LOGGED_PERSISTENT_CACHE_PATHS:
                return
            _LOGGED_PERSISTENT_CACHE_PATHS.add(normalized)
        logger.info("Enabled persistent Stockfish cache at %s", normalized)

    @staticmethod
    def _log_once_syzygy_loaded(paths: Sequence[str]) -> None:
        normalized_paths = tuple(os.path.abspath(path) for path in paths)
        with _INIT_LOG_LOCK:
            if normalized_paths in _LOGGED_SYZYGY_PATH_SETS:
                return
            _LOGGED_SYZYGY_PATH_SETS.add(normalized_paths)
        logger.info("Loaded Syzygy tablebase from %s", ", ".join(normalized_paths))

    @staticmethod
    def _build_persistent_key_prefix(config: StockfishHintConfig) -> str:
        cache_signature = {
            "schema": 2,
            "stockfish_path": config.stockfish_path,
            "wdl_model": config.wdl_model,
            "max_pv_plies": int(config.max_pv_plies),
            "unknown_score_cp_loss": float(config.unknown_score_cp_loss),
            "syzygy_path": config.syzygy_path or "",
            "syzygy_max_pieces": int(config.syzygy_max_pieces),
            "analysis_time_limit_sec": (
                None if config.analysis_time_limit_sec is None else float(config.analysis_time_limit_sec)
            ),
        }
        serialized = json.dumps(cache_signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return f"v2:{digest}"

    def _root_persistent_key(self, fen: str, depth: int, multipv: int) -> str:
        return f"{self._persistent_key_prefix}|{fen}|d={int(depth)}|m={int(multipv)}"

    def _verification_persistent_key(
        self,
        fen: str,
        depth: int,
        multipv: int,
        predicted_move_uci: str | None,
        illegal_move_cp_loss: float,
    ) -> str:
        return (
            f"{self._persistent_key_prefix}|{fen}|d={int(depth)}|m={int(multipv)}|p={predicted_move_uci or ''}"
            f"|ill={float(illegal_move_cp_loss):.3f}"
        )

    def _open_persistent_cache(self, cache_dir: str | None) -> sqlite3.Connection | None:
        if not cache_dir:
            return None
        try:
            os.makedirs(cache_dir, exist_ok=True)
            db_path = os.path.join(cache_dir, "stockfish_cache.sqlite3")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS root_pack (
                    cache_key TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verification (
                    cache_key TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
            self._log_once_persistent_cache_enabled(db_path)
            return conn
        except Exception as exc:
            logger.warning("Failed to open persistent Stockfish cache at %s: %s", cache_dir, exc)
            return None

    def _persistent_cache_get(self, table_name: str, cache_key: str) -> Any | None:
        if self._persistent_cache_conn is None:
            return None
        if table_name not in {self._PERSISTENT_ROOT_TABLE, self._PERSISTENT_VERIFICATION_TABLE}:
            return None
        try:
            with self._persistent_cache_lock:
                row = self._persistent_cache_conn.execute(
                    f"SELECT payload FROM {table_name} WHERE cache_key=?",
                    (cache_key,),
                ).fetchone()
            if row is None:
                return None
            payload = row[0]
            if not isinstance(payload, bytes):
                return None
            return pickle.loads(payload)
        except Exception:
            return None

    def _persistent_cache_set(self, table_name: str, cache_key: str, value: Any) -> None:
        if self._persistent_cache_conn is None:
            return
        if table_name not in {self._PERSISTENT_ROOT_TABLE, self._PERSISTENT_VERIFICATION_TABLE}:
            return
        try:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            with self._persistent_cache_lock:
                self._persistent_cache_conn.execute(
                    (
                        f"INSERT INTO {table_name} (cache_key, payload, updated_at) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(cache_key) DO UPDATE SET "
                        "payload=excluded.payload, updated_at=excluded.updated_at"
                    ),
                    (cache_key, payload, float(time.time())),
                )
                self._persistent_cache_conn.commit()
        except Exception:
            return

    @staticmethod
    def _split_syzygy_paths(path_value: str) -> list[str]:
        paths = [segment.strip() for segment in path_value.split(os.pathsep)]
        if len(paths) == 1 and "," in path_value:
            paths = [segment.strip() for segment in path_value.split(",")]
        return [path for path in paths if path]

    def _open_tablebase(self, path_value: str | None) -> Any | None:
        if not path_value:
            return None
        if chess is None or not hasattr(chess, "syzygy"):
            return None

        paths = self._split_syzygy_paths(path_value)
        if not paths:
            return None

        try:
            tablebase = chess.syzygy.open_tablebase(paths[0])
            for extra_path in paths[1:]:
                tablebase.add_directory(extra_path)
            self._log_once_syzygy_loaded(paths)
            return tablebase
        except Exception as exc:
            logger.warning("Failed to load Syzygy tablebase from %s: %s", ", ".join(paths), exc)
            return None

    def _probe_syzygy(self, board: Any) -> tuple[int | None, int | None]:
        if self._tablebase is None:
            return None, None

        if len(board.piece_map()) > max(1, self.config.syzygy_max_pieces):
            return None, None

        wdl: int | None = None
        dtz: int | None = None
        try:
            wdl = int(self._tablebase.probe_wdl(board))
        except Exception:
            wdl = None
        try:
            dtz = int(self._tablebase.probe_dtz(board))
        except Exception:
            dtz = None
        return wdl, dtz

    @staticmethod
    def _normalize_infos(raw_infos: Any) -> list[Mapping[str, Any]]:
        if isinstance(raw_infos, Mapping):
            return [raw_infos]
        if isinstance(raw_infos, list):
            return [info for info in raw_infos if isinstance(info, Mapping)]
        return []

    @staticmethod
    def _cache_get(cache: OrderedDict[Any, Any], key: Any) -> Any | None:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    @staticmethod
    def _cache_set(
        cache: OrderedDict[Any, Any],
        key: Any,
        value: Any,
        max_entries: int,
    ) -> None:
        limit = max(1, int(max_entries))
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _build_analysis_limit(self, depth: int) -> Any:
        if chess is None:
            raise RuntimeError("python-chess is required for Stockfish hint extraction")
        time_limit = self.config.analysis_time_limit_sec
        if time_limit is not None and time_limit > 0:
            return chess.engine.Limit(depth=depth, time=float(time_limit))
        return chess.engine.Limit(depth=depth)

    def _safe_analyse(
        self,
        board: Any,
        depth: int,
        multipv: int,
        root_moves: list[Any] | None = None,
    ) -> Any:
        retries = max(0, int(self.config.engine_max_retries))
        total_attempts = retries + 1
        last_exc: Exception | None = None
        for attempt in range(total_attempts):
            try:
                kwargs: dict[str, Any] = {
                    "limit": self._build_analysis_limit(depth),
                    "multipv": max(1, multipv),
                }
                if root_moves is not None:
                    kwargs["root_moves"] = root_moves
                return self._engine.analyse(board, **kwargs)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Stockfish analyse failed (attempt %d/%d): %s",
                    attempt + 1,
                    total_attempts,
                    exc,
                )
                if attempt < total_attempts - 1:
                    self._restart_engine()
        raise RuntimeError("Stockfish analyse failed after retries") from last_exc

    def _analyze_root_infos(self, board: Any, depth: int, multipv: int) -> list[Mapping[str, Any]]:
        cache_key = (board.fen(), depth, multipv)
        cached = self._cache_get(self._root_analysis_cache, cache_key)
        if cached is not None:
            return cached

        raw_infos = self._safe_analyse(
            board=board,
            depth=depth,
            multipv=multipv,
        )
        infos = self._normalize_infos(raw_infos)
        self._cache_set(
            self._root_analysis_cache,
            cache_key,
            infos,
            max_entries=self.config.max_root_cache_entries,
        )
        return infos

    def _analyze_move_info(self, board: Any, move: Any, depth: int) -> Mapping[str, Any] | None:
        cache_key = (board.fen(), depth, move.uci())
        cached = self._cache_get(self._move_analysis_cache, cache_key)
        if cached is not None:
            return cached

        raw_info = self._safe_analyse(
            board=board,
            depth=depth,
            multipv=1,
            root_moves=[move],
        )
        infos = self._normalize_infos(raw_info)
        if not infos:
            return None
        info = infos[0]
        self._cache_set(
            self._move_analysis_cache,
            cache_key,
            info,
            max_entries=self.config.max_move_cache_entries,
        )
        return info

    def analyze_fen(
        self,
        fen: str,
        depth: int | None = None,
        multipv: int | None = None,
    ) -> PositionHintPack:
        if chess is None:
            raise RuntimeError("python-chess is required for Stockfish hint extraction")
        board = chess.Board(fen)
        analysis_depth = int(depth) if depth is not None else int(self.config.depth)
        analysis_multipv = max(1, int(multipv)) if multipv is not None else max(1, int(self.config.multipv))
        pack_cache_key = (fen, analysis_depth, analysis_multipv)
        cached_pack = self._cache_get(self._position_hint_cache, pack_cache_key)
        if cached_pack is not None:
            return cached_pack
        persistent_pack = self._persistent_cache_get(
            self._PERSISTENT_ROOT_TABLE,
            self._root_persistent_key(fen, analysis_depth, analysis_multipv),
        )
        if isinstance(persistent_pack, PositionHintPack):
            self._cache_set(
                self._position_hint_cache,
                pack_cache_key,
                persistent_pack,
                max_entries=self.config.max_root_cache_entries,
            )
            return persistent_pack

        infos = self._analyze_root_infos(board, depth=analysis_depth, multipv=analysis_multipv)

        move_hints: list[MoveHint] = []
        for info in infos:
            hint = _extract_move_hint_from_info(board, info, self.config)
            if hint is not None:
                move_hints.append(hint)

        if move_hints:
            best_expected_score = max(move_hint.expected_score for move_hint in move_hints)
            move_hints = [
                replace(
                    move_hint,
                    delta_expected_score=best_expected_score - move_hint.expected_score,
                )
                for move_hint in move_hints
            ]
            move_hints.sort(key=lambda move_hint: move_hint.expected_score, reverse=True)

        root_info = infos[0] if infos else {}
        root_score = _score_to_wdl_stats(root_info.get("score"), board, self.config.wdl_model)
        if move_hints:
            root_score = replace(root_score, expected_score=move_hints[0].expected_score)

        root_centipawn = _score_to_centipawn(root_info.get("score"), board)
        root_depth = _to_optional_int(root_info.get("depth"))
        root_selective_depth = _to_optional_int(root_info.get("seldepth"))
        root_nodes = _to_optional_int(root_info.get("nodes"))
        root_nps = _to_optional_int(root_info.get("nps"))
        root_tablebase_hits = _to_optional_int(root_info.get("tbhits"))

        if move_hints:
            best_hint = move_hints[0]
            if best_hint.centipawn_score is not None:
                root_centipawn = best_hint.centipawn_score
            if root_depth is None:
                root_depth = best_hint.search_depth
            if root_selective_depth is None:
                root_selective_depth = best_hint.selective_depth
            if root_nodes is None:
                root_nodes = best_hint.nodes
            if root_nps is None:
                root_nps = best_hint.nps
            if root_tablebase_hits is None:
                root_tablebase_hits = best_hint.tablebase_hits

        syzygy_root_wdl, syzygy_root_dtz = self._probe_syzygy(board)

        pack = PositionHintPack(
            fen=fen,
            side_to_move="w" if board.turn == chess.WHITE else "b",
            root_wdl=root_score,
            threat_summary=summarize_threats(board),
            candidate_moves=tuple(move_hints),
            root_centipawn_score=root_centipawn,
            root_search_depth=root_depth,
            root_selective_depth=root_selective_depth,
            root_nodes=root_nodes,
            root_nps=root_nps,
            root_tablebase_hits=root_tablebase_hits,
            syzygy_root_wdl=syzygy_root_wdl,
            syzygy_root_dtz=syzygy_root_dtz,
        )
        self._cache_set(
            self._position_hint_cache,
            pack_cache_key,
            pack,
            max_entries=self.config.max_root_cache_entries,
        )
        self._persistent_cache_set(
            self._PERSISTENT_ROOT_TABLE,
            self._root_persistent_key(fen, analysis_depth, analysis_multipv),
            pack,
        )
        return pack

    def analyze_and_render(
        self,
        fen: str,
        depth: int | None = None,
        multipv: int | None = None,
    ) -> str:
        return render_hint_text(
            self.analyze_fen(
                fen=fen,
                depth=depth,
                multipv=multipv,
            ),
            self.config,
        )

    @staticmethod
    def _render_move_verification_feedback(verification: MoveVerification) -> str:
        cp_loss_text = f"{verification.cp_loss:.1f}"
        cp_loss_source_text = verification.cp_loss_source
        best_cp_text = (
            f"{verification.best_centipawn:+.1f}"
            if verification.best_centipawn is not None
            else "n/a"
        )
        predicted_cp_text = (
            f"{verification.predicted_centipawn:+.1f}"
            if verification.predicted_centipawn is not None
            else "n/a"
        )
        best_exp_text = (
            f"{verification.best_expected_score:.3f}"
            if verification.best_expected_score is not None
            else "n/a"
        )
        predicted_exp_text = (
            f"{verification.predicted_expected_score:.3f}"
            if verification.predicted_expected_score is not None
            else "n/a"
        )

        lines = ["Stockfish move verification:"]
        if verification.move_is_legal:
            predicted_desc = verification.predicted_move_uci or "n/a"
            if verification.predicted_move_san:
                predicted_desc += f" ({verification.predicted_move_san})"
            lines.append(f"- Predicted move: {predicted_desc}")
        else:
            lines.append("- Predicted move could not be parsed as a legal move from this FEN.")

        if verification.best_move_uci:
            best_desc = verification.best_move_uci
            if verification.best_move_san:
                best_desc += f" ({verification.best_move_san})"
            lines.append(f"- Stockfish best move at depth {verification.depth}: {best_desc}")

        lines.append(
            "- Centipawn scores (side to move): "
            f"best={best_cp_text}, predicted={predicted_cp_text}, "
            f"cp_loss={cp_loss_text} (source={cp_loss_source_text})"
        )
        lines.append(
            "- Expected score (WDL): "
            f"best={best_exp_text}, predicted={predicted_exp_text}"
        )
        if verification.syzygy_root_wdl is not None:
            root_syzygy_text = _syzygy_wdl_label(verification.syzygy_root_wdl)
            if verification.syzygy_root_dtz is not None:
                root_syzygy_text += f", DTZ={verification.syzygy_root_dtz}"
            lines.append("- Syzygy root: " + root_syzygy_text)
        if verification.syzygy_best_wdl is not None:
            best_syzygy_text = _syzygy_wdl_label(verification.syzygy_best_wdl)
            if verification.syzygy_best_dtz is not None:
                best_syzygy_text += f", DTZ={verification.syzygy_best_dtz}"
            lines.append("- Syzygy best move outcome: " + best_syzygy_text)
        if verification.syzygy_predicted_wdl is not None:
            predicted_syzygy_text = _syzygy_wdl_label(verification.syzygy_predicted_wdl)
            if verification.syzygy_predicted_dtz is not None:
                predicted_syzygy_text += f", DTZ={verification.syzygy_predicted_dtz}"
            lines.append("- Syzygy predicted move outcome: " + predicted_syzygy_text)

        if verification.best_pv_san:
            lines.append("- Best PV: " + " ".join(verification.best_pv_san))
        if verification.predicted_pv_san:
            lines.append("- Predicted-move PV: " + " ".join(verification.predicted_pv_san))

        return "\n".join(lines).strip()

    def verify_predicted_move(
        self,
        fen: str,
        predicted_text: str,
        depth: int = 20,
        illegal_move_cp_loss: float = 1000.0,
        verification_multipv: int | None = None,
    ) -> MoveVerification:
        if chess is None:
            raise RuntimeError("python-chess is required for Stockfish hint extraction")

        board = chess.Board(fen)
        predicted_move = extract_predicted_move(board, predicted_text)
        predicted_move_key = predicted_move.uci() if predicted_move is not None else None
        analysis_multipv = (
            max(1, int(verification_multipv))
            if verification_multipv is not None
            else max(self.config.multipv, 8)
        )
        cache_key = (
            fen,
            depth,
            analysis_multipv,
            predicted_move_key,
            float(illegal_move_cp_loss),
        )
        cached = self._cache_get(self._verification_cache, cache_key)
        if cached is not None:
            return cached
        persistent_key = self._verification_persistent_key(
            fen=fen,
            depth=depth,
            multipv=analysis_multipv,
            predicted_move_uci=predicted_move_key,
            illegal_move_cp_loss=float(illegal_move_cp_loss),
        )
        persistent_cached = self._persistent_cache_get(
            self._PERSISTENT_VERIFICATION_TABLE,
            persistent_key,
        )
        if isinstance(persistent_cached, MoveVerification):
            self._cache_set(
                self._verification_cache,
                cache_key,
                persistent_cached,
                max_entries=self.config.max_verification_cache_entries,
            )
            return persistent_cached

        pack = self.analyze_fen(
            fen=fen,
            depth=depth,
            multipv=analysis_multipv,
        )
        best_move = pack.candidate_moves[0] if pack.candidate_moves else None

        best_move_uci = best_move.uci if best_move else None
        best_move_san = best_move.san if best_move else None
        best_cp = best_move.centipawn_score if best_move else None
        best_expected = best_move.expected_score if best_move else None
        best_pv = best_move.pv_san if best_move else ()
        best_depth = best_move.search_depth if best_move else None
        best_seldepth = best_move.selective_depth if best_move else None
        best_nodes = best_move.nodes if best_move else None
        best_nps = best_move.nps if best_move else None
        syzygy_root_wdl, syzygy_root_dtz = self._probe_syzygy(board)

        syzygy_best_wdl: int | None = None
        syzygy_best_dtz: int | None = None
        if best_move is not None:
            try:
                best_move_obj = chess.Move.from_uci(best_move.uci)
                if best_move_obj in board.legal_moves:
                    best_board = board.copy(stack=False)
                    best_board.push(best_move_obj)
                    syzygy_best_wdl, syzygy_best_dtz = self._probe_syzygy(best_board)
            except Exception:
                syzygy_best_wdl, syzygy_best_dtz = None, None

        if predicted_move is None:
            verification = MoveVerification(
                fen=fen,
                depth=depth,
                predicted_move_uci=None,
                predicted_move_san=None,
                move_is_legal=False,
                best_move_uci=best_move_uci,
                best_move_san=best_move_san,
                predicted_centipawn=None,
                best_centipawn=best_cp,
                cp_loss=float(illegal_move_cp_loss),
                cp_loss_source="illegal_or_unparsed",
                predicted_expected_score=None,
                best_expected_score=best_expected,
                predicted_pv_san=(),
                best_pv_san=best_pv,
                predicted_search_depth=None,
                predicted_selective_depth=None,
                predicted_nodes=None,
                predicted_nps=None,
                best_search_depth=best_depth,
                best_selective_depth=best_seldepth,
                best_nodes=best_nodes,
                best_nps=best_nps,
                syzygy_root_wdl=syzygy_root_wdl,
                syzygy_root_dtz=syzygy_root_dtz,
                syzygy_predicted_wdl=None,
                syzygy_predicted_dtz=None,
                syzygy_best_wdl=syzygy_best_wdl,
                syzygy_best_dtz=syzygy_best_dtz,
                feedback_text="",
            )
            result = replace(
                verification,
                feedback_text=self._render_move_verification_feedback(verification),
            )
            self._cache_set(
                self._verification_cache,
                cache_key,
                result,
                max_entries=self.config.max_verification_cache_entries,
            )
            self._persistent_cache_set(
                self._PERSISTENT_VERIFICATION_TABLE,
                persistent_key,
                result,
            )
            return result

        predicted_san = board.san(predicted_move)
        predicted_hint = next(
            (move_hint for move_hint in pack.candidate_moves if move_hint.uci == predicted_move.uci()),
            None,
        )

        if predicted_hint is None:
            predicted_info = self._analyze_move_info(board, predicted_move, depth=depth)
            predicted_expected = (
                _score_to_wdl_stats(predicted_info.get("score"), board, self.config.wdl_model).expected_score
                if predicted_info is not None
                else None
            )
            predicted_cp = (
                _score_to_centipawn(predicted_info.get("score"), board)
                if predicted_info is not None
                else None
            )
            predicted_depth = _to_optional_int(predicted_info.get("depth")) if predicted_info else None
            predicted_seldepth = (
                _to_optional_int(predicted_info.get("seldepth")) if predicted_info else None
            )
            predicted_nodes = _to_optional_int(predicted_info.get("nodes")) if predicted_info else None
            predicted_nps = _to_optional_int(predicted_info.get("nps")) if predicted_info else None
            if predicted_info is not None and isinstance(predicted_info.get("pv"), list):
                predicted_pv = _pv_to_san(board, predicted_info["pv"], self.config.max_pv_plies)
            else:
                predicted_pv = (predicted_san,)
        else:
            predicted_expected = predicted_hint.expected_score
            predicted_cp = predicted_hint.centipawn_score
            predicted_pv = predicted_hint.pv_san
            predicted_depth = predicted_hint.search_depth
            predicted_seldepth = predicted_hint.selective_depth
            predicted_nodes = predicted_hint.nodes
            predicted_nps = predicted_hint.nps

        cp_loss, cp_loss_source = _compute_cp_loss(
            best_cp=best_cp,
            predicted_cp=predicted_cp,
            best_expected_score=best_expected,
            predicted_expected_score=predicted_expected,
            best_move_uci=best_move_uci,
            predicted_move_uci=predicted_move.uci(),
            unknown_score_cp_loss=self.config.unknown_score_cp_loss,
        )

        predicted_board = board.copy(stack=False)
        predicted_board.push(predicted_move)
        syzygy_predicted_wdl, syzygy_predicted_dtz = self._probe_syzygy(predicted_board)

        verification = MoveVerification(
            fen=fen,
            depth=depth,
            predicted_move_uci=predicted_move.uci(),
            predicted_move_san=predicted_san,
            move_is_legal=True,
            best_move_uci=best_move_uci,
            best_move_san=best_move_san,
            predicted_centipawn=predicted_cp,
            best_centipawn=best_cp,
            cp_loss=cp_loss,
            cp_loss_source=cp_loss_source,
            predicted_expected_score=predicted_expected,
            best_expected_score=best_expected,
            predicted_pv_san=predicted_pv,
            best_pv_san=best_pv,
            predicted_search_depth=predicted_depth,
            predicted_selective_depth=predicted_seldepth,
            predicted_nodes=predicted_nodes,
            predicted_nps=predicted_nps,
            best_search_depth=best_depth,
            best_selective_depth=best_seldepth,
            best_nodes=best_nodes,
            best_nps=best_nps,
            syzygy_root_wdl=syzygy_root_wdl,
            syzygy_root_dtz=syzygy_root_dtz,
            syzygy_predicted_wdl=syzygy_predicted_wdl,
            syzygy_predicted_dtz=syzygy_predicted_dtz,
            syzygy_best_wdl=syzygy_best_wdl,
            syzygy_best_dtz=syzygy_best_dtz,
            feedback_text="",
        )
        result = replace(
            verification,
            feedback_text=self._render_move_verification_feedback(verification),
        )
        self._cache_set(
            self._verification_cache,
            cache_key,
            result,
            max_entries=self.config.max_verification_cache_entries,
        )
        self._persistent_cache_set(
            self._PERSISTENT_VERIFICATION_TABLE,
            persistent_key,
            result,
        )
        return result

    def analyze_and_verify(
        self,
        fen: str,
        predicted_text: str,
        *,
        hint_depth: int | None = None,
        hint_multipv: int | None = None,
        verification_depth: int = 20,
        verification_multipv: int | None = None,
        illegal_move_cp_loss: float = 1000.0,
        mode: str = "two_pass",
    ) -> tuple[str, MoveVerification]:
        mode_normalized = mode.strip().lower()
        if mode_normalized not in {"single", "two_pass"}:
            raise ValueError(
                "mode must be 'single' or 'two_pass', "
                f"got {mode}"
            )

        hint_depth_eff = int(hint_depth) if hint_depth is not None else int(self.config.depth)
        hint_multipv_eff = (
            max(1, int(hint_multipv))
            if hint_multipv is not None
            else max(1, int(self.config.multipv))
        )
        verification_depth_eff = int(verification_depth)
        verification_multipv_eff = (
            max(1, int(verification_multipv))
            if verification_multipv is not None
            else max(1, max(int(self.config.multipv), 8))
        )

        if mode_normalized == "single":
            hint_pack = self.analyze_fen(
                fen=fen,
                depth=verification_depth_eff,
                multipv=verification_multipv_eff,
            )
        else:
            hint_pack = self.analyze_fen(
                fen=fen,
                depth=hint_depth_eff,
                multipv=hint_multipv_eff,
            )
            if (
                verification_depth_eff != hint_depth_eff
                or verification_multipv_eff != hint_multipv_eff
            ):
                # Warm deep cache once so verification can reuse it without another engine call.
                self.analyze_fen(
                    fen=fen,
                    depth=verification_depth_eff,
                    multipv=verification_multipv_eff,
                )

        verification = self.verify_predicted_move(
            fen=fen,
            predicted_text=predicted_text,
            depth=verification_depth_eff,
            illegal_move_cp_loss=illegal_move_cp_loss,
            verification_multipv=verification_multipv_eff,
        )
        return render_hint_text(hint_pack, self.config), verification


class StockfishHintPool:
    """Async dispatcher over multiple persistent Stockfish extractors."""

    def __init__(self, config: StockfishHintConfig, num_workers: int):
        if num_workers <= 0:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        self.config = config
        self.num_workers = int(num_workers)
        self._workers = [StockfishHintExtractor(config) for _ in range(self.num_workers)]
        self._available_workers: queue.Queue[int] = queue.Queue()
        for idx in range(self.num_workers):
            self._available_workers.put(idx)
        self._executor = ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="stockfish_pool",
        )
        self._closed = False

    def _run_with_worker(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("StockfishHintPool is closed")

        worker_idx = self._available_workers.get()
        try:
            worker = self._workers[worker_idx]
            method = getattr(worker, method_name)
            return method(*args, **kwargs)
        finally:
            self._available_workers.put(worker_idx)

    async def _run_async(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("StockfishHintPool is closed")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._run_with_worker, method_name, *args, **kwargs),
        )

    async def analyze_and_render_async(
        self,
        fen: str,
        depth: int | None = None,
        multipv: int | None = None,
    ) -> str:
        return await self._run_async(
            "analyze_and_render",
            fen,
            depth=depth,
            multipv=multipv,
        )

    async def verify_predicted_move_async(
        self,
        fen: str,
        predicted_text: str,
        depth: int = 20,
        illegal_move_cp_loss: float = 1000.0,
        verification_multipv: int | None = None,
    ) -> MoveVerification:
        return await self._run_async(
            "verify_predicted_move",
            fen,
            predicted_text,
            depth=depth,
            illegal_move_cp_loss=illegal_move_cp_loss,
            verification_multipv=verification_multipv,
        )

    async def analyze_and_verify_async(
        self,
        fen: str,
        predicted_text: str,
        *,
        hint_depth: int | None = None,
        hint_multipv: int | None = None,
        verification_depth: int = 20,
        verification_multipv: int | None = None,
        illegal_move_cp_loss: float = 1000.0,
        mode: str = "two_pass",
    ) -> tuple[str, MoveVerification]:
        return await self._run_async(
            "analyze_and_verify",
            fen,
            predicted_text,
            hint_depth=hint_depth,
            hint_multipv=hint_multipv,
            verification_depth=verification_depth,
            verification_multipv=verification_multipv,
            illegal_move_cp_loss=illegal_move_cp_loss,
            mode=mode,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        self._executor.shutdown(wait=True, cancel_futures=False)
        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass


def pick_random_game_fen(movetext: str, seed: int | None = None) -> str | None:
    if chess is None:
        raise RuntimeError("python-chess is required for chess dataset helpers")

    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None:
        return None

    moves = list(game.mainline_moves())
    if not moves:
        return None

    rng = random.Random(seed)
    ply_index = rng.randint(0, len(moves) - 1)
    board = game.board()
    for move in moves[:ply_index]:
        board.push(move)
    return board.fen()


def build_stockfish_hint_text_for_state(
    state: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, Any]],
    extractor: StockfishHintExtractor | None,
) -> str | None:
    if extractor is None:
        return None

    fen = extract_fen_from_state(state, prompt_messages)
    if fen is None:
        return None

    try:
        return extractor.analyze_and_render(fen)
    except Exception as exc:
        logger.warning("Failed to build Stockfish hint text for fen=%s: %s", fen, exc)
        return None


__all__ = [
    "MoveVerification",
    "MoveHint",
    "PositionHintPack",
    "StockfishHintConfig",
    "StockfishHintExtractor",
    "StockfishHintPool",
    "ThreatSummary",
    "WdlStats",
    "build_stockfish_hint_text_for_state",
    "extract_predicted_move",
    "extract_fen_from_state",
    "extract_fen_from_text",
    "render_hint_text",
    "summarize_threats",
    "wdl_to_stats",
]
