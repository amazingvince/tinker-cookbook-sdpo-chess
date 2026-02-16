from __future__ import annotations

import io
import logging
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

try:
    import chess
    import chess.engine
    import chess.pgn
except ImportError:  # pragma: no cover - guarded at runtime
    chess = None

logger = logging.getLogger(__name__)

_FEN_RE = re.compile(
    r"((?:[pnbrqkPNBRQK1-8]{1,8}/){7}[pnbrqkPNBRQK1-8]{1,8}\s[wb]\s(?:-|[KQkq]{1,4})\s(?:-|[a-h][36])\s\d+\s\d+)"
)

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


@dataclass(frozen=True)
class PositionHintPack:
    fen: str
    side_to_move: str
    root_wdl: WdlStats
    threat_summary: ThreatSummary
    candidate_moves: tuple[MoveHint, ...]


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


def summarize_threats(board: Any) -> ThreatSummary:
    if chess is None:
        raise RuntimeError("python-chess is required for Stockfish hint extraction")

    side = board.turn
    side_hanging: list[str] = []
    opponent_hanging: list[str] = []
    side_threatened_count = 0
    opponent_threatened_count = 0

    for square, piece in board.piece_map().items():
        attackers = board.attackers(not piece.color, square)
        defenders = board.attackers(piece.color, square)
        if attackers:
            if piece.color == side:
                side_threatened_count += 1
            else:
                opponent_threatened_count += 1
        if attackers and not defenders:
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


def _move_hangs_piece(board: Any, move: Any) -> bool:
    moved_board = board.copy(stack=False)
    moved_board.push(move)
    moved_square = move.to_square
    attackers = moved_board.attackers(moved_board.turn, moved_square)
    defenders = moved_board.attackers(not moved_board.turn, moved_square)
    return bool(attackers) and not bool(defenders)


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
    pv_san = _pv_to_san(board, pv, config.max_pv_plies)

    return MoveHint(
        uci=move.uci(),
        san=board.san(move),
        expected_score=move_score.expected_score,
        delta_expected_score=0.0,
        pv_san=pv_san,
        refutation_san=pv_san[1] if len(pv_san) > 1 else None,
        is_capture=board.is_capture(move),
        gives_check=board.gives_check(move),
        is_promotion=move.promotion is not None,
        hangs_moved_piece=_move_hangs_piece(board, move),
    )


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
    if pack.threat_summary.side_to_move_hanging:
        lines.append(
            "Hanging own pieces: " + ", ".join(pack.threat_summary.side_to_move_hanging)
        )
    if pack.threat_summary.opponent_hanging:
        lines.append("Hanging opponent pieces: " + ", ".join(pack.threat_summary.opponent_hanging))

    if pack.candidate_moves:
        lines.append("Top candidate moves by expected score:")
    for move in pack.candidate_moves[: config.max_good_moves]:
        pv_text = " ".join(move.pv_san) if move.pv_san else "(no pv)"
        lines.append(
            f"- {move.uci} ({move.san}): E={move.expected_score:.3f}, "
            f"delta_E={move.delta_expected_score:+.3f}, pv={pv_text}"
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

    return "\n".join(lines).strip()


class StockfishHintExtractor:
    def __init__(self, config: StockfishHintConfig):
        if chess is None:
            raise ImportError(
                "python-chess is required for Stockfish hint extraction. "
                "Install `python-chess` to enable enable_stockfish_hints."
            )
        self.config = config
        self._engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
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

    def analyze_fen(self, fen: str) -> PositionHintPack:
        if chess is None:
            raise RuntimeError("python-chess is required for Stockfish hint extraction")
        board = chess.Board(fen)
        limit = chess.engine.Limit(depth=self.config.depth)
        raw_infos = self._engine.analyse(
            board,
            limit=limit,
            multipv=max(1, self.config.multipv),
        )
        if isinstance(raw_infos, Mapping):
            infos = [raw_infos]
        else:
            infos = [info for info in raw_infos if isinstance(info, Mapping)]

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

        return PositionHintPack(
            fen=fen,
            side_to_move="w" if board.turn == chess.WHITE else "b",
            root_wdl=root_score,
            threat_summary=summarize_threats(board),
            candidate_moves=tuple(move_hints),
        )

    def analyze_and_render(self, fen: str) -> str:
        return render_hint_text(self.analyze_fen(fen), self.config)


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
    "MoveHint",
    "PositionHintPack",
    "StockfishHintConfig",
    "StockfishHintExtractor",
    "ThreatSummary",
    "WdlStats",
    "build_stockfish_hint_text_for_state",
    "extract_fen_from_state",
    "extract_fen_from_text",
    "render_hint_text",
    "summarize_threats",
    "wdl_to_stats",
]
