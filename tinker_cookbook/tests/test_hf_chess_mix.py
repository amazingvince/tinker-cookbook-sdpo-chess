from __future__ import annotations

import asyncio
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

chess = pytest.importorskip("chess")
vf = pytest.importorskip("verifiers")

from tinker_cookbook.sdpo.chess_hints import MoveVerification  # noqa: E402

from tinker_cookbook.recipes.verifiers_rl.hf_chess_mix import (  # noqa: E402
    ChessMoveRubric,
    _assemble_mixed_examples,
    _extract_first_uci,
    _game_quality_from_verification,
    _label_selected_game_rows_with_best_moves,
    _parse_game_row,
    _parse_puzzle_row,
    _pv_motif_overlap,
    _search_confidence,
    _syzygy_dtz_penalty,
    _syzygy_wdl_penalty,
)


def test_extract_first_uci_from_text_and_list():
    assert _extract_first_uci("I would play e2e4 here.") == "e2e4"
    assert _extract_first_uci(["bad", "Nf3", "a7a8q"]) == "a7a8q"
    assert _extract_first_uci("no legal move token") is None


def test_parse_puzzle_row_expands_solver_moves_with_per_move_fens():
    fen = chess.STARTING_FEN
    row = {
        "PuzzleId": "00008",
        "GameId": "787zsVup/black#48",
        "FEN": fen,
        "Moves": "e2e4 e7e5 g1f3",
        "Rating": 2037,
        "Themes": ["crushing", "hangingPiece"],
    }

    parsed = _parse_puzzle_row(
        row,
        min_puzzle_rating=0,
        puzzle_solver_moves_only=True,
        max_puzzle_solver_moves_per_puzzle=-1,
    )
    assert len(parsed) == 2
    assert [row["answer"] for row in parsed] == ["e2e4", "g1f3"]
    assert all(row["info"]["source"] == "lichess_puzzle" for row in parsed)
    assert all(row["info"]["puzzle_solver_move"] is True for row in parsed)
    assert "FEN:" in parsed[0]["prompt"][0]["content"]

    second_board = chess.Board(fen)
    second_board.push(chess.Move.from_uci("e2e4"))
    second_board.push(chess.Move.from_uci("e7e5"))
    assert parsed[1]["info"]["fen"] == second_board.fen()


def test_parse_game_row_extracts_fen_answer_and_metadata():
    row = {
        "movetext": "1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6",
        "Site": "https://lichess.org/j1dkb5dw",
        "Opening": "French Defense: Normal Variation",
        "Result": "1-0",
        "WhiteElo": 1639,
        "BlackElo": 1403,
    }
    parsed = _parse_game_row(
        row=row,
        rng=random.Random(7),
        min_game_ply=0,
        max_game_ply=6,
        min_game_average_elo=0,
        game_positions_per_game=2,
        game_answer_mode="pgn",
        best_move_oracle=None,
    )
    assert len(parsed) == 2
    for sample in parsed:
        assert sample["task"] == "chess_next_move"
        assert sample["info"]["source"] == "lichess_game"
        fen = sample["info"]["fen"]
        board = chess.Board(fen)
        move = chess.Move.from_uci(sample["answer"])
        assert move in board.legal_moves
        assert sample["info"]["site"] == "https://lichess.org/j1dkb5dw"
        assert sample["info"]["game_answer_mode"] == "pgn"
        assert sample["answer"] == sample["info"]["pgn_next_move"]


def test_game_quality_from_verification_expected_score_delta():
    verification = SimpleNamespace(
        move_is_legal=True,
        best_expected_score=0.74,
        predicted_expected_score=0.66,
        cp_loss=40.0,
    )
    quality = _game_quality_from_verification(
        verification=verification,
        expected_score_temperature=0.08,
        cp_loss_scale=120.0,
    )
    assert quality == pytest.approx(0.367879, rel=1e-5)


def test_game_quality_from_verification_cp_fallback_and_illegal():
    verification = SimpleNamespace(
        move_is_legal=True,
        best_expected_score=None,
        predicted_expected_score=None,
        cp_loss=120.0,
    )
    quality = _game_quality_from_verification(
        verification=verification,
        expected_score_temperature=0.08,
        cp_loss_scale=120.0,
    )
    assert quality == pytest.approx(0.367879, rel=1e-5)

    illegal = SimpleNamespace(
        move_is_legal=False,
        best_expected_score=0.9,
        predicted_expected_score=0.0,
        cp_loss=1000.0,
    )
    assert (
        _game_quality_from_verification(
            verification=illegal,
            expected_score_temperature=0.08,
            cp_loss_scale=120.0,
        )
        == 0.0
    )


class _FakeVerifierPool:
    def __init__(self, verification: MoveVerification):
        self.verification = verification

    async def verify_predicted_move_async(self, **kwargs):
        return self.verification


def _build_move_verification(
    *,
    move_is_legal: bool,
    predicted_move_uci: str | None,
    best_move_uci: str | None,
    predicted_expected_score: float | None,
    best_expected_score: float | None,
    cp_loss: float,
) -> MoveVerification:
    return MoveVerification(
        fen=chess.STARTING_FEN,
        depth=20,
        predicted_move_uci=predicted_move_uci,
        predicted_move_san=None,
        move_is_legal=move_is_legal,
        best_move_uci=best_move_uci,
        best_move_san=None,
        predicted_centipawn=None,
        best_centipawn=None,
        cp_loss=cp_loss,
        cp_loss_source="wdl_scaled",
        predicted_expected_score=predicted_expected_score,
        best_expected_score=best_expected_score,
        predicted_pv_san=(),
        best_pv_san=(),
        predicted_search_depth=20,
        predicted_selective_depth=30,
        predicted_nodes=500000,
        predicted_nps=3000000,
        best_search_depth=20,
        best_selective_depth=30,
        best_nodes=500000,
        best_nps=3000000,
        syzygy_root_wdl=None,
        syzygy_root_dtz=None,
        syzygy_predicted_wdl=None,
        syzygy_predicted_dtz=None,
        syzygy_best_wdl=None,
        syzygy_best_dtz=None,
        feedback_text="",
    )


def test_chess_move_rubric_game_reward_uses_legal_floor_and_best_bonus():
    verification = _build_move_verification(
        move_is_legal=True,
        predicted_move_uci="e2e4",
        best_move_uci="e2e4",
        predicted_expected_score=0.70,
        best_expected_score=0.74,
        cp_loss=20.0,
    )
    rubric = ChessMoveRubric(
        game_stockfish_pool=_FakeVerifierPool(verification),
        game_stockfish_depth=20,
        game_reward_illegal_move_cp_loss=1000.0,
        game_reward_legal_floor=0.2,
        game_reward_best_move_bonus=0.05,
        game_reward_expected_score_temperature=0.08,
        game_reward_cp_loss_scale=120.0,
        game_reward_syzygy_wdl_scale=1.0,
        game_reward_syzygy_dtz_scale=20.0,
        game_reward_pv_overlap_bonus=0.05,
        game_reward_pv_motif_plies=6,
        game_reward_use_confidence_weighting=True,
        game_reward_confidence_neutral=0.5,
        game_reward_confidence_nodes_reference=500000,
        game_reward_confidence_seldepth_factor=1.5,
    )
    parser = vf.Parser()
    completion = [{"role": "assistant", "content": "e2e4"}]
    info = {"source": "lichess_game", "fen": chess.STARTING_FEN}
    state = {"task": "chess_next_move", "trajectory": []}
    reward = asyncio.run(
        rubric.chess_move_reward(
            parser=parser,
            completion=completion,
            answer="e2e4",
            info=info,
            state=state,
        )
    )
    assert reward == pytest.approx(0.7352, rel=1e-3)


def test_chess_move_rubric_puzzle_reward_remains_exact_match():
    rubric = ChessMoveRubric(
        game_stockfish_pool=None,
        game_stockfish_depth=20,
        game_reward_illegal_move_cp_loss=1000.0,
        game_reward_legal_floor=0.2,
        game_reward_best_move_bonus=0.05,
        game_reward_expected_score_temperature=0.08,
        game_reward_cp_loss_scale=120.0,
        game_reward_syzygy_wdl_scale=1.0,
        game_reward_syzygy_dtz_scale=20.0,
        game_reward_pv_overlap_bonus=0.05,
        game_reward_pv_motif_plies=6,
        game_reward_use_confidence_weighting=True,
        game_reward_confidence_neutral=0.5,
        game_reward_confidence_nodes_reference=500000,
        game_reward_confidence_seldepth_factor=1.5,
    )
    parser = vf.Parser()
    completion = [{"role": "assistant", "content": "e2e4"}]
    info = {"source": "lichess_puzzle", "fen": chess.STARTING_FEN}
    state = {"task": "chess_next_move", "trajectory": []}
    reward = asyncio.run(
        rubric.chess_move_reward(
            parser=parser,
            completion=completion,
            answer="e2e4",
            info=info,
            state=state,
        )
    )
    assert reward == 1.0


def test_syzygy_penalties_and_pv_overlap_helpers():
    verification = _build_move_verification(
        move_is_legal=True,
        predicted_move_uci="e2e4",
        best_move_uci="g1f3",
        predicted_expected_score=0.5,
        best_expected_score=0.7,
        cp_loss=80.0,
    )
    verification = replace(
        verification,
        best_pv_san=("Nf3", "Nc6", "d4", "d5"),
        predicted_pv_san=("e4", "Nc6", "d4", "Nf6"),
        syzygy_best_wdl=-2,
        syzygy_predicted_wdl=0,
        syzygy_best_dtz=3,
        syzygy_predicted_dtz=9,
    )
    assert _syzygy_wdl_penalty(verification) == 0.0
    assert _syzygy_dtz_penalty(verification) == 0.0
    assert _pv_motif_overlap(verification, motif_plies=3) == pytest.approx(0.5, rel=1e-6)

    worse_wdl = replace(
        verification,
        syzygy_best_wdl=0,
        syzygy_predicted_wdl=-2,
    )
    assert _syzygy_wdl_penalty(worse_wdl) == 2.0

    same_wdl = replace(
        verification,
        syzygy_best_wdl=0,
        syzygy_predicted_wdl=0,
        syzygy_best_dtz=3,
        syzygy_predicted_dtz=9,
    )
    assert _syzygy_dtz_penalty(same_wdl) == 6.0


def test_search_confidence_uses_depth_seldepth_and_nodes():
    verification = _build_move_verification(
        move_is_legal=True,
        predicted_move_uci="e2e4",
        best_move_uci="e2e4",
        predicted_expected_score=0.65,
        best_expected_score=0.70,
        cp_loss=30.0,
    )
    confidence = _search_confidence(
        verification=verification,
        target_depth=20,
        nodes_reference=500000,
        seldepth_factor=1.5,
    )
    assert confidence == pytest.approx(1.0, rel=1e-6)


def test_assemble_mixed_examples_backfills_short_source():
    puzzles = [
        {"task": "chess_next_move", "prompt": [], "answer": "e2e4", "info": {"source": "puzzle"}}
    ]
    games = [
        {"task": "chess_next_move", "prompt": [], "answer": "d2d4", "info": {"source": "game"}},
        {"task": "chess_next_move", "prompt": [], "answer": "g1f3", "info": {"source": "game"}},
        {"task": "chess_next_move", "prompt": [], "answer": "c2c4", "info": {"source": "game"}},
    ]
    mixed = _assemble_mixed_examples(
        puzzle_rows=puzzles,
        game_rows=games,
        max_examples=4,
        puzzles_fraction=0.75,
        seed=42,
    )
    assert len(mixed) == 4
    assert sorted(row["example_id"] for row in mixed) == [0, 1, 2, 3]
    assert sum(1 for row in mixed if row["info"]["source"] == "puzzle") == 1
    assert sum(1 for row in mixed if row["info"]["source"] == "game") == 3


def test_label_selected_game_rows_with_best_moves_labels_only_games_and_dedupes_fens():
    fen_1 = chess.STARTING_FEN
    board = chess.Board(fen_1)
    board.push(chess.Move.from_uci("e2e4"))
    fen_2 = board.fen()
    rows = [
        {
            "task": "chess_next_move",
            "prompt": [],
            "answer": "e2e4",
            "info": {"source": "lichess_game", "fen": fen_1, "game_answer_mode": "pgn"},
        },
        {
            "task": "chess_next_move",
            "prompt": [],
            "answer": "d7d5",
            "info": {"source": "lichess_game", "fen": fen_1, "game_answer_mode": "pgn"},
        },
        {
            "task": "chess_next_move",
            "prompt": [],
            "answer": "e2e4",
            "info": {"source": "lichess_puzzle", "fen": fen_1},
        },
        {
            "task": "chess_next_move",
            "prompt": [],
            "answer": "e7e5",
            "info": {"source": "lichess_game", "fen": fen_2, "game_answer_mode": "pgn"},
        },
    ]
    calls: list[str] = []

    def _fake_best_move_lookup(fen: str) -> tuple[str, float | None] | None:
        calls.append(fen)
        if fen == fen_1:
            return ("g1f3", 0.56)
        return None

    labeled, skipped = _label_selected_game_rows_with_best_moves(
        rows,
        get_best_move=_fake_best_move_lookup,
        num_workers=4,
    )
    assert labeled == 2
    assert skipped == 1
    assert set(calls) == {fen_1, fen_2}
    assert len(calls) == 2
    assert rows[0]["answer"] == "g1f3"
    assert rows[1]["answer"] == "g1f3"
    assert rows[0]["info"]["game_answer_mode"] == "stockfish"
    assert rows[1]["info"]["game_answer_mode"] == "stockfish"
    assert rows[0]["info"]["stockfish_best_move"] == "g1f3"
    assert rows[1]["info"]["stockfish_best_move"] == "g1f3"
    assert rows[2]["answer"] == "e2e4"
    assert rows[2]["info"]["source"] == "lichess_puzzle"
    assert rows[3]["answer"] == "e7e5"
    assert rows[3]["info"]["game_answer_mode"] == "pgn"
