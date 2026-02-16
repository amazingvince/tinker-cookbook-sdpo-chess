from __future__ import annotations

from typing import Any

import pytest

chess = pytest.importorskip("chess")

from tinker_cookbook.sdpo.chess_hints import (  # noqa: E402
    MoveHint,
    PositionHintPack,
    StockfishHintConfig,
    ThreatSummary,
    build_stockfish_hint_text_for_state,
    extract_fen_from_state,
    extract_fen_from_text,
    extract_predicted_move,
    pick_random_game_fen,
    render_hint_text,
    summarize_threats,
    wdl_to_stats,
)


def test_wdl_to_stats_expected_score():
    stats = wdl_to_stats(wins=300, draws=200, losses=500)
    assert stats.win_probability == 0.3
    assert stats.draw_probability == 0.2
    assert stats.loss_probability == 0.5
    assert stats.expected_score == 0.4


def test_extract_fen_from_state_and_prompt():
    fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
    state = {"info": {"FEN": fen}}
    assert extract_fen_from_state(state) == fen

    prompt_messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Play strong chess."},
        {"role": "user", "content": f"Given this FEN: {fen}\nReturn one UCI move."},
    ]
    assert extract_fen_from_state({}, prompt_messages) == fen
    assert extract_fen_from_text(f"Example {fen} end") == fen


def test_summarize_threats_hanging_piece_counts():
    board = chess.Board("r7/8/8/8/8/8/Q7/7K w - - 0 1")
    summary = summarize_threats(board)
    assert summary.side_to_move_hanging == ("wQ@a2",)
    assert summary.opponent_hanging == ("bR@a8",)
    assert summary.side_to_move_threatened_count >= 1
    assert summary.opponent_threatened_count >= 1


def test_render_hint_text_includes_good_and_bad_move_sections():
    pack = PositionHintPack(
        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        side_to_move="w",
        root_wdl=wdl_to_stats(540, 300, 160),
        threat_summary=ThreatSummary(
            side_to_move_hanging=(),
            opponent_hanging=(),
            side_to_move_threatened_count=0,
            opponent_threatened_count=0,
            legal_checking_moves=0,
        ),
        candidate_moves=(
            MoveHint(
                uci="a1a2",
                san="Ka2",
                expected_score=0.61,
                delta_expected_score=0.0,
                centipawn_score=35.0,
                pv_san=("Ka2", "Kh2"),
                refutation_san="Kh2",
                is_capture=False,
                gives_check=False,
                is_promotion=False,
                hangs_moved_piece=False,
            ),
            MoveHint(
                uci="a1b1",
                san="Kb1",
                expected_score=0.48,
                delta_expected_score=0.13,
                centipawn_score=-12.0,
                pv_san=("Kb1", "Kh2"),
                refutation_san="Kh2",
                is_capture=False,
                gives_check=False,
                is_promotion=False,
                hangs_moved_piece=True,
            ),
        ),
    )
    text = render_hint_text(
        pack,
        StockfishHintConfig(max_good_moves=2, max_bad_moves=2, bad_move_threshold=0.05),
    )
    assert "Root expected score (WDL)" in text
    assert "Top candidate moves by expected score:" in text
    assert "Moves likely to be bad:" in text
    assert "cp=+35.0" in text
    assert "hangs moved piece" in text


def test_extract_predicted_move_from_uci_and_san():
    board = chess.Board()
    from_uci = extract_predicted_move(board, "My move is e2e4.")
    assert from_uci is not None
    assert from_uci.uci() == "e2e4"

    from_san = extract_predicted_move(board, "I choose Nf3 here.")
    assert from_san is not None
    assert from_san.uci() == "g1f3"


def test_pick_random_game_fen_returns_valid_fen():
    movetext = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
    fen = pick_random_game_fen(movetext, seed=3)
    assert fen is not None
    board = chess.Board(fen)
    assert board.is_valid()


def test_build_stockfish_hint_text_for_state_without_extractor():
    state = {"info": {"fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}}
    text = build_stockfish_hint_text_for_state(
        state=state,
        prompt_messages=[],
        extractor=None,
    )
    assert text is None
