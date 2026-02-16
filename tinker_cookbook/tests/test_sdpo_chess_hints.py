from __future__ import annotations

from typing import Any

import pytest

chess = pytest.importorskip("chess")

from tinker_cookbook.sdpo.chess_hints import (  # noqa: E402
    MoveHint,
    PositionHintPack,
    StockfishHintConfig,
    StockfishHintExtractor,
    ThreatSummary,
    _compute_cp_loss,
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


def test_summarize_threats_ignores_pinned_defender():
    board = chess.Board("4r2k/6b1/8/8/3P4/8/4N3/4K3 w - - 0 1")
    summary = summarize_threats(board)
    assert "wP@d4" in summary.side_to_move_hanging


def test_fen_decode_ignores_pinned_attacker_pressure():
    fen = "4k3/4n3/8/3P4/8/8/8/4R2K w - - 0 1"
    board = chess.Board(fen)
    pack = PositionHintPack(
        fen=fen,
        side_to_move="w",
        root_wdl=wdl_to_stats(500, 300, 200),
        threat_summary=summarize_threats(board),
        candidate_moves=(),
    )
    text = render_hint_text(pack, StockfishHintConfig(include_fen_decode=True))
    assert "Threat summary: threatened_own=0" in text
    assert "white pieces under pressure: none" in text


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
    assert "Position decode from FEN:" in text
    assert "Board (white uppercase, black lowercase):" in text
    assert "pieces under pressure" in text
    assert "Weak king-zone squares" in text
    assert "Top candidate moves by expected score:" in text
    assert "Moves likely to be bad:" in text
    assert "Trap analysis (future-state refutations):" in text
    assert "refutation:" in text
    assert "motifs:" in text
    assert "cp=+35.0" in text
    assert "hangs moved piece" in text

    no_decode_text = render_hint_text(
        pack,
        StockfishHintConfig(
            include_fen_decode=False,
            max_good_moves=2,
            max_bad_moves=2,
            bad_move_threshold=0.05,
        ),
    )
    assert "Position decode from FEN:" not in no_decode_text


def test_render_hint_text_includes_search_and_syzygy_summary():
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
        candidate_moves=(),
        root_centipawn_score=22.0,
        root_search_depth=18,
        root_selective_depth=28,
        root_nodes=1_250_000,
        root_nps=4_500_000,
        root_tablebase_hits=12,
        syzygy_root_wdl=2,
        syzygy_root_dtz=7,
    )
    text = render_hint_text(pack, StockfishHintConfig(include_search_stats=True))
    assert "Root centipawn score (side to move): +22.0" in text
    assert "Root search stats: d=18, sd=28, nodes=1.2M, nps=4.5M, tbhits=12" in text
    assert "Syzygy root: tablebase win, DTZ=7" in text


def test_compute_cp_loss_sources():
    cp_loss, source = _compute_cp_loss(
        best_cp=40.0,
        predicted_cp=-10.0,
        best_expected_score=None,
        predicted_expected_score=None,
        best_move_uci="e2e4",
        predicted_move_uci="d2d4",
        unknown_score_cp_loss=80.0,
    )
    assert cp_loss == 50.0
    assert source == "centipawn"

    cp_loss, source = _compute_cp_loss(
        best_cp=None,
        predicted_cp=None,
        best_expected_score=0.72,
        predicted_expected_score=0.55,
        best_move_uci="e2e4",
        predicted_move_uci="d2d4",
        unknown_score_cp_loss=80.0,
    )
    assert cp_loss == pytest.approx(170.0, rel=1e-6)
    assert source == "wdl_scaled"

    cp_loss, source = _compute_cp_loss(
        best_cp=None,
        predicted_cp=None,
        best_expected_score=None,
        predicted_expected_score=None,
        best_move_uci="e2e4",
        predicted_move_uci="d2d4",
        unknown_score_cp_loss=80.0,
    )
    assert cp_loss == 80.0
    assert source == "fallback_penalty"

    cp_loss, source = _compute_cp_loss(
        best_cp=None,
        predicted_cp=None,
        best_expected_score=None,
        predicted_expected_score=None,
        best_move_uci="e2e4",
        predicted_move_uci="e2e4",
        unknown_score_cp_loss=80.0,
    )
    assert cp_loss == 0.0
    assert source == "same_move"


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


def test_persistent_cache_key_prefix_changes_with_semantic_config():
    base_cfg = StockfishHintConfig(
        stockfish_path="/tmp/stockfish-a",
        wdl_model="sf",
        max_pv_plies=6,
        syzygy_path="/tmp/syzygy",
        syzygy_max_pieces=5,
        unknown_score_cp_loss=80.0,
    )
    same_cfg = StockfishHintConfig(
        stockfish_path="/tmp/stockfish-a",
        wdl_model="sf",
        max_pv_plies=6,
        syzygy_path="/tmp/syzygy",
        syzygy_max_pieces=5,
        unknown_score_cp_loss=80.0,
    )
    changed_cfg = StockfishHintConfig(
        stockfish_path="/tmp/stockfish-a",
        wdl_model="lichess",
        max_pv_plies=6,
        syzygy_path="/tmp/syzygy",
        syzygy_max_pieces=5,
        unknown_score_cp_loss=80.0,
    )

    base_prefix = StockfishHintExtractor._build_persistent_key_prefix(base_cfg)
    same_prefix = StockfishHintExtractor._build_persistent_key_prefix(same_cfg)
    changed_prefix = StockfishHintExtractor._build_persistent_key_prefix(changed_cfg)

    assert base_prefix == same_prefix
    assert base_prefix != changed_prefix


def test_build_stockfish_hint_text_for_state_without_extractor():
    state = {"info": {"fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}}
    text = build_stockfish_hint_text_for_state(
        state=state,
        prompt_messages=[],
        extractor=None,
    )
    assert text is None
