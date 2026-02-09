"""
Unit tests for solver heuristics.
"""
import pytest
import numpy as np
from model import Board, Piece
from solver import evaluate_board, evaluate_move, best_move


def test_evaluate_board_empty():
    """Test board evaluation for empty board."""
    board = Board(8, 8)
    score = evaluate_board(board)
    
    # Empty board should have positive score (lots of empty cells)
    assert score > 0


def test_evaluate_board_holes():
    """Test hole penalty."""
    board1 = Board(8, 8)
    
    board2 = Board(8, 8)
    # Create a hole (empty cell surrounded by filled cells)
    board2.grid[3, 3] = 0
    board2.grid[2, 3] = 1
    board2.grid[4, 3] = 1
    board2.grid[3, 2] = 1
    board2.grid[3, 4] = 1
    
    score1 = evaluate_board(board1)
    score2 = evaluate_board(board2)
    
    # Board with hole should score lower
    assert score2 < score1


def test_evaluate_move_line_clear():
    """Test that moves clearing lines score higher."""
    board = Board(8, 8)
    board.grid[0, :7] = 1  # Almost complete row
    
    piece = Piece(id=0, cells=[(0, 0)], width=1, height=1)
    
    from model import Move
    move_clear = Move(piece_index=0, row=0, col=7)  # Completes row
    move_normal = Move(piece_index=0, row=5, col=5)  # Normal placement
    
    score_clear = evaluate_move(board, piece, move_clear)
    score_normal = evaluate_move(board, piece, move_normal)
    
    # Line-clearing move should score much higher
    assert score_clear > score_normal


def test_best_move_simple():
    """Test best move selection."""
    board = Board(8, 8)
    pieces = [
        Piece(id=0, cells=[(0, 0)], width=1, height=1),
    ]
    
    move = best_move(board, pieces)
    
    assert move is not None
    assert move.piece_index == 0
    assert 0 <= move.row < 8
    assert 0 <= move.col < 8


def test_best_move_prefers_line_clear():
    """Test that solver prefers line-clearing moves."""
    board = Board(8, 8)
    board.grid[0, :7] = 1  # Almost complete row
    
    pieces = [
        Piece(id=0, cells=[(0, 0)], width=1, height=1),
    ]
    
    move = best_move(board, pieces)
    
    # Should choose to complete the row
    assert move.row == 0
    assert move.col == 7


def test_best_move_no_legal_moves():
    """Test when no legal moves exist."""
    board = Board(8, 8)
    board.grid[:, :] = 1  # Fill entire board
    
    pieces = [
        Piece(id=0, cells=[(0, 0)], width=1, height=1),
    ]
    
    move = best_move(board, pieces)
    
    assert move is None


def test_best_move_multiple_pieces():
    """Test with multiple pieces."""
    board = Board(8, 8)
    board.grid[0, :6] = 1  # Partial row
    
    pieces = [
        Piece(id=0, cells=[(0, 0)], width=1, height=1),  # Single
        Piece(id=1, cells=[(0, 0), (0, 1)], width=2, height=1),  # 2-block
        Piece(id=2, cells=[(0, 0), (0, 1), (0, 2)], width=3, height=1),  # 3-block
    ]
    
    move = best_move(board, pieces)
    
    assert move is not None
    # Should prefer the 2-block piece to complete the row
    assert move.piece_index == 1
    assert move.row == 0
    assert move.col == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
