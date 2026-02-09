"""
Unit tests for game model (Board, Piece, Move).
"""
import pytest
import numpy as np
from model import Board, Piece, Move, is_legal, apply_move, generate_moves


def test_board_creation():
    """Test board initialization."""
    board = Board(8, 8)
    assert board.rows == 8
    assert board.cols == 8
    assert board.grid.shape == (8, 8)
    assert np.all(board.grid == 0)


def test_board_copy():
    """Test board copying."""
    board = Board(8, 8)
    board.grid[0, 0] = 1
    
    copy = board.copy()
    assert np.array_equal(board.grid, copy.grid)
    
    # Modify copy shouldn't affect original
    copy.grid[1, 1] = 1
    assert board.grid[1, 1] == 0


def test_piece_from_mask():
    """Test piece creation from mask."""
    mask = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])
    
    piece = Piece.from_mask(0, mask)
    assert piece.id == 0
    assert piece.width == 3
    assert piece.height == 3
    assert len(piece.cells) == 4


def test_is_legal_valid():
    """Test legal move detection."""
    board = Board(8, 8)
    piece = Piece(id=0, cells=[(0, 0), (0, 1)], width=2, height=1)
    
    # Should be legal in empty board
    assert is_legal(board, piece, 0, 0)
    assert is_legal(board, piece, 3, 3)


def test_is_legal_out_of_bounds():
    """Test out of bounds detection."""
    board = Board(8, 8)
    piece = Piece(id=0, cells=[(0, 0), (0, 1), (0, 2)], width=3, height=1)
    
    # Should be illegal at right edge
    assert not is_legal(board, piece, 0, 7)
    assert not is_legal(board, piece, 0, 6)
    assert is_legal(board, piece, 0, 5)


def test_is_legal_occupied():
    """Test occupied cell detection."""
    board = Board(8, 8)
    board.grid[3, 3] = 1
    
    piece = Piece(id=0, cells=[(0, 0), (0, 1)], width=2, height=1)
    
    # Should be illegal if overlaps occupied cell
    assert not is_legal(board, piece, 3, 3)
    assert not is_legal(board, piece, 3, 2)
    assert is_legal(board, piece, 3, 4)


def test_apply_move_simple():
    """Test applying a move."""
    board = Board(8, 8)
    piece = Piece(id=0, cells=[(0, 0), (0, 1), (0, 2)], width=3, height=1)
    
    new_board, lines, score = apply_move(board, piece, 0, 0)
    
    # Check piece was placed
    assert new_board.grid[0, 0] == 1
    assert new_board.grid[0, 1] == 1
    assert new_board.grid[0, 2] == 1
    
    # Original board unchanged
    assert np.all(board.grid == 0)


def test_apply_move_line_clear_row():
    """Test line clearing (row)."""
    board = Board(8, 8)
    board.grid[0, :7] = 1  # Fill first row except last cell
    
    piece = Piece(id=0, cells=[(0, 0)], width=1, height=1)
    
    new_board, lines, score = apply_move(board, piece, 0, 7)
    
    # Row should be cleared
    assert lines == 1
    assert np.all(new_board.grid[0, :] == 0)


def test_apply_move_line_clear_col():
    """Test line clearing (column)."""
    board = Board(8, 8)
    board.grid[:7, 0] = 1  # Fill first column except last cell
    
    piece = Piece(id=0, cells=[(0, 0)], width=1, height=1)
    
    new_board, lines, score = apply_move(board, piece, 7, 0)
    
    # Column should be cleared
    assert lines == 1
    assert np.all(new_board.grid[:, 0] == 0)


def test_apply_move_line_clear_both():
    """Test clearing both row and column."""
    board = Board(8, 8)
    board.grid[3, :] = 1  # Fill row 3
    board.grid[:, 3] = 1  # Fill col 3
    board.grid[3, 3] = 0  # Except intersection
    
    piece = Piece(id=0, cells=[(0, 0)], width=1, height=1)
    
    new_board, lines, score = apply_move(board, piece, 3, 3)
    
    # Both should be cleared
    assert lines == 2
    assert np.all(new_board.grid[3, :] == 0)
    assert np.all(new_board.grid[:, 3] == 0)


def test_generate_moves():
    """Test move generation."""
    board = Board(8, 8)
    pieces = [
        Piece(id=0, cells=[(0, 0)], width=1, height=1),
        Piece(id=1, cells=[(0, 0), (0, 1)], width=2, height=1),
    ]
    
    moves = generate_moves(board, pieces)
    
    # Single block: 64 positions
    # 2-block horizontal: 8 rows * 7 cols = 56 positions
    # Total: 120
    assert len(moves) == 120


def test_generate_moves_constrained():
    """Test move generation with obstacles."""
    board = Board(8, 8)
    board.grid[0, :] = 1  # Fill first row
    
    pieces = [
        Piece(id=0, cells=[(0, 0)], width=1, height=1),
    ]
    
    moves = generate_moves(board, pieces)
    
    # Should have 64 - 8 = 56 positions
    assert len(moves) == 56


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
