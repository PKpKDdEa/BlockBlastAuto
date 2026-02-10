"""
Game state model and logic.
Pure functions for board representation and move validation.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Piece:
    """Represents a game piece."""
    id: int
    cells: List[Tuple[int, int]]  # List of (row, col) offsets from anchor
    width: int
    height: int
    
    @classmethod
    def from_mask(cls, piece_id: int, mask: np.ndarray) -> 'Piece':
        """
        Create piece from binary mask.
        
        Args:
            piece_id: Unique identifier for this piece
            mask: 2D binary array where 1 = filled cell
        
        Returns:
            Piece instance
        """
        cells = []
        rows, cols = np.where(mask == 1)
        for r, c in zip(rows, cols):
            cells.append((int(r), int(c)))
        
        height, width = mask.shape
        return cls(id=piece_id, cells=cells, width=width, height=height)
    
    @property
    def anchor_offset(self) -> Tuple[float, float]:
        """
        Return the offset from piece cell (0,0) to the visual center 
        in grid units.
        """
        if not self.cells:
            return 0.0, 0.0
        
        # Center of the bounding box
        center_r = (self.height - 1) / 2.0
        center_c = (self.width - 1) / 2.0
        
        return center_r, center_c


@dataclass
class Move:
    """Represents a move: placing a piece at a position."""
    piece_index: int  # Index in the pieces list (0-2)
    row: int  # Target row on board
    col: int  # Target col on board
    score: float = 0.0  # Evaluation score


class Board:
    """Represents the game board state."""
    
    def __init__(self, rows: int = 8, cols: int = 8):
        """
        Initialize empty board.
        
        Args:
            rows: Number of rows
            cols: Number of columns
        """
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=np.int8)
        self.combo_streak = 0  # Number of consecutive placements that cleared lines
        self.total_score = 0
    
    @classmethod
    def from_array(cls, grid: np.ndarray) -> 'Board':
        """Create board from existing array."""
        board = cls(rows=grid.shape[0], cols=grid.shape[1])
        board.grid = grid.copy()
        return board
    
    def copy(self) -> 'Board':
        """Create a fast copy of this board."""
        new_board = Board(self.rows, self.cols)
        new_board.grid[:] = self.grid
        new_board.combo_streak = self.combo_streak
        new_board.total_score = self.total_score
        return new_board
    
    def __str__(self) -> str:
        """ASCII representation of board."""
        lines = []
        for row in self.grid:
            line = ''.join(['█' if cell else '·' for cell in row])
            lines.append(line)
        return '\n'.join(lines)


def is_legal(board: Board, piece: Piece, row: int, col: int) -> bool:
    """
    Check if placing a piece at position is legal.
    
    Args:
        board: Current board state
        piece: Piece to place
        row: Target row (anchor position)
        col: Target col (anchor position)
    
    Returns:
        True if move is legal, False otherwise
    """
    for dr, dc in piece.cells:
        r = row + dr
        c = col + dc
        
        # Check bounds
        if r < 0 or r >= board.rows or c < 0 or c >= board.cols:
            return False
        
        # Check if cell is already occupied
        if board.grid[r, c] != 0:
            return False
    
    return True


def apply_move(board: Board, piece: Piece, row: int, col: int) -> Tuple[Board, int, int]:
    """
    Apply a move and return new board state.
    
    Args:
        board: Current board state
        piece: Piece to place
        row: Target row
        col: Target col
    
    Returns:
        Tuple of (new_board, lines_cleared, score_gain)
    """
    # Create new board
    new_board = board.copy()
    
    # Place piece
    for dr, dc in piece.cells:
        r = row + dr
        c = col + dc
        new_board.grid[r, c] = 1
    
    # Check for completed lines
    lines_cleared = 0
    rows_to_clear = []
    cols_to_clear = []
    
    # Check rows
    for r in range(new_board.rows):
        if np.all(new_board.grid[r, :] == 1):
            rows_to_clear.append(r)
            lines_cleared += 1
    
    # Check columns
    for c in range(new_board.cols):
        if np.all(new_board.grid[:, c] == 1):
            cols_to_clear.append(c)
            lines_cleared += 1
    
    # Clear lines
    for r in rows_to_clear:
        new_board.grid[r, :] = 0
    for c in cols_to_clear:
        new_board.grid[:, c] = 0
    
    # Calculate score based on official-like rules
    # 1. Base points for pieces placed
    base_points = len(piece.cells)
    
    # 2. Line clear points (exponential for combos)
    clear_points = 0
    if lines_cleared > 0:
        # Combo: multiple lines in one move
        # e.g., 1 line = 10, 2 lines = 30, 3 lines = 60, etc.
        clear_points = (lines_cleared * (lines_cleared + 1) // 2) * 10
        
        # Streak: consecutive moves that clear lines
        # This usually multiples the clear_points
        new_board.combo_streak += 1
        streak_multiplier = max(1, new_board.combo_streak)
        clear_points *= streak_multiplier
    else:
        new_board.combo_streak = 0
    
    score_gain = base_points + clear_points
    new_board.total_score += score_gain
    
    return new_board, lines_cleared, score_gain


def generate_moves(board: Board, pieces: List[Piece]) -> List[Move]:
    """
    Generate all legal moves for given pieces.
    
    Args:
        board: Current board state
        pieces: List of available pieces
    
    Returns:
        List of legal moves
    """
    moves = []
    
    for piece_idx, piece in enumerate(pieces):
        # Try all positions
        for row in range(board.rows):
            for col in range(board.cols):
                if is_legal(board, piece, row, col):
                    moves.append(Move(piece_index=piece_idx, row=row, col=col))
    
    return moves


if __name__ == "__main__":
    # Test the model
    print("Testing Board and Piece model...\n")
    
    # Create a test board
    board = Board(8, 8)
    board.grid[0, :] = 1  # Fill first row
    board.grid[:, 0] = 1  # Fill first column
    board.grid[0, 0] = 1  # Overlap
    
    print("Initial board:")
    print(board)
    print()
    
    # Create a test piece (L-shape)
    piece = Piece(id=1, cells=[(0, 0), (1, 0), (2, 0), (2, 1)], width=2, height=3)
    
    # Test legal move
    print(f"Is (3, 3) legal? {is_legal(board, piece, 3, 3)}")
    print(f"Is (0, 0) legal? {is_legal(board, piece, 0, 0)}")
    print()
    
    # Apply move
    new_board, lines, score = apply_move(board, piece, 3, 3)
    print("After placing piece at (3, 3):")
    print(new_board)
    print(f"Lines cleared: {lines}, Score: {score}")
