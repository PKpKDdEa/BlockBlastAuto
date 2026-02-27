"""
Game state model and logic.
Pure functions for board representation and move validation.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass(eq=False)
class Piece:
    """Represents a game piece."""
    id: int
    cells: List[Tuple[int, int]]  # List of (row, col) offsets from anchor
    width: int
    height: int
    raw_mask: Optional[np.ndarray] = None # Original 5x5 detection for learning
    is_new: bool = False # Whether this piece is a new unidentified pattern
    bitmask: int = 0  # Bitboard representation (64-bit)

    @classmethod
    def from_mask(cls, piece_id: int, mask: np.ndarray, is_new: bool = False) -> 'Piece':
        """
        Create piece from binary mask. Automatically trims empty space
        to keep piece offsets relative to its own top-left.
        """
        raw_mask = mask.copy() if mask.shape == (5, 5) else None
        
        rows, cols = np.where(mask == 1)
        if len(rows) == 0:
            return cls(id=piece_id, cells=[], width=0, height=0, raw_mask=raw_mask)
            
        min_r, max_r = np.min(rows), np.max(rows)
        min_c, max_c = np.min(cols), np.max(cols)
        
        cells = []
        bitmask = 0
        for r, c in zip(rows, cols):
            rel_r, rel_c = int(r - min_r), int(c - min_c)
            cells.append((rel_r, rel_c))
            bitmask |= (1 << (rel_r * 8 + rel_c))
            
        height = int(max_r - min_r + 1)
        width = int(max_c - min_c + 1)
        
        return cls(id=piece_id, cells=cells, width=width, height=height, 
                   raw_mask=raw_mask, is_new=is_new, bitmask=bitmask)
    
    def __eq__(self, other):
        """Custom equality check to handle numpy arrays in raw_mask."""
        if not isinstance(other, Piece):
            return False
        if self.id != other.id or self.width != other.width or self.height != other.height:
            return False
        if self.cells != other.cells:
            return False
        if self.bitmask != other.bitmask:
            return False
        if self.raw_mask is not None and other.raw_mask is not None:
            return np.array_equal(self.raw_mask, other.raw_mask)
        return self.raw_mask is other.raw_mask
    
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
        self.bitboard = 0  # 64-bit integer
        self.combo_streak = 0  # Number of consecutive placements that cleared lines
        self.total_score = 0
    
    @classmethod
    def from_array(cls, grid: np.ndarray) -> 'Board':
        """Create board from existing array."""
        board = cls(rows=grid.shape[0], cols=grid.shape[1])
        board.grid = grid.copy()
        
        # Sync bitboard
        board.bitboard = 0
        rows, cols = np.where(grid == 1)
        for r, c in zip(rows, cols):
            board.bitboard |= (1 << (r * 8 + c))
            
        return board
    
    def copy(self) -> 'Board':
        """Create a fast copy of this board."""
        new_board = Board(self.rows, self.cols)
        new_board.grid[:] = self.grid
        new_board.bitboard = self.bitboard
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
    """
    # Quick bounds check
    if row < 0 or row + piece.height > board.rows or col < 0 or col + piece.width > board.cols:
        return False
    
    # Bitmask-based collision check
    piece_mask = piece.bitmask << (row * 8 + col)
    if (board.bitboard & piece_mask) != 0:
        return False
        
    return True


def apply_move(board: Board, piece: Piece, row: int, col: int) -> Tuple[Board, int, int]:
    """
    Apply a move and return new board state.
    """
    new_board = board.copy()
    
    # Place piece on bitboard
    piece_mask = piece.bitmask << (row * 8 + col)
    new_board.bitboard |= piece_mask
    
    # Sync numpy grid
    for dr, dc in piece.cells:
        new_board.grid[row + dr, col + dc] = 1
    
    # Check for completed lines using bitmasks
    lines_cleared = 0
    
    # Pre-calculated masks for whole rows/cols
    # Row masks: 0xFF, 0xFF00, ...
    # Col masks: 0x0101010101010101, 0x0202020202020202, ...
    
    rows_to_clear = []
    for r in range(8):
        row_mask = 0xFF << (r * 8)
        if (new_board.bitboard & row_mask) == row_mask:
            rows_to_clear.append(r)
            lines_cleared += 1
            
    cols_to_clear = []
    for c in range(8):
        col_mask = 0x0101010101010101 << c
        if (new_board.bitboard & col_mask) == col_mask:
            cols_to_clear.append(c)
            lines_cleared += 1
            
    # Clear lines on bitboard
    for r in rows_to_clear:
        row_mask = 0xFF << (r * 8)
        new_board.bitboard &= ~row_mask
        new_board.grid[r, :] = 0
        
    for c in cols_to_clear:
        col_mask = 0x0101010101010101 << c
        new_board.bitboard &= ~col_mask
        new_board.grid[:, c] = 0
    
    # Score calculation
    base_points = len(piece.cells)
    clear_points = 0
    if lines_cleared > 0:
        clear_points = (lines_cleared * (lines_cleared + 1) // 2) * 10
        new_board.combo_streak += 1
        clear_points *= max(1, new_board.combo_streak)
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
