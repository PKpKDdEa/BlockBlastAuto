import numpy as np
import itertools
from typing import List, Tuple, Optional
from model import Board, Piece, Move, is_legal, apply_move
from config import config

def evaluate_board(board: Board) -> float:
    """
    Evaluate board state using weighted heuristic features.
    """
    score = 0.0
    
    # 1. Empty cells (count zeros)
    # Using python's bit_count (Python 3.10+) or alternative
    occupied_count = bin(board.bitboard).count('1')
    empty_cells = 64 - occupied_count
    score += empty_cells * config.WEIGHT_EMPTY_CELLS
    
    # 2. Holes penalty (Cells that are empty but surrounded by filled cells)
    # A simple way: empty cells that have filled neighbors in 4 directions
    # With bitboards, we can shift to find neighbors
    b = board.bitboard
    empty = (~b) & 0xFFFFFFFFFFFFFFFF
    
    # Find empty cells with filled neighbors
    # This is a proxy for holes that is fast to compute
    up = (b >> 8)
    down = (b << 8) & 0xFFFFFFFFFFFFFFFF
    left = (b >> 1) & 0x7F7F7F7F7F7F7F7F
    right = (b << 1) & 0xFEFEFEFEFEFEFEFE
    
    # Hole candidate: empty cell with at least 2 filled neighbors (horizontal or vertical trap)
    # This matches the user's previous logic in a vectorized way
    horizontal_trap = empty & (left & right)
    vertical_trap = empty & (up & down)
    holes = bin(horizontal_trap | vertical_trap).count('1')
    
    score += holes * config.WEIGHT_HOLES_PENALTY
    
    # 3. Bumpiness (Height differences between columns)
    # Harder to do purely bitwise, but we can get column heights
    heights = []
    for c in range(8):
        col_mask = 0x0101010101010101 << c
        col_data = (b & col_mask) >> c
        if col_data == 0:
            heights.append(0)
        else:
            # Find the highest bit set in this column
            # Bits are at 0, 8, 16, 24, 32, 40, 48, 56
            # We want the max r where (1 << (r*8)) is set
            h = 0
            for r in range(8):
                if col_data & (1 << (r * 8)):
                    h = 8 - r
                    break
            heights.append(h)
    
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i+1])
    score += bumpiness * config.WEIGHT_BUMPINESS
    
    # 4. Near-complete lines
    near_complete = 0
    for r in range(8):
        row_mask = 0xFF << (r * 8)
        row_bits = (b & row_mask) >> (r * 8)
        empty_in_row = 8 - bin(row_bits).count('1')
        if 1 <= empty_in_row <= 2:
            near_complete += 1
    
    for c in range(8):
        col_mask = 0x0101010101010101 << c
        col_bits = (b & col_mask)
        # Shift bits to be contiguous for counting effectively
        # Or just count '1's in the sparse bitmask
        empty_in_col = 8 - bin(col_bits).count('1')
        if 1 <= empty_in_col <= 2:
            near_complete += 1
            
    score += near_complete * config.WEIGHT_NEAR_COMPLETE
    
    # 5. Combo streak bonus
    score += board.combo_streak * config.WEIGHT_STREAK_BONUS
    
    return score


def find_sequence_best_move(board: Board, pieces: List[Piece], depth: int = 0, max_depth: int = 99) -> Tuple[float, Optional[List[Move]]]:
    """
    Find the best sequence of moves for all available pieces using depth-first search.
    Optimized with bitboards.
    """
    if not pieces or depth >= max_depth:
        return board.total_score + evaluate_board(board), []
    
    best_score = float('-inf')
    best_seq = None
    
    # Bitboard optimization: we can iterate through possible placements very quickly
    for i, piece in enumerate(pieces):
        remaining_pieces = pieces[:i] + pieces[i+1:]
        
        # Limit search space to valid bounds for this piece
        for r in range(board.rows - piece.height + 1):
            row_shift = r * 8
            for c in range(board.cols - piece.width + 1):
                piece_mask = piece.bitmask << (row_shift + c)
                
                # Fast bitwise collision check
                if (board.bitboard & piece_mask) == 0:
                    # Apply move (bitboard only for deeper search, board.copy handles the rest)
                    new_board, _, _ = apply_move(board, piece, r, c)
                    
                    score, seq = find_sequence_best_move(new_board, remaining_pieces, depth + 1, max_depth)
                    
                    if score > best_score:
                        best_score = score
                        best_seq = [Move(piece_index=piece.id, row=r, col=c)] + (seq if seq else [])
    
    return best_score, best_seq


def best_move(board: Board, pieces: List[Piece], time_budget_ms: int = None) -> Optional[Move]:
    """
    Compute the best move by looking at the entire sequence of available pieces.
    """
    valid_pieces = [p for p in pieces if p is not None]
    if not valid_pieces:
        return None
        
    # Bitboard solver is fast enough to always look ahead if pieces <= 3
    # but we keep the depth limit logic for safety
    empty_cells = 64 - bin(board.bitboard).count('1')
    max_depth = 99
    if empty_cells > 45:
        max_depth = 1 # Still use fast mode for very empty boards to save time
        
    best_score, best_seq = find_sequence_best_move(board, valid_pieces, max_depth=max_depth)
    
    if best_seq and len(best_seq) > 0:
        return best_seq[0]
    
    return None


if __name__ == "__main__":
    # Test solver
    print("Testing solver...\n")
    
    # Create test board with some filled cells
    board = Board(8, 8)
    board.grid[7, :] = 1  # Fill bottom row except one cell
    board.grid[7, 3] = 0
    
    print("Test board:")
    print(board)
    print()
    
    # Create test pieces
    pieces = [
        Piece(id=0, cells=[(0, 0)], width=1, height=1),  # Single block
        Piece(id=1, cells=[(0, 0), (0, 1), (0, 2)], width=3, height=1),  # Horizontal line
        Piece(id=2, cells=[(0, 0), (1, 0)], width=1, height=2),  # Vertical 2-block
    ]
    
    # Find best move
    move = best_move(board, pieces)
    
    if move:
        print(f"\nBest move: Place piece {move.piece_index} at ({move.row}, {move.col})")
        print(f"Score: {move.score:.2f}")
        
        # Apply and show result
        new_board, lines, score = apply_move(board, pieces[move.piece_index], move.row, move.col)
        print(f"\nResulting board (lines cleared: {lines}):")
        print(new_board)
    else:
        print("No legal moves!")
