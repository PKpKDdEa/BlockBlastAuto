"""
Solver engine for computing best moves using heuristics.
"""
import numpy as np
from typing import List, Optional
from model import Board, Piece, Move, is_legal, apply_move, generate_moves
from config import config


def evaluate_board(board: Board) -> float:
    """
    Evaluate board state using heuristic features.
    
    Features:
    - Empty cells (more is better)
    - Holes (cells surrounded by filled cells, fewer is better)
    - Near-complete lines (rows/cols with 1-2 empty cells)
    - Central free space (prefer empty cells near center)
    
    Args:
        board: Board state to evaluate
    
    Returns:
        Score (higher is better)
    """
    score = 0.0
    
    # Count empty cells
    empty_cells = np.sum(board.grid == 0)
    score += empty_cells * 0.1
    
    # Count holes (empty cells surrounded by filled cells)
    holes = 0
    for r in range(1, board.rows - 1):
        for c in range(1, board.cols - 1):
            if board.grid[r, c] == 0:
                # Check if surrounded
                neighbors = [
                    board.grid[r-1, c], board.grid[r+1, c],
                    board.grid[r, c-1], board.grid[r, c+1]
                ]
                if all(n == 1 for n in neighbors):
                    holes += 1
    score -= holes * 2.0
    
    # Near-complete lines (rows/cols with 1-2 empty cells)
    near_complete = 0
    for r in range(board.rows):
        empty_in_row = np.sum(board.grid[r, :] == 0)
        if 1 <= empty_in_row <= 2:
            near_complete += 1
    
    for c in range(board.cols):
        empty_in_col = np.sum(board.grid[:, c] == 0)
        if 1 <= empty_in_col <= 2:
            near_complete += 1
    score += near_complete * 0.5
    
    # Central free space (prefer keeping center clear)
    center_r = board.rows // 2
    center_c = board.cols // 2
    center_region = board.grid[center_r-1:center_r+2, center_c-1:center_c+2]
    central_empty = np.sum(center_region == 0)
    score += central_empty * 0.3
    
    return score


def evaluate_move(board: Board, piece: Piece, move: Move) -> float:
    """
    Evaluate a specific move.
    
    Args:
        board: Current board state
        piece: Piece to place
        move: Move to evaluate
    
    Returns:
        Score for this move (higher is better)
    """
    # Apply move and get resulting board
    new_board, lines_cleared, score_gain = apply_move(board, piece, move.row, move.col)
    
    # Base score from lines cleared (heavily weighted)
    score = lines_cleared * 10.0
    
    # Add immediate score gain
    score += score_gain * 0.5
    
    # Evaluate resulting board state
    board_score = evaluate_board(new_board)
    score += board_score
    
    return score


def best_move(board: Board, pieces: List[Piece], time_budget_ms: int = None) -> Optional[Move]:
    """
    Find the best move using heuristic evaluation.
    
    Args:
        board: Current board state
        pieces: Available pieces
        time_budget_ms: Time budget in milliseconds (currently unused, for future optimization)
    
    Returns:
        Best move or None if no legal moves exist
    """
    if time_budget_ms is None:
        time_budget_ms = config.SOLVER_TIME_BUDGET_MS
    
    # Generate all legal moves
    moves = generate_moves(board, pieces)
    
    if not moves:
        return None
    
    # Evaluate each move
    best_score = float('-inf')
    best_mv = None
    
    for move in moves:
        piece = pieces[move.piece_index]
        score = evaluate_move(board, piece, move)
        move.score = score
        
        if score > best_score:
            best_score = score
            best_mv = move
    
    if config.DEBUG:
        print(f"Evaluated {len(moves)} moves, best score: {best_score:.2f}")
        if best_mv:
            print(f"Best move: piece {best_mv.piece_index} at ({best_mv.row}, {best_mv.col})")
    
    return best_mv


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
