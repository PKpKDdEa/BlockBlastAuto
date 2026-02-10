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
    
    # 1. Empty cells
    empty_cells = np.sum(board.grid == 0)
    score += empty_cells * config.WEIGHT_EMPTY_CELLS
    
    # 2. Holes penalty
    holes = 0
    for r in range(board.rows):
        for c in range(board.cols):
            if board.grid[r, c] == 0:
                is_hole = False
                if (c > 0 and board.grid[r, c-1] == 1) and (c < board.cols-1 and board.grid[r, c+1] == 1):
                    is_hole = True
                if (r > 0 and board.grid[r-1, c] == 1) and (r < board.rows-1 and board.grid[r+1, c] == 1):
                    is_hole = True
                if is_hole:
                    holes += 1
    score += holes * config.WEIGHT_HOLES_PENALTY
    
    # 3. Bumpiness
    heights = []
    for c in range(board.cols):
        h = 0
        for r in range(board.rows):
            if board.grid[r, c] == 1:
                h = board.rows - r
                break
        heights.append(h)
    
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i+1])
    score += bumpiness * config.WEIGHT_BUMPINESS
    
    # 4. Near-complete lines
    near_complete = 0
    for r in range(board.rows):
        empty_in_row = np.sum(board.grid[r, :] == 0)
        if 1 <= empty_in_row <= 2:
            near_complete += 1
    
    for c in range(board.cols):
        empty_in_col = np.sum(board.grid[:, c] == 0)
        if 1 <= empty_in_col <= 2:
            near_complete += 1
    score += near_complete * config.WEIGHT_NEAR_COMPLETE
    
    # 5. Combo streak bonus
    score += board.combo_streak * config.WEIGHT_STREAK_BONUS
    
    return score


def find_sequence_best_move(board: Board, pieces: List[Piece], depth: int = 0) -> Tuple[float, Optional[List[Move]]]:
    """
    Find the best sequence of moves for all available pieces using depth-first search.
    """
    if not pieces:
        return evaluate_board(board), []
    
    best_score = float('-inf')
    best_seq = None
    
    # Try all available pieces as the first piece in this subsequence
    for i, piece in enumerate(pieces):
        remaining_pieces = pieces[:i] + pieces[i+1:]
        
        # Generate all legal moves for this piece
        moves = []
        for r in range(board.rows):
            for c in range(board.cols):
                if is_legal(board, piece, r, c):
                    moves.append(Move(piece_index=piece.id, row=r, col=c))
        
        # Optimization: Sort moves by immediate score gain to prune or prioritize
        for move in moves:
            new_board, _, _ = apply_move(board, piece, move.row, move.col)
            
            # Recursive call for remaining pieces
            score, seq = find_sequence_best_move(new_board, remaining_pieces, depth + 1)
            
            if score > best_score:
                best_score = score
                best_seq = [move] + (seq if seq else [])
    
    return best_score, best_seq


def best_move(board: Board, pieces: List[Piece], time_budget_ms: int = None) -> Optional[Move]:
    """
    Compute the best move by looking at the entire sequence of available pieces.
    """
    # Filter out None pieces (empty slots)
    valid_pieces = [p for p in pieces if p is not None]
    if not valid_pieces:
        return None
        
    print(f"Solving for {len(valid_pieces)} pieces sequence...")
    
    # Find the best sequence
    best_score, best_seq = find_sequence_best_move(board, valid_pieces)
    
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
