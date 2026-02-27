"""
Oracle Feedback Bridge for Block Blast Automation.
Compares the bot's moves against an expert reference (KevinGu/BlockBlastSolver).
"""
import json
import os
import time
from typing import Dict, List, Optional
from model import Board, Piece

class OracleFeedback:
    def __init__(self, replay_path: str = "game_replays.json"):
        self.replay_path = replay_path
        self.replays = self.load_replays()
        self.oracle_enabled = False

    def load_replays(self) -> List[Dict]:
        if os.path.exists(self.replay_path):
            try:
                with open(self.replay_path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def record_state(self, board: Board, pieces: List[Piece], my_move: Optional[Dict], oracle_move: Optional[Dict], score: int):
        """Record a game snapshot for comparison and learning."""
        entry = {
            "timestamp": time.time(),
            "board_grid": board.grid.tolist(),
            "pieces": [p.id if p else None for p in pieces],
            "my_score": score,
            "my_move": my_move,
            "oracle_move": oracle_move,
            "match_ratio": 1.0 if my_move == oracle_move else 0.0
        }
        self.replays.append(entry)
        self.save_replays()

    def save_replays(self):
        with open(self.replay_path, 'w') as f:
            json.dump(self.replays, f, indent=2)

    def get_oracle_comparison(self, board: Board, pieces: List[Piece]) -> Optional[Dict]:
        """
        [PLACEHOLDER] - In a real setup, this would call web-based solver or a local binary.
        For now, it returns the current best move from YOUR solver to establish baseline.
        """
        from solver import best_move
        # This acts as a 'performance mirror' until we have KevinGu API access
        move = best_move(board, pieces)
        if move:
            return {"row": move.row, "col": move.col, "piece_index": move.piece_index}
        return None

    def calculate_performance_ratio(self) -> float:
        """
        Calculate how well we match the Oracle over the last batch of moves.
        Ratio = (My Total Score Improvement) / (Oracle Predicted Score Improvement)
        """
        if not self.replays:
            return 1.0
            
        recent = self.replays[-20:] # Last 20 moves
        matches = sum(1 for r in recent if r["match_ratio"] == 1.0)
        return matches / float(len(recent)) if recent else 1.0
