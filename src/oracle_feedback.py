"""
Oracle Feedback Bridge for Block Blast Automation.
Compares the bot's moves against an expert reference (KevinGu/BlockBlastSolver).
"""
import json
import os
import time
import subprocess
import shlex
from urllib import request as urlrequest
from typing import Dict, List, Optional
from model import Board, Piece
from config import config

class OracleFeedback:
    def __init__(self, replay_path: str = "game_replays.json"):
        self.replay_path = replay_path
        self.replays = self.load_replays()
        self.oracle_enabled = False

    def _build_payload(self, board: Board, pieces: List[Piece]) -> Dict:
        return {
            "board": board.grid.tolist(),
            "pieces": [
                None if p is None else {
                    "piece_index": p.id,
                    "cells": [list(cell) for cell in p.cells],
                    "width": p.width,
                    "height": p.height,
                }
                for p in pieces
            ],
            "rows": board.rows,
            "cols": board.cols,
            "timestamp": time.time(),
        }

    def _parse_move(self, data: Dict) -> Optional[Dict]:
        """Normalize external response to {row, col, piece_index}."""
        if not isinstance(data, dict):
            return None

        if "move" in data and isinstance(data["move"], dict):
            data = data["move"]

        required = {"row", "col", "piece_index"}
        if not required.issubset(set(data.keys())):
            return None

        return {
            "row": int(data["row"]),
            "col": int(data["col"]),
            "piece_index": int(data["piece_index"]),
        }

    def _query_external_cmd(self, payload: Dict) -> Optional[Dict]:
        if not config.ORACLE_COMMAND.strip():
            return None

        cmd = shlex.split(config.ORACLE_COMMAND)
        timeout_s = max(0.2, config.ORACLE_TIMEOUT_MS / 1000.0)
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if proc.returncode != 0:
                if config.DEBUG:
                    print(f"[ORACLE] external_cmd failed ({proc.returncode}): {proc.stderr.strip()}")
                return None
            out = (proc.stdout or "").strip()
            if not out:
                return None
            data = json.loads(out)
            return self._parse_move(data)
        except Exception as e:
            if config.DEBUG:
                print(f"[ORACLE] external_cmd exception: {e}")
            return None

    def _query_http(self, payload: Dict) -> Optional[Dict]:
        if not config.ORACLE_URL.strip():
            return None

        timeout_s = max(0.2, config.ORACLE_TIMEOUT_MS / 1000.0)
        try:
            req = urlrequest.Request(
                config.ORACLE_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            return self._parse_move(data)
        except Exception as e:
            if config.DEBUG:
                print(f"[ORACLE] http exception: {e}")
            return None

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
        Oracle move source (config.ORACLE_MODE):
        - internal: use local best_move (baseline)
        - external_cmd: execute local command, JSON in stdin / JSON out stdout
        - http: POST JSON payload to external service
        """
        payload = self._build_payload(board, pieces)

        if config.ORACLE_MODE == "external_cmd":
            move = self._query_external_cmd(payload)
            if move is not None:
                return move
        elif config.ORACLE_MODE == "http":
            move = self._query_http(payload)
            if move is not None:
                return move

        # Fallback baseline
        from solver import best_move
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
