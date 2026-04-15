"""
oracle_adapter.py  –  stdin→stdout JSON adapter for the external_cmd oracle mode.

Protocol
--------
  stdin:   JSON payload as produced by OracleFeedback._build_payload()
           {
               "board":  [[int, ...], ...],   # 8×8 grid, 1=filled 0=empty
               "pieces": [ null | {
                   "piece_index": int,
                   "cells": [[row, col], ...],
                   "width":  int,
                   "height": int,
               }, ... ],
               "rows": int,   # optional, default 8
               "cols": int,   # optional, default 8
           }

  stdout:  JSON move
           {"row": int, "col": int, "piece_index": int}

  exit 0  on success, non-zero on error (stderr carries the message).

Usage
-----
  Set in config.py / calibration_config.txt:
      ORACLE_MODE    = external_cmd
      ORACLE_COMMAND = python tools/oracle_adapter.py

  Or with an explicit venv interpreter on Windows:
      ORACLE_COMMAND = venv\Scripts\python.exe tools\oracle_adapter.py

Replacing the solver
--------------------
  To plug in any other solver, edit _solve() below and return a dict like
      {"row": int, "col": int, "piece_index": int}
  The rest of the wiring stays the same.
"""

import json
import sys
import os

# ── resolve src/ whether the script is run from the project root or from tools/ ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.normpath(_SRC))

# ─────────────────────────────────────────────────────────────────────────────
# Internal imports (only loaded once the path is set up)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from typing import Optional
from model import Board, Piece
from solver import best_move


# ─────────────────────────────────────────────────────────────────────────────
# Payload → domain objects
# ─────────────────────────────────────────────────────────────────────────────

def _board_from_payload(payload: dict) -> Board:
    rows = int(payload.get("rows", 8))
    cols = int(payload.get("cols", 8))
    board = Board(rows, cols)
    raw = payload.get("board", [])
    arr = np.array(raw, dtype=np.int8)
    if arr.shape == (rows, cols):
        board.grid[:] = arr
    board.sync_bitboard_from_grid()
    return board


def _pieces_from_payload(payload: dict):
    result = []
    for p in payload.get("pieces", []):
        if p is None:
            result.append(None)
            continue
        cells = [tuple(c) for c in p.get("cells", [])]
        piece = Piece(
            id=int(p.get("piece_index", 0)),
            cells=cells,
            width=int(p.get("width", 1)),
            height=int(p.get("height", 1)),
        )
        result.append(piece)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core solve – replace this function to plug in a different engine
# ─────────────────────────────────────────────────────────────────────────────

def _solve(board: Board, pieces) -> Optional[dict]:
    """
    Default implementation: local greedy DFS solver (best_move).

    Returns {"row": int, "col": int, "piece_index": int} or None if no move.

    ─── Custom adapter hook ────────────────────────────────────────────────────
    To use an external engine instead, replace the body of this function, e.g.:
        import subprocess, json
        payload = build_your_format(board, pieces)
        out = subprocess.check_output(["my_engine"], input=json.dumps(payload), text=True)
        return json.loads(out)
    ────────────────────────────────────────────────────────────────────────────
    """
    move = best_move(board, pieces)
    if move is None:
        return None
    return {"row": int(move.row), "col": int(move.col), "piece_index": int(move.piece_index)}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.stderr.write("[oracle_adapter] empty stdin\n")
        return 1

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"[oracle_adapter] JSON parse error: {exc}\n")
        return 1

    try:
        board  = _board_from_payload(payload)
        pieces = _pieces_from_payload(payload)
    except Exception as exc:
        sys.stderr.write(f"[oracle_adapter] payload reconstruction error: {exc}\n")
        return 1

    try:
        result = _solve(board, pieces)
    except Exception as exc:
        sys.stderr.write(f"[oracle_adapter] solver error: {exc}\n")
        return 1

    if result is None:
        sys.stderr.write("[oracle_adapter] solver returned no move\n")
        return 1

    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
