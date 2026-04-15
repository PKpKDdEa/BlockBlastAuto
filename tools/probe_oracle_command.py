"""
Probe an external oracle runtime command and inspect its raw output.

Usage examples:
  python tools/probe_oracle_command.py --command "python my_oracle.py"
  python tools/probe_oracle_command.py --command "node oracle.js" --payload payload.json

The command is expected to read JSON from stdin and write JSON to stdout.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def build_sample_payload() -> dict:
    return {
        "rows": 8,
        "cols": 8,
        "board": [[0 for _ in range(8)] for _ in range(8)],
        "pieces": [
            {"piece_index": 0, "cells": [[0, 0]], "width": 1, "height": 1},
            {"piece_index": 1, "cells": [[0, 0], [0, 1]], "width": 2, "height": 1},
            {"piece_index": 2, "cells": [[0, 0], [1, 0]], "width": 1, "height": 2},
        ],
    }


def parse_move_candidate(text: str):
    """Try to parse known move formats for quick diagnostics."""
    try:
        data = json.loads(text)
    except Exception:
        return None, "stdout is not valid JSON"

    if isinstance(data, dict):
        move = data.get("move", data)
        if isinstance(move, dict) and all(k in move for k in ("row", "col", "piece_index")):
            try:
                parsed = {
                    "row": int(move["row"]),
                    "col": int(move["col"]),
                    "piece_index": int(move["piece_index"]),
                }
                return parsed, None
            except Exception:
                return None, "row/col/piece_index exist but not int-castable"

    return None, "JSON parsed, but move format not recognized"


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe external oracle command")
    parser.add_argument("--command", required=True, help="Command to run (quoted)")
    parser.add_argument(
        "--payload",
        default="",
        help="Optional path to payload JSON file. If omitted, a sample payload is used.",
    )
    parser.add_argument("--timeout-ms", type=int, default=1500)
    args = parser.parse_args()

    if args.payload:
        payload_path = Path(args.payload)
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    else:
        payload = build_sample_payload()

    payload_text = json.dumps(payload)
    cmd = shlex.split(args.command)

    print("=" * 60)
    print("Oracle Command Probe")
    print("=" * 60)
    print(f"Command: {args.command}")
    print(f"Timeout: {args.timeout_ms} ms")
    print("\nPayload preview:")
    print(payload_text[:500] + ("..." if len(payload_text) > 500 else ""))

    try:
        proc = subprocess.run(
            cmd,
            input=payload_text,
            capture_output=True,
            text=True,
            timeout=max(0.2, args.timeout_ms / 1000.0),
        )
    except Exception as e:
        print(f"\nExecution failed: {e}")
        return 2

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    print("\n--- Process Result ---")
    print(f"Return code: {proc.returncode}")
    print(f"STDOUT length: {len(stdout)}")
    print(f"STDERR length: {len(stderr)}")

    print("\n--- STDOUT ---")
    print(stdout if stdout else "<empty>")

    if stderr:
        print("\n--- STDERR ---")
        print(stderr)

    parsed_move, parse_error = parse_move_candidate(stdout)
    print("\n--- Parse Attempt ---")
    if parsed_move is not None:
        print("Detected move:", json.dumps(parsed_move))
        print("You can set ORACLE_COMMAND to exactly this command string.")
        print("Current adapter can parse this output format.")
        return 0

    print(f"Could not parse move: {parse_error}")
    print("Next step: share raw STDOUT so adapter parser can be extended.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
