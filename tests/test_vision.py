"""
Unit tests for vision template matching.
"""
import json
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from vision import TemplateManager


def test_all_canonical_templates_match_themselves():
    manager = TemplateManager()
    data = json.loads(Path("data/templates.json").read_text(encoding="utf-8"))

    for category, pieces in data.items():
        for name, grid_list in pieces.items():
            grid = np.array(grid_list, dtype=np.uint8)
            rows = int(np.any(grid, axis=1).sum())
            cols = int(np.any(grid, axis=0).sum())
            matched, score, matched_name, info = manager.match_and_validate(
                grid,
                expected_dims=(rows, cols),
            )
            assert matched_name == f"{category}/{name}"
            assert np.array_equal(matched, grid)
            assert score == 1.0


def test_impossible_pattern_snaps_to_nearest_valid_template():
    manager = TemplateManager()
    raw = np.zeros((5, 5), dtype=np.uint8)
    raw[1, 1:4] = 1
    raw[2:4, 1] = 1
    raw[2, 2] = 1

    matched, score, matched_name, info = manager.match_and_validate(raw, expected_dims=(3, 3))

    assert matched_name != "unknown"
    assert "/" in matched_name
    assert int(np.sum(matched)) in {1, 2, 3, 4, 5, 6, 9}
    assert score >= 0.35
