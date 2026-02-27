"""
Diagnostic tool to verify the v2.1 Vision Upgrade (NMS + Validation).
Benchmarks TemplateManager against known board/piece states.
"""
import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))
from vision import template_manager, get_piece_grid
from config import config

def test_cv_accuracy():
    print("=" * 50)
    print("BlockBlastAuto v2.1 CV Benchmark")
    print("=" * 50)
    
    # Check templates
    print(f"Loaded {len(template_manager.templates)} templates across {len(template_manager.data)} categories.")
    
    # Mock a noisy grid (Overread Example)
    # 2x2 square with one noisy block outside
    noisy_grid = np.zeros((5, 5), dtype=np.uint8)
    noisy_grid[1:3, 1:3] = 1
    noisy_grid[4, 4] = 1 # Noise
    
    print("\n[Test 1] Noisy 2x2 Square (Overread Protection)")
    print("Raw Grid:\n", noisy_grid)
    
    grid, score, name, info = template_manager.match_and_validate(noisy_grid)
    print(f"Detected: {name} (Score: {score:.2f})")
    print("Validated Shape:", info["is_valid"])
    if name != "unknown":
        print("Final Snapped Grid:\n", grid)
    
    # Mock an illegal line (6 blocks)
    long_line = np.zeros((5, 5), dtype=np.uint8)
    long_line[2, :] = 1 # 5 blocks
    # (Actually 5 is the limit, let's try to mock something else)
    
    print("\n[Test 2] Exact Piece Matching")
    # Take a real template
    if template_manager.templates:
        sample = template_manager.templates[0]
        grid, score, name, info = template_manager.match_and_validate(sample)
        print(f"Sample Piece: {name} | Match Score: {score:.2f} | Valid: {info['is_valid']}")

if __name__ == "__main__":
    test_cv_accuracy()
