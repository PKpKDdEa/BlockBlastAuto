"""
Configuration for Block Blast automation.
Stores all calibration values and runtime parameters.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict


@dataclass
class GameRegion:
    """Defines a rectangular region on screen."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class Config:
    """Main configuration for the bot."""
    
    # Window detection
    WINDOW_TITLE: str = "Android Device"
    
    # Grid configuration
    GRID_ROWS: int = 8
    GRID_COLS: int = 8
    
    # Board region (screen coordinates, to be calibrated)
    GRID_TOP_LEFT: Tuple[int, int] = (41, 238)
    GRID_BOTTOM_RIGHT: Tuple[int, int] = (594, 791)
    
    # Computed cell dimensions
    @property
    def CELL_WIDTH(self) -> float:
        return (self.GRID_BOTTOM_RIGHT[0] - self.GRID_TOP_LEFT[0]) / float(self.GRID_COLS)
    
    @property
    def CELL_HEIGHT(self) -> float:
        return (self.GRID_BOTTOM_RIGHT[1] - self.GRID_TOP_LEFT[1]) / float(self.GRID_ROWS)
    
    # Piece tray parameters (v5.4: Revert to 34x34 based on MuMu standard scale)
    TRAY_CELL_SIZE: Tuple[int, int] = (34, 34)
    TRAY_SLOT_CENTERS: List[Tuple[int, int]] = None
    PIECE_SLOTS: List[GameRegion] = None
    
    def __post_init__(self):
        # v3.8: Attempt to load from calibration_config.txt automatically
        import os
        import ast
        calib_path = "calibration_config.txt"
        if os.path.exists(calib_path):
            try:
                with open(calib_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if '=' in line:
                            key, val = line.split('=', 1)
                            key = key.strip()
                            val = val.strip()
                            if key == "GRID_TOP_LEFT": self.GRID_TOP_LEFT = ast.literal_eval(val)
                            elif key == "GRID_BOTTOM_RIGHT": self.GRID_BOTTOM_RIGHT = ast.literal_eval(val)
                            elif key == "TRAY_CELL_SIZE": self.TRAY_CELL_SIZE = ast.literal_eval(val)
                            elif key == "TRAY_SLOT_CENTERS": self.TRAY_SLOT_CENTERS = ast.literal_eval(val)
                            elif key == "DRAG_OFFSET_X": self.DRAG_OFFSET_X = float(val)
                print(f"✓ Config: Auto-loaded calibration from {calib_path}")
            except Exception as e:
                print(f"! Config: Error loading {calib_path}: {e}")

        if self.TRAY_SLOT_CENTERS is None:
            # Default centers for 1080x1920
            self.TRAY_SLOT_CENTERS = [(134, 942), (320, 941), (506, 942)]
            
        if self.PIECE_SLOTS is None:
            # v5.2: Enlarge slots to 7.5x and remove height cap to fit 1x5 vertical pieces
            self.PIECE_SLOTS = []
            
            # Calculate min horizontal distance for overlap prevention
            min_dist_x = 999
            if len(self.TRAY_SLOT_CENTERS) >= 2:
                for i in range(len(self.TRAY_SLOT_CENTERS) - 1):
                    dist = abs(self.TRAY_SLOT_CENTERS[i+1][0] - self.TRAY_SLOT_CENTERS[i][0])
                    min_dist_x = min(min_dist_x, dist)
            
            for (cx, cy) in self.TRAY_SLOT_CENTERS:
                # v5.4: Restore 1:1 square slots with a safe 10px buffer
                size = min_dist_x - 10
                
                self.PIECE_SLOTS.append(GameRegion(x=cx - size//2, y=cy - size//2, width=size, height=size))
    
    # Mouse control
    MOUSE_DRAG_DURATION_MS: int = 300  # Duration of drag in milliseconds
    MOUSE_MOVE_RANDOMNESS: int = 3  # Pixels of randomness to add
    
    # Vision parameters
    FILLED_CELL_COLOR_HSV_LOWER: Tuple[int, int, int] = (0, 0, 100)  # To be tuned
    FILLED_CELL_COLOR_HSV_UPPER: Tuple[int, int, int] = (180, 255, 255)
    
    # Drag Offsets (Phase 5 fix)
    # The piece is held ABOVE the cursor.
    
    # v4.9 Nonlinear Displacement Tables (Multipliers for Cell Size)
    # Key: Distance in cells from slot to target
    DISPLACEMENT_X_TABLE: Dict[int, float] = field(default_factory=lambda: {
        0: 0, 1: 0.1, 2: 0.35, 3: 0.5, 4: 1.1, 5: 1.4, 6: 1.95, 7: 2.5, 8: 2.8
    })
    DISPLACEMENT_Y_TABLE: Dict[int, float] = field(default_factory=lambda: {
        1: 1, 2: 1.8, 3: 2.3, 4: 2.4, 5: 2.5, 6: 2.8, 7: 3.0, 8: 4.1, 9: 4.5
    })
    
    # Vision Throttles (Aggressive Sensitivity)
    VISION_SAT_THRESHOLD: int = 150
    VISION_VAL_THRESHOLD: int = 60
    # Exclude Tray Blue-Gray Only if Saturation is low
    # v3.2: Surgical MuMu background exclusion (wider range)
    VISION_EXCLUDE_HUE_MIN: int = 100
    VISION_EXCLUDE_HUE_MAX: int = 145
    
    # Debug & Control
    DEBUG: bool = True
    SAVE_DEBUG_FRAMES: bool = False
    HOTKEY_PAUSE: str = "f10"
    HOTKEY_AUTO_TOGGLE: str = "f11"
    AUTO_PLAY: bool = False # Start in observation mode by default
    
    # Heuristic Weights (Phase 3 improvements)
    WEIGHT_EMPTY_CELLS: float = 2.0
    WEIGHT_HOLES_PENALTY: float = -15.0
    WEIGHT_BUMPINESS: float = -0.5
    WEIGHT_NEAR_COMPLETE: float = 3.0
    WEIGHT_STREAK_BONUS: float = 20.0
    
    def save(self, path: str = "config.json"):
        """Save weights to JSON."""
        import json
        data = {
            "WEIGHT_EMPTY_CELLS": self.WEIGHT_EMPTY_CELLS,
            "WEIGHT_HOLES_PENALTY": self.WEIGHT_HOLES_PENALTY,
            "WEIGHT_BUMPINESS": self.WEIGHT_BUMPINESS,
            "WEIGHT_NEAR_COMPLETE": self.WEIGHT_NEAR_COMPLETE,
            "WEIGHT_STREAK_BONUS": self.WEIGHT_STREAK_BONUS
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load(self, path: str = "config.json"):
        """Load weights from JSON."""
        import json
        import os
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.WEIGHT_EMPTY_CELLS = data.get("WEIGHT_EMPTY_CELLS", self.WEIGHT_EMPTY_CELLS)
                self.WEIGHT_HOLES_PENALTY = data.get("WEIGHT_HOLES_PENALTY", self.WEIGHT_HOLES_PENALTY)
                self.WEIGHT_BUMPINESS = data.get("WEIGHT_BUMPINESS", self.WEIGHT_BUMPINESS)
                self.WEIGHT_NEAR_COMPLETE = data.get("WEIGHT_NEAR_COMPLETE", self.WEIGHT_NEAR_COMPLETE)
                self.WEIGHT_STREAK_BONUS = data.get("WEIGHT_STREAK_BONUS", self.WEIGHT_STREAK_BONUS)


# Global config instance
config = Config()
