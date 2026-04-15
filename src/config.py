"""
Configuration for Block Blast automation.
Stores all calibration values and runtime parameters.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import os


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
        # v3.8: Attempt to load from calibration_config.txt automatically.
        # Use exec on the trusted local calibration file so multiline Python-like
        # values such as lists and GameRegion(...) entries load correctly.
        calib_path = "calibration_config.txt"
        if os.path.exists(calib_path):
            try:
                with open(calib_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                namespace = {"GameRegion": GameRegion}
                exec(content, namespace, namespace)

                for key in [
                    "GRID_TOP_LEFT",
                    "GRID_BOTTOM_RIGHT",
                    "TRAY_CELL_SIZE",
                    "TRAY_SLOT_CENTERS",
                    "PIECE_SLOTS",
                    "WINDOW_TITLE",
                    "ADB_PATH",
                    "ADB_DEVICE_ID",
                    "CONTROL_BACKEND",
                ]:
                    if key in namespace:
                        setattr(self, key, namespace[key])
                print(f"[OK] Config: Auto-loaded calibration from {calib_path}")
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

    # Input backend and drag model
    # CONTROL_BACKEND: "pyautogui" or "adb"
    CONTROL_BACKEND: str = "adb"
    # Coordinates from calibration tool are window-relative by default
    COORDINATE_SPACE: str = "window-relative"  # "window-relative" or "screen-absolute"
    # DRAG_MODEL: "ratio" (recommended) or "table" (legacy piecewise)
    DRAG_MODEL: str = "ratio"
    # Empirical cursor-to-piece movement ratio (conversation reference)
    DRAG_DISTANCE_RATIO: float = 0.725
    # Fixed vertical lift of piece above cursor during drag
    DRAG_LIFT_Y: int = 26

    # ADB control settings
    ADB_PATH: str = "adb"
    ADB_DEVICE_ID: str = ""  # optional, e.g. emulator-5554
    ADB_DRAG_DURATION_MS: int = 220
    
    # Vision parameters
    FILLED_CELL_COLOR_HSV_LOWER: Tuple[int, int, int] = (0, 0, 100)  # To be tuned
    FILLED_CELL_COLOR_HSV_UPPER: Tuple[int, int, int] = (180, 255, 255)
    
    # Drag Offsets (Phase 5 fix)
    # The piece is held ABOVE the cursor.
    
    # v4.9 Nonlinear Displacement Tables (Multipliers for Cell Size)
    # Key: Distance in cells from slot to target
    DISPLACEMENT_X_TABLE: Dict[int, float] = field(default_factory=lambda: {
        0: 0, 1: 0.1, 2: 0.55, 3: 0.6, 4: 1.3, 5: 1.5, 6: 1.85, 7: 2.5, 8: 2.8
    })
    DISPLACEMENT_Y_TABLE: Dict[int, float] = field(default_factory=lambda: {
        1: 1, 2: 1.8, 3: 2.2, 4: 2.4, 5: 2.6, 6: 3.0, 7: 3.1, 8: 4.0, 9: 4.4
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
    HOTKEY_AUTO_TOGGLE: str = "f5"
    AUTO_PLAY: bool = False # Start in observation mode by default
    
    # Heuristic Weights (Phase 3 improvements)
    WEIGHT_EMPTY_CELLS: float = 2.0
    WEIGHT_HOLES_PENALTY: float = -15.0
    WEIGHT_BUMPINESS: float = -0.5
    WEIGHT_NEAR_COMPLETE: float = 3.0
    WEIGHT_STREAK_BONUS: float = 20.0

    # Oracle integration
    # ORACLE_MODE: "internal" (current solver), "external_cmd", or "http"
    ORACLE_MODE: str = "external_cmd"
    ORACLE_COMMAND: str = ""
    ORACLE_URL: str = ""
    ORACLE_TIMEOUT_MS: int = 1500
    
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
