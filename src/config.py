"""
Configuration for Block Blast automation.
Stores all calibration values and runtime parameters.
"""
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class GameRegion:
    """Defines a rectangular region on screen."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class Config:
    """Main configuration for the Block Blast bot."""
    
    # Window detection
    WINDOW_TITLE: str = "雷電模擬器"  # Supports Chinese localized LDPlayer
    
    # Game region (will be calibrated)
    # Default values for 1080x1920 resolution
    GAME_REGION: GameRegion = GameRegion(x=0, y=0, width=1080, height=1920)
    
    # Grid configuration
    GRID_ROWS: int = 8
    GRID_COLS: int = 8
    
    # Grid boundaries (screen coordinates, to be calibrated)
    GRID_TOP_LEFT: Tuple[int, int] = (35, 229)  # (x, y)
    GRID_BOTTOM_RIGHT: Tuple[int, int] = (586, 780)  # (x, y)
    
    # Computed cell dimensions
    @property
    def CELL_WIDTH(self) -> int:
        return (self.GRID_BOTTOM_RIGHT[0] - self.GRID_TOP_LEFT[0]) // self.GRID_COLS
    
    @property
    def CELL_HEIGHT(self) -> int:
        return (self.GRID_BOTTOM_RIGHT[1] - self.GRID_TOP_LEFT[1]) // self.GRID_ROWS
    
    # Tray configuration (Sampling pieces at bottom)
    TRAY_CELL_SIZE: Tuple[int, int] = (30, 30)  # Calibrated size of a single block in the tray
    TRAY_SLOT_CENTERS: List[Tuple[int, int]] = None
    PIECE_SLOTS: List[GameRegion] = None
    
    def __post_init__(self):
        if self.TRAY_SLOT_CENTERS is None:
            # Default centers for 1080x1920
            self.TRAY_SLOT_CENTERS = [(113, 942), (309, 941), (490, 942)]
            
        if self.PIECE_SLOTS is None:
            # Calibrated piece slots (Strict 5.0x bounds to avoid horizontal overlap)
            self.PIECE_SLOTS = []
            
            # Calculate min distance between centers for overlap prevention
            min_dist = 999
            if len(self.TRAY_SLOT_CENTERS) >= 2:
                for i in range(len(self.TRAY_SLOT_CENTERS) - 1):
                    dist = abs(self.TRAY_SLOT_CENTERS[i+1][0] - self.TRAY_SLOT_CENTERS[i][0])
                    min_dist = min(min_dist, dist)
            
            for (cx, cy) in self.TRAY_SLOT_CENTERS:
                # Enforce SQUARE slots: height = width
                # Use 5.0x multiplier, but cap at the distance between slots - 2px buffer
                size = int(self.TRAY_CELL_SIZE[0] * 5.0)
                if min_dist < size:
                    size = min_dist - 2
                
                self.PIECE_SLOTS.append(GameRegion(x=cx - size//2, y=cy - size//2, width=size, height=size))
    
    # Mouse control
    MOUSE_DRAG_DURATION_MS: int = 300  # Duration of drag in milliseconds
    MOUSE_MOVE_RANDOMNESS: int = 3  # Pixels of randomness to add
    
    # Vision parameters
    FILLED_CELL_COLOR_HSV_LOWER: Tuple[int, int, int] = (0, 0, 100)  # To be tuned
    FILLED_CELL_COLOR_HSV_UPPER: Tuple[int, int, int] = (180, 255, 255)
    
    # Drag Offsets (Phase 5 fix)
    # The piece is held ABOVE the cursor.
    # User reports offset increases as the piece moves UP the screen.
    DRAG_OFFSET_Y_BOTTOM: int = 150  # Offset at the bottom (piece slots)
    DRAG_OFFSET_Y_TOP: int = 300     # Offset at the top of the board
    
    # Vision Throttles (Aggressive Sensitivity)
    VISION_SAT_THRESHOLD: int = 70
    VISION_VAL_THRESHOLD: int = 50
    # Exclude Tray Blue-Gray Only if Saturation is low
    VISION_EXCLUDE_HUE_MIN: int = 80
    VISION_EXCLUDE_HUE_MAX: int = 150
    
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
