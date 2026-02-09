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
    WINDOW_TITLE: str = "LDPlayer"  # Adjust if needed
    
    # Game region (will be calibrated)
    # Default values for 1080x1920 resolution
    GAME_REGION: GameRegion = GameRegion(x=0, y=0, width=1080, height=1920)
    
    # Grid configuration
    GRID_ROWS: int = 8
    GRID_COLS: int = 8
    
    # Grid boundaries (screen coordinates, to be calibrated)
    GRID_TOP_LEFT: Tuple[int, int] = (100, 400)  # (x, y)
    GRID_BOTTOM_RIGHT: Tuple[int, int] = (980, 1280)  # (x, y)
    
    # Computed cell dimensions
    @property
    def CELL_WIDTH(self) -> int:
        return (self.GRID_BOTTOM_RIGHT[0] - self.GRID_TOP_LEFT[0]) // self.GRID_COLS
    
    @property
    def CELL_HEIGHT(self) -> int:
        return (self.GRID_BOTTOM_RIGHT[1] - self.GRID_TOP_LEFT[1]) // self.GRID_ROWS
    
    # Piece slots (3 pieces at bottom, to be calibrated)
    # Each slot is (x, y, width, height)
    PIECE_SLOTS: List[GameRegion] = None
    
    def __post_init__(self):
        if self.PIECE_SLOTS is None:
            # Default piece slots for 1080x1920
            slot_y = 1400
            slot_width = 200
            slot_height = 200
            spacing = 100
            
            self.PIECE_SLOTS = [
                GameRegion(x=140, y=slot_y, width=slot_width, height=slot_height),
                GameRegion(x=440, y=slot_y, width=slot_width, height=slot_height),
                GameRegion(x=740, y=slot_y, width=slot_width, height=slot_height),
            ]
    
    # Mouse control
    MOUSE_DRAG_DURATION_MS: int = 300  # Duration of drag in milliseconds
    MOUSE_MOVE_RANDOMNESS: int = 3  # Pixels of randomness to add
    
    # Vision parameters
    FILLED_CELL_COLOR_HSV_LOWER: Tuple[int, int, int] = (0, 0, 100)  # To be tuned
    FILLED_CELL_COLOR_HSV_UPPER: Tuple[int, int, int] = (180, 255, 255)
    
    # Performance
    CAPTURE_FPS: int = 5  # Frames per second for screen capture
    SOLVER_TIME_BUDGET_MS: int = 50  # Max time for solver
    
    # Debug
    DEBUG: bool = False  # Enable debug visualization
    SAVE_DEBUG_FRAMES: bool = False  # Save frames to disk


# Global config instance
config = Config()
