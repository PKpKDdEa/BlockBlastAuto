"""
Mouse controller for executing moves via drag-and-drop.
"""
import pyautogui
import time
import random
from typing import Tuple
from config import config
from model import Piece


# Configure pyautogui
pyautogui.PAUSE = 0.05  # Small pause between actions
pyautogui.FAILSAFE = True  # Move mouse to corner to abort


def cell_center(row: int, col: int) -> Tuple[int, int]:
    """
    Convert grid coordinates to screen coordinates (cell center).
    """
    grid_width = config.GRID_BOTTOM_RIGHT[0] - config.GRID_TOP_LEFT[0]
    grid_height = config.GRID_BOTTOM_RIGHT[1] - config.GRID_TOP_LEFT[1]
    cell_w = grid_width / config.GRID_COLS
    cell_h = grid_height / config.GRID_ROWS
    
    x = int(config.GRID_TOP_LEFT[0] + (col + 0.5) * cell_w)
    y = int(config.GRID_TOP_LEFT[1] + (row + 0.5) * cell_h)
    
    # Add small randomness if configured
    if config.MOUSE_MOVE_RANDOMNESS > 0:
        x += random.randint(-config.MOUSE_MOVE_RANDOMNESS, config.MOUSE_MOVE_RANDOMNESS)
        y += random.randint(-config.MOUSE_MOVE_RANDOMNESS, config.MOUSE_MOVE_RANDOMNESS)
    
    return (x, y)


def piece_slot_center(piece_index: int) -> Tuple[int, int]:
    """
    Get screen coordinates for piece slot center.
    
    Args:
        piece_index: Index of piece slot (0-2)
    
    Returns:
        Tuple of (x, y) screen coordinates
    """
    if piece_index < 0 or piece_index >= len(config.PIECE_SLOTS):
        raise ValueError(f"Invalid piece index: {piece_index}")
    
    slot = config.PIECE_SLOTS[piece_index]
    x = slot.x + slot.width // 2
    y = slot.y + slot.height // 2
    
    return (x, y)


def move_mouse_and_drag(start_xy: Tuple[int, int], end_xy: Tuple[int, int], duration_ms: int = None) -> None:
    """
    Move mouse and perform drag operation.
    
    Args:
        start_xy: Starting (x, y) coordinates
        end_xy: Ending (x, y) coordinates
        duration_ms: Duration of drag in milliseconds
    """
    if duration_ms is None:
        duration_ms = config.MOUSE_DRAG_DURATION_MS
    
    duration_sec = duration_ms / 1000.0
    
    # Move to start position
    pyautogui.moveTo(start_xy[0], start_xy[1], duration=0.2)
    pyautogui.mouseDown()
    time.sleep(0.1) # Wait for click to register
    
    # Perform drag
    pyautogui.moveTo(end_xy[0], end_xy[1], duration=duration_sec)
    time.sleep(0.1) # Wait for movement to finish
    pyautogui.mouseUp()
    
    time.sleep(0.1)


def drag_piece(piece: Piece, target_row: int, target_col: int) -> None:
    """
    Drag a piece from its slot to a target position on the board.
    Optimized for MuMu emulator with dynamic line-based offsets.
    """
    # v3.9: Use the ACTUAL detected coordinates in the tray for the starting grab
    slot = config.PIECE_SLOTS[piece.id]
    start_pos = (int(slot.x + piece.tray_cx), int(slot.y + piece.tray_cy))
    
    # Target line index (1 to 7)
    # Line 1 is between Row 0 and 1
    # Line 7 is between Row 6 and 7
    # For a piece starting at target_row, its center relative to lines:
    piece_center_row = target_row + (piece.height / 2.0)
    
    # Determine reference line (the line closest to or at the piece's vertical center)
    # Even rows (relative to lines): align center to line
    # Odd rows (relative to lines): align center to cell
    
    # v4.9 Non-Linear Displacement Logic (Piecewise)
    start_x, start_y = piece_slot_center(piece.id)
    
    # Target screen positions for the pieces internal center
    anchor_center_x, anchor_center_y = cell_center(target_row, target_col)
    anchor_dr, anchor_dc = piece.anchor_offset
    piece_target_x = anchor_center_x + int(anchor_dc * config.CELL_WIDTH)
    piece_target_y = anchor_center_y + int(anchor_dr * config.CELL_HEIGHT)
    
    # Distance in cells from starting slot center
    dx_cells = round(abs(piece_target_x - start_x) / config.CELL_WIDTH)
    dy_cells = round(abs(start_y - piece_target_y) / config.CELL_HEIGHT)
    
    # Lookup multipliers (capped at 8)
    mult_x = config.DISPLACEMENT_X_TABLE.get(min(8, dx_cells), 0.6)
    mult_y = config.DISPLACEMENT_Y_TABLE.get(min(8, dy_cells), 1.6)
    
    y_offset = int(mult_y * config.CELL_HEIGHT)
    x_pull = int(mult_x * config.CELL_WIDTH)
    
    dest_x = piece_target_x
    dest_y = piece_target_y + y_offset
    
    # v4.8.1 Piece Center / Line Alignment
    piece_center_row = target_row + anchor_dr + piece.height / 2.0
    ref_line = round(piece_center_row)
    ref_line = max(1, min(7, ref_line))
    
    if piece.height % 2 == 0:
        line_y = config.GRID_TOP_LEFT[1] + ref_line * config.CELL_HEIGHT
        dest_y = line_y + y_offset

    # Apply Directional X-Pull
    if piece_target_x > start_x: # Moving Right
        dest_x -= x_pull
    elif piece_target_x < start_x: # Moving Left
        dest_x += x_pull

    end_pos = (dest_x, dest_y)
    
    if config.DEBUG:
        print(f"Dragging piece {piece.id} to board ({target_row}, {target_col})")
        print(f"  Distance Cells: X={dx_cells}, Y={dy_cells}")
        print(f"  Multipliers: X={mult_x:.2f}, Y={mult_y:.2f}")
        print(f"  Offsets: Pull_X={x_pull}, Y={y_offset}")
        print(f"  Final Cursor Target: {end_pos}")
    
    move_mouse_and_drag(start_pos, end_pos)


def click_position(x: int, y: int) -> None:
    """
    Click at a specific screen position.
    
    Args:
        x: X coordinate
        y: Y coordinate
    """
    pyautogui.click(x, y)


if __name__ == "__main__":
    # Test controller (be careful - this will move your mouse!)
    print("Testing controller...")
    print("Move your mouse to the top-left corner to abort!")
    time.sleep(3)
    
    print("\nTesting cell_center calculations:")
    for row in [0, 3, 7]:
        for col in [0, 3, 7]:
            x, y = cell_center(row, col)
            print(f"Cell ({row}, {col}) -> Screen ({x}, {y})")
    
    print("\nTesting piece_slot_center:")
    for i in range(3):
        x, y = piece_slot_center(i)
        print(f"Piece slot {i} -> Screen ({x}, {y})")
    
    print("\nTest complete!")
