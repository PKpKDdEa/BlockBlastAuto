"""
Mouse controller for executing moves via drag-and-drop.
"""
import pyautogui
import time
import random
from typing import Tuple
from config import config


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
    Drag a piece from its slot to a target cell on the board.
    """
    start_pos = piece_slot_center(piece.id)
    top_left_cell_pos = cell_center(target_row, target_col)
    
    # Calculate scaling vertical offset
    y_top, y_bottom = config.GRID_TOP_LEFT[1], config.GRID_BOTTOM_RIGHT[1]
    progress = max(0, min(1, (y_bottom - top_left_cell_pos[1]) / (y_bottom - y_top)))
    current_offset_y = int(config.DRAG_OFFSET_Y_BOTTOM + progress * (config.DRAG_OFFSET_Y_TOP - config.DRAG_OFFSET_Y_BOTTOM))
    
    # Visual center offset in pixels
    anchor_dr, anchor_dc = piece.anchor_offset
    dest_x = top_left_cell_pos[0] + int(anchor_dc * config.CELL_WIDTH)
    dest_y = top_left_cell_pos[1] + int(anchor_dr * config.CELL_HEIGHT) + current_offset_y
    
    end_pos_offset = (dest_x, dest_y)
    
    if config.DEBUG:
        print(f"Dragging piece {piece.id} to board ({target_row}, {target_col})")
        print(f"  Target Cell: {top_left_cell_pos}, Offset: {current_offset_y}, Anchor: {piece.anchor_offset}")
        print(f"  Final Cursor Target: {end_pos_offset}")
    
    move_mouse_and_drag(start_pos, end_pos_offset)


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
