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
    start_pos = piece_slot_center(piece.id)
    
    # Target line index (1 to 7)
    # Line 1 is between Row 0 and 1
    # Line 7 is between Row 6 and 7
    # For a piece starting at target_row, its center relative to lines:
    piece_center_row = target_row + (piece.height / 2.0)
    
    # Determine reference line (the line closest to or at the piece's vertical center)
    # Even rows (relative to lines): align center to line
    # Odd rows (relative to lines): align center to cell
    
    ref_line = round(piece_center_row) 
    # Bounds check ref_line (1-7)
    ref_line = max(1, min(7, ref_line))
    
    # MuMu Offset Table
    # Line 1: 4.2, Line 2: 4.0, Line 3: 3.6, Line 4: 3.2, Line 5: 3.2, Line 6: 2.5, Line 7: 2.5
    offsets = {
        1: 4.2, 2: 4.0, 3: 3.6, 4: 3.2, 5: 3.2, 6: 2.5, 7: 2.5
    }
    multiplier = offsets.get(ref_line, 2.5)
    
    y_offset = multiplier * config.CELL_HEIGHT
    x_offset = piece.height * 0.4 * config.CELL_WIDTH
    
    # Screen position of the target cell (top-left of piece anchor)
    anchor_center_x, anchor_center_y = cell_center(target_row, target_col)
    
    # Adjust for piece size (anchor is at 0,0 in Piece.cells)
    # Visual center of the piece in grid units
    anchor_dr, anchor_dc = piece.anchor_offset
    
    dest_x = anchor_center_x + int(anchor_dc * config.CELL_WIDTH) + int(x_offset)
    dest_y = anchor_center_y + int(anchor_dr * config.CELL_HEIGHT) + int(y_offset)
    
    # Alignment Rule: 
    # "place the object block with even number row's center on the line 
    #  and odd number row's center on the target cell"
    # Even number of rows -> center is on a line
    # Odd number of rows -> center is in a cell center
    
    if piece.height % 2 == 0:
        # Align to nearest line
        line_y = config.GRID_TOP_LEFT[1] + ref_line * config.CELL_HEIGHT
        # Note: cell_center already adds 0.5 * cell_h, so anchor_center_y is at row + 0.5
        # We need to shift to the line
        dest_y = line_y + int(y_offset)
    else:
        # Align to cell center (already handled by anchor_center_y + anchor_dr)
        pass

    end_pos = (dest_x, dest_y)
    
    if config.DEBUG:
        print(f"Dragging piece {piece.id} to board ({target_row}, {target_col})")
        print(f"  Line: {ref_line}, Multiplier: {multiplier}, Rows: {piece.height}")
        print(f"  Offsets: X={int(x_offset)}, Y={int(y_offset)}")
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
