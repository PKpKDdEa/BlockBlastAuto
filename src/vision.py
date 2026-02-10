"""
Computer vision module for detecting board state and pieces.
"""
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from config import config
from model import Board, Piece


def read_board(frame: np.ndarray) -> Board:
    """
    Detect board state from game screenshot.
    
    Args:
        frame: BGR image of game window
    
    Returns:
        Board object with detected state
    """
    board = Board(config.GRID_ROWS, config.GRID_COLS)
    
    # Extract grid region
    x1, y1 = config.GRID_TOP_LEFT
    x2, y2 = config.GRID_BOTTOM_RIGHT
    grid_region = frame[y1:y2, x1:x2]
    
    # Process each cell
    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            # Calculate cell center
            cell_x = col * config.CELL_WIDTH + config.CELL_WIDTH // 2
            cell_y = row * config.CELL_HEIGHT + config.CELL_HEIGHT // 2
            
            # Extract small patch around center
            patch_size = min(config.CELL_WIDTH, config.CELL_HEIGHT) // 3
            y_start = max(0, cell_y - patch_size // 2)
            y_end = min(grid_region.shape[0], cell_y + patch_size // 2)
            x_start = max(0, cell_x - patch_size // 2)
            x_end = min(grid_region.shape[1], cell_x + patch_size // 2)
            
            patch = grid_region[y_start:y_end, x_start:x_end]
            
            is_filled = classify_cell(patch)
            board.grid[row, col] = 1 if is_filled else 0
            
    if config.DEBUG:
        filled_count = np.sum(board.grid == 1)
        print(f"Board detected: {filled_count}/64 cells filled")
    
    return board


def classify_cell(patch: np.ndarray) -> bool:
    """
    Classify a cell patch as filled or empty.
    
    Uses color-based detection. Filled cells typically have vibrant colors,
    while empty cells are darker/grayer.
    
    Args:
        patch: Small BGR image patch from cell center
    
    Returns:
        True if cell is filled, False if empty
    """
    if patch.size == 0:
        return False
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    
    # Calculate average brightness (V channel)
    avg_brightness = np.mean(hsv[:, :, 2])
    
    # Calculate saturation (S channel)
    avg_saturation = np.mean(hsv[:, :, 1])
    
    # Filled cells are typically brighter and more saturated
    # These thresholds may need tuning based on actual game appearance
    is_filled = avg_brightness > 80 and avg_saturation > 30
    
    return is_filled


def read_pieces(frame: np.ndarray) -> List[Piece]:
    """
    Detect available pieces from piece slots.
    
    Args:
        frame: BGR image of game window
    
    Returns:
        List of detected pieces (up to 3)
    """
    pieces = []
    
    for slot_idx, slot in enumerate(config.PIECE_SLOTS):
        # Extract piece slot region
        piece_region = frame[slot.y:slot.y+slot.height, slot.x:slot.x+slot.width]
        
        # Detect piece mask
        mask = detect_piece_mask(piece_region)
        
        if mask is not None and np.sum(mask) > 0:
            # Create piece from mask
            piece = Piece.from_mask(piece_id=slot_idx, mask=mask)
            pieces.append(piece)
            if config.DEBUG:
                print(f"Piece {slot_idx} detected: {piece.width}x{piece.height}")
        else:
            # Empty slot
            pieces.append(None)
    
    return pieces


def detect_piece_mask(piece_region: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect binary mask of piece shape from piece slot image.
    
    Args:
        piece_region: BGR image of piece slot
    
    Returns:
        Binary mask (2D array) where 1 = piece cell, 0 = empty
        Returns None if no piece detected
    """
    if piece_region.size == 0:
        return None
    
    # Convert to HSV
    hsv = cv2.cvtColor(piece_region, cv2.COLOR_BGR2HSV)
    
    # Inset the region slightly to avoid capturing slot borders
    inset = 10
    hsv_subset = hsv[inset:-inset, inset:-inset]
    
    # Create mask for colored regions (piece cells)
    # We use two ranges to EXCLUDE the tray blue (approx 100-135)
    lower1 = np.array([0, config.VISION_SAT_THRESHOLD, config.VISION_VAL_THRESHOLD])
    upper1 = np.array([config.VISION_EXCLUDE_HUE_MIN, 255, 255])
    
    lower2 = np.array([config.VISION_EXCLUDE_HUE_MAX, config.VISION_SAT_THRESHOLD, config.VISION_VAL_THRESHOLD])
    upper2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_subset, lower1, upper1)
    mask2 = cv2.inRange(hsv_subset, lower2, upper2)
    mask_small = cv2.bitwise_or(mask1, mask2)
    
    # Reconstruct full-size mask with black border
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    mask[inset:-inset, inset:-inset] = mask_small
    
    if config.SAVE_DEBUG_FRAMES:
        import os
        os.makedirs("debug", exist_ok=True)
        cv2.imwrite(f"debug/piece_region_{int(time.time()*1000)}.png", piece_region)
        cv2.imwrite(f"debug/mask_{int(time.time()*1000)}.png", mask)
    
    # Morphological clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill gaps
    
    # Use contours to isolate the piece and ignore slot borders/noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Find bounding box of the piece (all contours combined)
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Extra safety: crop must be at least a few pixels
    if w < 5 or h < 5:
        return None
        
    mask_cropped = mask[y:y+h, x:x+w]
    
    # Estimate grid dimensions based on pixel size relative to slot size
    # In Block Blast, a 1-unit block is roughly 1/5 of the tray slot dimension.
    slot_w = piece_region.shape[1]
    cell_size = slot_w / 5.0
    
    cols = max(1, int(round(w / cell_size)))
    rows = max(1, int(round(h / cell_size)))
    
    # Cap to 5x5
    cols = min(5, cols)
    rows = min(5, rows)
    
    if config.DEBUG:
        print(f"  Slot Region: {piece_region.shape}, Mask BBox: ({x},{y},{w},{h}), Inferred: {rows}x{cols}")
    
    # Resize to the inferred grid dimensions
    mask_resized = cv2.resize(mask_cropped, (cols, rows), interpolation=cv2.INTER_NEAREST)
    
    # Threshold to binary
    mask_binary = (mask_resized > 127).astype(np.uint8)
    
    return mask_binary


def load_piece_templates() -> dict:
    """
    Load pre-labeled piece templates.
    
    Returns:
        Dictionary mapping piece_name -> binary mask
    """
    # TODO: Implement template loading from files
    # For now, return empty dict
    # In future, this would load from templates/ directory
    return {}


def is_board_animating(frame1: np.ndarray, frame2: np.ndarray) -> bool:
    """
    Check if the board is currently animating by comparing two frames.
    """
    x1, y1 = config.GRID_TOP_LEFT
    x2, y2 = config.GRID_BOTTOM_RIGHT
    
    grid1 = frame1[y1:y2, x1:x2]
    grid2 = frame2[y1:y2, x1:x2]
    
    # Calculate absolute difference
    diff = cv2.absdiff(grid1, grid2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Blur to ignore minor noise
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    
    # Count changed pixels
    changed_pixels = np.sum(gray_diff > 30)
    
    # Threshold for animation (tuned for 1080p)
    return changed_pixels > 500


def wait_for_board_stable(capture, timeout_s: float = 3.0):
    """
    Wait until the board stops changing (animations finished).
    """
    start_time = time.time()
    last_frame = capture.capture_frame()
    if last_frame is None:
        return
        
    stable_count = 0
    required_stable = 2  # consecutive stable frames
    
    while time.time() - start_time < timeout_s:
        time.sleep(0.1)
        current_frame = capture.capture_frame()
        if current_frame is None:
            continue
            
        if not is_board_animating(last_frame, current_frame):
            stable_count += 1
            if stable_count >= required_stable:
                return True
        else:
            stable_count = 0
            
        last_frame = current_frame
        
    return False


def visualize_detection(frame: np.ndarray, board: Board, pieces: List[Piece]) -> np.ndarray:
    """
    Create visualization of detected board and pieces.
    
    Args:
        frame: Original BGR image
        board: Detected board state
        pieces: Detected pieces
    
    Returns:
        Annotated image
    """
    vis = frame.copy()
    
    # Draw grid
    x1, y1 = config.GRID_TOP_LEFT
    x2, y2 = config.GRID_BOTTOM_RIGHT
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw grid cells
    for row in range(config.GRID_ROWS + 1):
        y = y1 + row * config.CELL_HEIGHT
        cv2.line(vis, (x1, y), (x2, y), (0, 255, 0), 1)
    
    for col in range(config.GRID_COLS + 1):
        x = x1 + col * config.CELL_WIDTH
        cv2.line(vis, (x, y1), (x, y2), (0, 255, 0), 1)
    
    # Mark filled cells
    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            if board.grid[row, col] == 1:
                x = x1 + col * config.CELL_WIDTH + config.CELL_WIDTH // 2
                y = y1 + row * config.CELL_HEIGHT + config.CELL_HEIGHT // 2
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
    
    # Draw piece slots
    for slot in config.PIECE_SLOTS:
        cv2.rectangle(vis, (slot.x, slot.y), (slot.x + slot.width, slot.y + slot.height), (255, 0, 0), 2)
    
    return vis


def visualize_drag(frame: np.ndarray, move: Move, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> np.ndarray:
    """
    Visualize the drag move with red dots for start/end.
    """
    vis = frame.copy()
    # Draw Piece Start (Red Circle)
    cv2.circle(vis, start_pos, 10, (0, 0, 255), -1)
    # Draw Piece Actual Target Board Center (Yellow Circle)
    # Note: Move doesn't store target screen pos directly, we recalculated it
    # But for visualization, let's just use the end_pos (the one with offset)
    cv2.circle(vis, end_pos, 20, (0, 0, 255), 3) # Big Red Circle (where mouse goes)
    
    # Draw an X at the theoretical center of the cell (no offset)
    # This helps see the distance of the offset
    target_no_offset = (end_pos[0], end_pos[1] - config.DRAG_OFFSET_Y_BOTTOM) # Approximate
    cv2.drawMarker(vis, target_no_offset, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    
    return vis
    
    print("Testing vision module...")
    print("Make sure LDPlayer with Block Blast is running!")
    
    capture = WindowCapture()
    if capture.find_window():
        frame = capture.capture_frame()
        if frame is not None:
            print("Captured frame, detecting board...")
            
            board = read_board(frame)
            print("\nDetected board:")
            print(board)
            
            pieces = read_pieces(frame)
            print(f"\nDetected {len(pieces)} pieces")
            
            # Visualize
            vis = visualize_detection(frame, board, pieces)
            cv2.imshow("Detection Visualization", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Failed to find window!")
