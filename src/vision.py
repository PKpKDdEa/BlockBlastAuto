"""
Computer vision module for detecting board state and pieces.
"""
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from config import config
from model import Board, Piece, Move


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


def get_piece_vibrancy_mask(hsv_img: np.ndarray) -> np.ndarray:
    """
    Unified vibrancy-aware mask for pieces. 
    Handles Stage 1 (High Vibrancy) and Stage 2 (Color Selective).
    """
    # Stage 1: Absolute Vibrancy (Catches any block with high saturation)
    # Background tray is usually saturation < 120. Pieces are high.
    lower_vibrant = np.array([0, 150, 80])
    upper_vibrant = np.array([180, 255, 255])
    mask_vibrant = cv2.inRange(hsv_img, lower_vibrant, upper_vibrant)
    
    # Stage 2: Color Selective (Catches Red, Green, etc. while avoiding Tray Blue)
    lower_r1 = np.array([0, config.VISION_SAT_THRESHOLD, config.VISION_VAL_THRESHOLD])
    upper_r1 = np.array([config.VISION_EXCLUDE_HUE_MIN, 255, 255])
    
    lower_r2 = np.array([config.VISION_EXCLUDE_HUE_MAX, config.VISION_SAT_THRESHOLD, config.VISION_VAL_THRESHOLD])
    upper_r2 = np.array([180, 255, 255])
    
    mask_r1 = cv2.inRange(hsv_img, lower_r1, upper_r1)
    mask_r2 = cv2.inRange(hsv_img, lower_r2, upper_r2)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_vibrant, mask_r1)
    mask = cv2.bitwise_or(mask, mask_r2)
    
    # Cleanup noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Dilate slightly to bridge beveled edges
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def read_pieces(frame: np.ndarray) -> List[Piece]:
    """
    Detect available pieces from piece slots.
    """
    pieces = []
    
    for slot_idx, slot in enumerate(config.PIECE_SLOTS):
        piece_region = frame[slot.y:slot.y+slot.height, slot.x:slot.x+slot.width]
        mask = detect_piece_mask(piece_region)
        
        if mask is not None and np.sum(mask) > 0:
            piece = Piece.from_mask(piece_id=slot_idx, mask=mask)
            pieces.append(piece)
            if config.DEBUG:
                print(f"Piece {slot_idx} detected: {piece.width}x{piece.height}")
        else:
            pieces.append(None)
    
    return pieces


def detect_piece_mask(piece_region: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the piece shape by finding its bounding box and sampling a relative grid.
    This handles cases where the game auto-centers pieces in the tray.
    """
    if piece_region.size == 0:
        return None
    
    # HSV for vibrancy check
    hsv = cv2.cvtColor(piece_region, cv2.COLOR_BGR2HSV)
    
    # Use unified mask logic
    mask = get_piece_vibrancy_mask(hsv)
    
    # Find bounding box of the piece (all significant contours combined)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    piece_points = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 40: # Threshold for a unit block segment
            piece_points.append(cnt)
            
    if not piece_points:
        return None
        
    all_points = np.concatenate(piece_points)
    bx, by, bw, bh = cv2.boundingRect(all_points)
    
    # Infer grid dimensions
    # A single unit block is always approximately 40-42 pixels in a standardized 720/1080p tray
    # Better to infer from the slot_w which we know is calibrated to hold 5 units.
    # If slot_w is 250, unit_size = slot_w / 5.0 = 50. 
    # BUT the pieces aren't actually 250px wide. 
    # Let's use a more robust way: unit_size is derived from the fact that 5 units = ~200px.
    # Since we expanded the slot to 250, let's just use slot_w / 5.0 but clamp it or use a constant.
    unit_size = 40.0 # Standard block size in tray
    
    cols = int(round(bw / unit_size))
    rows = int(round(bh / unit_size))
    
    # Clamp 1-5
    cols = max(1, min(5, cols))
    rows = max(1, min(5, rows))
    
    # Now sample a grid WITHIN the bounding box (bx, by, bw, bh)
    grid = np.zeros((rows, cols), dtype=np.uint8)
    
    sub_w = bw / float(cols)
    sub_h = bh / float(rows)
    
    for r in range(rows):
        for c in range(cols):
            # Calculate sub-cell boundaries relative to piece_region
            y1 = by + int(r * sub_h)
            y2 = by + int((r + 1) * sub_h)
            x1 = bx + int(c * sub_w)
            x2 = bx + int((c + 1) * sub_w)
            
            # Sample center patch of this sub-cell
            margin_h = max(1, int((y2 - y1) * 0.2))
            margin_w = max(1, int((x2 - x1) * 0.2))
            
            # Use the same two-stage logic to decide if individual cell is filled
            hsv_patch = hsv[y1+margin_h:y2-margin_h, x1+margin_w:x2-margin_w]
            if hsv_patch.size > 0:
                avg_h = np.mean(hsv_patch[:, :, 0])
                avg_s = np.mean(hsv_patch[:, :, 1])
                avg_v = np.mean(hsv_patch[:, :, 2])
                
                is_filled = False
                if avg_s > 150 and avg_v > 80: # Absolute vibrancy pass (Stage 1)
                    is_filled = True
                elif (avg_h < config.VISION_EXCLUDE_HUE_MIN or avg_h > config.VISION_EXCLUDE_HUE_MAX) and \
                     (avg_s > config.VISION_SAT_THRESHOLD and avg_v > config.VISION_VAL_THRESHOLD):
                    is_filled = True
                
                if is_filled:
                    grid[r, c] = 1
                elif config.DEBUG:
                    print(f"    Block at ({r},{c}) failed: Hue={avg_h:.0f}, Sat={avg_s:.1f}, Val={avg_v:.1f}")

    # Create a standardized 5x5 grid and center the detected piece within it
    grid_5x5 = np.zeros((5, 5), dtype=np.uint8)
    
    # Calculate start positions to center the (rows x cols) piece in (5 x 5)
    # This follows standard Block Blast centering (Width 1=index 2, Width 2=index 1)
    start_row = (5 - rows) // 2
    start_col = (5 - cols) // 2
    
    # Slice the sampled piece into the 5x5 grid
    grid_5x5[start_row : start_row + rows, start_col : start_col + cols] = grid
    
    if config.DEBUG:
        print(f"  Piece BBox: ({bx},{by},{bw},{bh}), Grid Detected: {rows}x{cols} (Centered in 5x5)")
        for row in grid_5x5:
            print("  " + "".join(["#" if x else "." for x in row]))
            
    return grid_5x5


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
    
    # Draw piece slots and their internal relative grids
    for slot_idx, slot in enumerate(config.PIECE_SLOTS):
        cv2.rectangle(vis, (slot.x, slot.y), (slot.x + slot.width, slot.y + slot.height), (255, 0, 0), 1)
        
        # Determine the piece's actual boundaries for overlay
        piece_region = frame[slot.y:slot.y+slot.height, slot.x:slot.x+slot.width]
        if piece_region.size == 0: continue
        hsv = cv2.cvtColor(piece_region, cv2.COLOR_BGR2HSV)
        
        # Use unified mask
        mask = get_piece_vibrancy_mask(hsv)
        # Bridge gaps
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        
        piece_points = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 40:
                piece_points.append(cnt)
        
        if not piece_points: continue
        
        all_points = np.concatenate(piece_points)
        bx, by, bw, bh = cv2.boundingRect(all_points)
        
        # Infer dimensions for visualization (same as detection logic)
        unit_size = slot.width / 5.0
        cols = max(1, min(5, int(round(bw / unit_size))))
        rows = max(1, min(5, int(round(bh / unit_size))))
        
        # Sub-cell dimensions
        # Sub-cell dimensions
        sub_w = bw / float(cols)
        sub_h = bh / float(rows)
        
        # Center the 5x5 grid on the piece BBox
        start_row = (5 - rows) // 2
        start_col = (5 - cols) // 2
        
        # 1. Draw Standardized 5x5 Background Grid
        grid_x0 = int(slot.x + bx - start_col * sub_w)
        grid_y0 = int(slot.y + by - start_row * sub_h)
        for r in range(6):
            py = int(grid_y0 + r * sub_h)
            cv2.line(vis, (grid_x0, py), (int(grid_x0 + 5 * sub_w), py), (70, 70, 70), 1)
        for c in range(6):
            px = int(grid_x0 + c * sub_w)
            cv2.line(vis, (px, grid_y0), (px, int(grid_y0 + 5 * sub_h)), (70, 70, 70), 1)
            
        # 2. Highlight Detected Cells
        for r in range(rows):
            for c in range(cols):
                y1, y2 = by + int(r * sub_h), by + int((r + 1) * sub_h)
                x1, x2 = bx + int(c * sub_w), bx + int((c + 1) * sub_w)
                margin_h, margin_w = max(1, int((y2 - y1) * 0.2)), max(1, int((x2 - x1) * 0.2))
                
                hsv_patch = hsv[y1+margin_h:y2-margin_h, x1+margin_w:x2-margin_w]
                if hsv_patch.size > 0:
                    avg_h, avg_s, avg_v = np.mean(hsv_patch[:, :, 0]), np.mean(hsv_patch[:, :, 1]), np.mean(hsv_patch[:, :, 2])
                    
                    is_filled = False
                    if avg_s > 150 and avg_v > 80:
                        is_filled = True
                    elif (avg_h < config.VISION_EXCLUDE_HUE_MIN or avg_h > config.VISION_EXCLUDE_HUE_MAX) and \
                         (avg_s > config.VISION_SAT_THRESHOLD and avg_v > config.VISION_VAL_THRESHOLD):
                        is_filled = True
                        
                    if is_filled:
                        # Draw detection markers perfectly center-aligned
                        cx, cy = int(slot.x + x1 + sub_w / 2), int(slot.y + y1 + sub_h / 2)
                        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
                        # Highlighting Box (Green)
                        cv2.rectangle(vis, (int(slot.x + x1 + margin_w), int(slot.y + y1 + margin_h)), 
                                      (int(slot.x + x2 - margin_w), int(slot.y + y2 - margin_h)), (0, 255, 0), 1)
    
    return vis


def visualize_drag(frame: np.ndarray, move: Move, start_pos: Tuple[int, int], click_pos: Tuple[int, int], expected_pos: Tuple[int, int]) -> np.ndarray:
    """
    Visualize the drag move with:
    - Red Cross: Actual cursor destination (click_pos, with offset)
    - Yellow Cross: Intended piece/board center (expected_pos)
    """
    vis = frame.copy()
    
    # 1. ACTUAL CURSOR DESTINATION (RED CROSS)
    cv2.drawMarker(vis, click_pos, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    
    # 2. EXPECTED BOARD DESTINATION (YELLOW CROSS)
    cv2.drawMarker(vis, expected_pos, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    
    # Optional: Draw a line between them to show the offset distance
    cv2.line(vis, expected_pos, click_pos, (200, 200, 200), 1)
    
    return vis
    
if __name__ == "__main__":
    print("Testing vision module...")
    # ... rest of test code ...
