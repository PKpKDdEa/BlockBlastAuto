import cv2
import numpy as np
import time
import json
import os
from typing import List, Tuple, Optional, Dict
from config import config
from model import Board, Piece, Move


class TemplateManager:
    """Manages valid piece patterns and provides snapping/learning logic."""
    
    def __init__(self, templates_path: str = "data/templates.json"):
        self.templates_path = templates_path
        self.templates: List[np.ndarray] = []
        self.data: Dict = {}
        self._load_templates()
        
    def _load_templates(self):
        """Load templates from JSON into numpy arrays."""
        if not os.path.exists(self.templates_path):
            os.makedirs(os.path.dirname(self.templates_path), exist_ok=True)
            with open(self.templates_path, 'w') as f:
                json.dump({}, f)
                self.data = {}
            return
            
        try:
            with open(self.templates_path, 'r') as f:
                self.data = json.load(f)
                
            self.templates = []
            # Flatten the nested structure (categories -> names -> grids)
            for category in self.data.values():
                for grid_list in category.values():
                    self.templates.append(np.array(grid_list, dtype=np.uint8))
        except Exception as e:
            if config.DEBUG:
                print(f"Error loading templates: {e}")
                
    def match_and_validate(self, grid: np.ndarray, threshold: float = 0.82) -> Tuple[np.ndarray, float, str, Dict]:
        """
        Compare input grid against library with NMS and shape validation.
        Returns (final_grid, score, name, match_info)
        """
        if len(self.templates) == 0:
            return grid, 0.0, "unknown", {"is_new": True}
            
        best_match = None
        best_score = 0.0
        best_name = "unknown"
        
        # 1. Multi-template matching
        for category, pieces in self.data.items():
            for name, grid_list in pieces.items():
                template = np.array(grid_list, dtype=np.uint8)
                int_sum = np.logical_and(grid, template).sum()
                uni_sum = np.logical_or(grid, template).sum()
                
                if uni_sum == 0: continue
                score = int_sum / float(uni_sum)
                
                if score > best_score:
                    # v2.3: Mass-Awareness. Check if the candidate matching blocks actually exists.
                    if self.check_mass_mismatch(grid, template):
                        best_score = score
                        best_match = template
                        best_name = f"{category}/{name}"
        
        # 2. Shape Validation Rules (Geometry Enforcement)
        is_valid = self.validate_shape(grid, best_name, best_score)
        
        match_info = {
            "best_score": best_score,
            "is_valid": is_valid,
            "category": best_name.split('/')[0] if '/' in best_name else "unknown"
        }

        # 3. High Confidence Snapping with Validation
        # v2.2: Aggressive Snapping. If score is extremely high (>0.85), trust the match even if validation is iffy.
        if best_match is not None:
            if (best_score >= threshold and is_valid) or (best_score >= 0.85):
                return best_match, best_score, best_name, match_info
            
        # 4. Strict Overread Protection (NMS-like filtering)
        # If we have a lot of blocks but low score, it's likely a misread/noise
        block_count = np.sum(grid)
        if best_score < 0.5 and block_count > 10:
             return np.zeros((5, 5), dtype=np.uint8), 0.0, "noise", match_info

        return grid, best_score, "unknown", match_info

    def validate_shape(self, grid: np.ndarray, name: str, score: float) -> bool:
        """
        Enforce geometric rules for specific shape categories.
        Rules (loosened in v2.2):
        - dot: exactly 1 block
        - line: 2-5 blocks (+/- 1 noise tolerance)
        - square: 4 blocks (2x2) or 9 blocks (3x3) (+/- 1 noise tolerance)
        """
        blocks = np.sum(grid)
        if blocks == 0: return True
        
        category = name.split('/')[0] if '/' in name else ""
        
        if category == "dots":
            return blocks == 1
        elif category == "lines":
            # Allow +/- 1 block variance for noisy detections
            if not (1 <= blocks <= 6): return False
            rows = np.any(grid, axis=1)
            cols = np.any(grid, axis=0)
            # Must be primarily elongated
            return np.sum(rows) == 1 or np.sum(cols) == 1
        elif category == "squares":
            # 2x2 (4 blocks) or 3x3 (9 blocks) with +/- 1 block tolerance
            if 3 <= blocks <= 5: # 2x2 variant
                return True
            if 7 <= blocks <= 10: # 3x3 variant
                return True
            return False
        elif category in ["corners", "l_shapes", "t_shapes", "zs_shapes", "diag_shapes"]:
            # Complex shapes usually have 3-5 blocks. Loosen range.
            return 2 <= blocks <= 6
            
        return score > 0.75 # Default fallback for unknown categories

    def check_mass_mismatch(self, grid: np.ndarray, template: np.ndarray) -> bool:
        """
        Check if the total volume (number of blocks) is vastly different.
        Prevents a 4-block piece from matching a 9-block square.
        """
        visual_blocks = np.sum(grid)
        template_blocks = np.sum(template)
        
        # Allow +/- 1 block variance for noise
        return abs(visual_blocks - template_blocks) <= 2

    def learn_pattern(self, grid: np.ndarray):
        """[DISABLED in v2.1] - Managed templates only."""
        pass

# Global instance
template_manager = TemplateManager()


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
            # Use float-based centers to avoid cumulative rounding drift
            cell_x = int(col * config.CELL_WIDTH + config.CELL_WIDTH / 2)
            cell_y = int(row * config.CELL_HEIGHT + config.CELL_HEIGHT / 2)
            
            # Extract small patch around center
            patch_size = int(min(config.CELL_WIDTH, config.CELL_HEIGHT) // 3)
            y_start = int(max(0, cell_y - patch_size // 2))
            y_end = int(min(grid_region.shape[0], cell_y + patch_size // 2))
            x_start = int(max(0, cell_x - patch_size // 2))
            x_end = int(min(grid_region.shape[1], cell_x + patch_size // 2))
            
            patch = grid_region[y_start:y_end, x_start:x_end]
            
            is_filled = classify_cell(patch)
            board.grid[row, col] = 1 if is_filled else 0
            if is_filled:
                board.bitboard |= (1 << (row * 8 + col))
            
    if config.DEBUG:
        filled_count = np.sum(board.grid == 1)
        print(f"Board detected: {filled_count}/64 cells filled")
    
    return board


def classify_cell(patch: np.ndarray) -> bool:
    """
    Classifies a board cell as filled or empty.
    v3.2: Increased threshold to avoid board background bleed.
    """
    if patch.size == 0:
        return False
    
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = get_piece_vibrancy_mask(hsv)
    
    # v3.1 Brightness Guard: Empty cells are dark (dimmed)
    val_channel = hsv[:, :, 2]
    if np.mean(val_channel) < 80:
        return False
        
    # v3.2: High threshold (140) to ignore dimmed board background
    return np.mean(mask) > 140


def get_piece_vibrancy_mask(hsv_img: np.ndarray) -> np.ndarray:
    """
    Unified vibrancy-aware mask for pieces. 
    v3.2: Surgical Hue Masking with high-saturation preservation.
    """
    # Stage 1: Absolute Vibrancy (Catches any block with high saturation)
    # v3.2: Baseline S=150 to catch dark but vibrant pieces
    lower_vibrant = np.array([0, config.VISION_SAT_THRESHOLD, config.VISION_VAL_THRESHOLD])
    upper_vibrant = np.array([180, 255, 255])
    mask_high_sat = cv2.inRange(hsv_img, lower_vibrant, upper_vibrant)
    
    # Stage 2: Color Selective (Green, Red, Yellow, Purple)
    # We allow lower saturation for non-blue colors
    lower_r1 = np.array([0, 100, 60])
    upper_r1 = np.array([config.VISION_EXCLUDE_HUE_MIN, 255, 255])
    
    lower_r2 = np.array([config.VISION_EXCLUDE_HUE_MAX, 100, 60])
    upper_r2 = np.array([180, 255, 255])
    
    mask_r1 = cv2.inRange(hsv_img, lower_r1, upper_r1)
    mask_r2 = cv2.inRange(hsv_img, lower_r2, upper_r2)
    
    # Stage 3: SURGICAL BLUE EXCLUSION (Specifically targets the Background)
    # v3.2: Widened Hue [100-145] and Raised Saturation Gate to 220.
    # MuMu background is S~160. Pieces are S~240.
    lower_bg_blue = np.array([config.VISION_EXCLUDE_HUE_MIN, 0, 0])
    upper_bg_blue = np.array([config.VISION_EXCLUDE_HUE_MAX, 220, 255])
    mask_bg_blue = cv2.inRange(hsv_img, lower_bg_blue, upper_bg_blue)
    
    # Combined Logic:
    # (High Saturation OR Selective Color) AND (NOT Background Blue)
    mask = cv2.bitwise_or(mask_high_sat, cv2.bitwise_or(mask_r1, mask_r2))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_bg_blue))
    
    # Cleanup noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def read_pieces(frame: np.ndarray) -> List[Piece]:
    """
    Detect available pieces from piece slots.
    """
    pieces = []
    
    for slot_idx, slot in enumerate(config.PIECE_SLOTS):
        piece_region = frame[slot.y:slot.y+slot.height, slot.x:slot.x+slot.width]
        mask, is_new = detect_piece_mask(piece_region)
        
        if mask is not None and np.sum(mask) > 0:
            piece = Piece.from_mask(piece_id=slot_idx, mask=mask, is_new=is_new)
            pieces.append(piece)
            if config.DEBUG:
                print(f"Piece {slot_idx} detected: {piece.width}x{piece.height}")
        else:
            pieces.append(None)
    
    return pieces


def detect_piece_mask(piece_region: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
    """
    Detect the piece shape and identify if it is a new pattern.
    Returns (final_grid, is_new)
    """
    grid_5x5 = get_piece_grid(piece_region)
    if grid_5x5 is None:
        return None, False
        
    # SNAPPING: Use template manager to clean up the detection
    final_grid, score, name, match_info = template_manager.match_and_validate(grid_5x5)
    
    if config.DEBUG and np.sum(final_grid) > 0:
        print(f"  Grid Detected (Match: {name}@{score:.2f})")
        for row in final_grid:
            print("  " + "".join(["#" if x else "." for x in row]))
            
    return final_grid, match_info.get("is_new", False)


def get_piece_grid(piece_region: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a raw 5x5 binary grid from a piece slot region.
    v3.0 Precision: Contour-Driven Grid Scaling + Boundary Enforcement.
    """
    if piece_region.size == 0:
        return None
    
    hsv = cv2.cvtColor(piece_region, cv2.COLOR_BGR2HSV)
    mask = get_piece_vibrancy_mask(hsv)
    sh, sw = piece_region.shape[:2]
    
    # 1. Use contours to find the piece boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx_cnt = int(M["m10"] / M["m00"])
        cy_cnt = int(M["m01"] / M["m00"])
        dist = abs(cx_cnt - sw//2) + abs(cy_cnt - sh//2)
        candidates.append((area, dist, cx_cnt, cy_cnt, cnt))
        
    if not candidates:
        return None
        
    # Closest to center is our piece
    best_cnt_data = min(candidates, key=lambda x: x[1])
    main_cnt = best_cnt_data[4]
    
    # v3.2: Noise Filter - Relaxed to 95% to allow for 5x1 bars
    bx, by, bw, bh = cv2.boundingRect(main_cnt)
    if bw > sw * 0.95 and bh > sh * 0.95:
        if config.DEBUG: print(f"  [v3.2] Rejected: Contour too large ({bw}x{bh})")
        return None
    
    # v3.0: Snap to nearest 42px cell count
    cols = int(round(bw / 42.0))
    rows = int(round(bh / 42.0))
    
    # Bounds safety
    cols = max(1, min(5, cols))
    rows = max(1, min(5, rows))
    
    # Anchor point: top-left cell center of the piece
    piece_cx = bx + bw / 2.0
    piece_cy = by + bh / 2.0
    
    # Cell size
    cw, ch = config.TRAY_CELL_SIZE
    
    grid_5x5 = np.zeros((5, 5), dtype=np.uint8)
    
    # 3. SAMPLE ONLY WITHIN THE DISCOVERED DIMENSIONS
    # Calculate offset to center the inferred rows/cols within the 5x5 grid
    start_r = (5 - rows) // 2
    start_c = (5 - cols) // 2
    
    for r_idx in range(rows):
        for c_idx in range(cols):
            # Calculate pixel center for this block cell
            # The piece_cx/piece_cy is the center of the total object
            # We need to offset from this center based on which block we are at
            rel_cx = (c_idx - (cols - 1) / 2.0) * cw
            rel_cy = (r_idx - (rows - 1) / 2.0) * ch
            
            cx_cell = int(piece_cx + rel_cx)
            cy_cell = int(piece_cy + rel_cy)
            
            # Sub-grid consensus + Point Boundary Check
            offset = int(cw * 0.22)
            points_on = 0
            
            for my in [-offset, 0, offset]:
                for mx in [-offset, 0, offset]:
                    px, py = cx_cell + mx, cy_cell + my
                    if 0 <= px < sw and 0 <= py < sh:
                        # v3.0: MUST be vibrant AND inside the polygon contour
                        if mask[py, px] > 0 and cv2.pointPolygonTest(main_cnt, (float(px), float(py)), False) >= 0:
                            points_on += 1
            
            if points_on >= 4: # Lowered threshold (4/9) due to strict boundary exclusion
                grid_5x5[start_r + r_idx, start_c + c_idx] = 1
                
    if config.DEBUG:
        block_count = np.sum(grid_5x5)
        if block_count > 0:
            print(f"  [v3.2 Precision] Inferred: {cols}x{rows} at ({piece_cx:.1f}, {piece_cy:.1f}) -> Blocks: {block_count}")
            
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
        y = int(y1 + row * config.CELL_HEIGHT)
        cv2.line(vis, (x1, y), (x2, y), (0, 255, 0), 1)
    
    for col in range(config.GRID_COLS + 1):
        x = int(x1 + col * config.CELL_WIDTH)
        cv2.line(vis, (x, y1), (x, y2), (0, 255, 0), 1)
    
    # Mark filled cells (Red Dots)
    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            cx = int(x1 + col * config.CELL_WIDTH + config.CELL_WIDTH / 2.0)
            cy = int(y1 + row * config.CELL_HEIGHT + config.CELL_HEIGHT / 2.0)
            color = (0, 0, 255) if board.grid[row, col] == 1 else (100, 100, 100)
            cv2.circle(vis, (cx, cy), 3, color, -1)
    
    # Draw piece slots and their internal relative grids (v3.0 Precision logic)
    for slot_idx, slot in enumerate(config.PIECE_SLOTS):
        cv2.rectangle(vis, (slot.x, slot.y), (slot.x + slot.width, slot.y + slot.height), (255, 0, 0), 1)
        
        piece_region = frame[slot.y:slot.y+slot.height, slot.x:slot.x+slot.width]
        if piece_region.size == 0: continue
        hsv = cv2.cvtColor(piece_region, cv2.COLOR_BGR2HSV)
        mask = get_piece_vibrancy_mask(hsv)
        
        # 1. Exact duplicate of get_piece_grid v3.0 logic for viz sync
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100: continue
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx_cnt = int(M["m10"] / M["m00"])
            cy_cnt = int(M["m01"] / M["m00"])
            dist = abs(cx_cnt - piece_region.shape[1]//2) + abs(cy_cnt - piece_region.shape[0]//2)
            candidates.append((area, dist, cx_cnt, cy_cnt, cnt))
            
        if not candidates: continue
        best_cnt_data = min(candidates, key=lambda x: x[1])
        main_cnt = best_cnt_data[4]
        bx, by, bw, bh = cv2.boundingRect(main_cnt)
        
        # Inferred dims
        cols = int(round(bw / 42.0))
        rows = int(round(bh / 42.0))
        cols = max(1, min(5, cols))
        rows = max(1, min(5, rows))
        
        piece_cx = bx + bw / 2.0
        piece_cy = by + bh / 2.0
        cw, ch = config.TRAY_CELL_SIZE

        # Draw the "White Bracket" (Tight Bounding Box)
        cv2.rectangle(vis, (slot.x + bx, slot.y + by), (slot.x + bx + bw, slot.y + by + bh), (255, 255, 255), 1)

        # Draw the inferred grid samples
        for r_idx in range(rows):
            for c_idx in range(cols):
                rel_cx = (c_idx - (cols - 1) / 2.0) * cw
                rel_cy = (r_idx - (rows - 1) / 2.0) * ch
                
                cx_cell = int(piece_cx + rel_cx)
                cy_cell = int(piece_cy + rel_cy)
                offset = int(cw * 0.22)
                
                points_on = 0
                for my in [-offset, 0, offset]:
                    for mx in [-offset, 0, offset]:
                        px, py = slot.x + cx_cell + mx, slot.y + cy_cell + my
                        if 0 <= cx_cell + mx < piece_region.shape[1] and 0 <= cy_cell + my < piece_region.shape[0]:
                            # v3.0 boundary test
                            is_on = mask[cy_cell + my, cx_cell + mx] > 0 and cv2.pointPolygonTest(main_cnt, (float(cx_cell+mx), float(cy_cell+my)), False) >= 0
                            color = (0, 0, 255) if is_on else (100, 100, 100)
                            cv2.circle(vis, (px, py), 1, color, -1)
                            if is_on: points_on += 1
                
                # If cell is ON, draw a green indicator box
                if points_on >= 4:
                    cv2.rectangle(vis, (slot.x + cx_cell - 8, slot.y + cy_cell - 8), 
                                  (slot.x + cx_cell + 8, slot.y + cy_cell + 8), (0, 255, 0), 1)

    return vis


def visualize_piece_analysis(frame: np.ndarray, pieces: List[Piece]) -> np.ndarray:
    """
    Creates a dedicated, stable analysis frame showing piece identities and grids.
    This window only updates when a new turn starts, so it won't 'flash'.
    """
    # Create a nice dark background for analysis
    vis = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(vis, "PIECE DIAGNOSTICS (Updated Once Per Turn)", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    for i, slot in enumerate(config.PIECE_SLOTS):
        x_off = 20 + i * 260
        y_off = 60
        
        # Draw slot header
        cv2.putText(vis, f"Slot {i}", (x_off, y_off), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Get piece from slots
        piece_region = frame[slot.y:slot.y+slot.height, slot.x:slot.x+slot.width]
        if piece_region.size == 0: continue
        
        # Show piece crop (resized for visibility)
        crop = cv2.resize(piece_region, (200, 180))
        vis[y_off+20:y_off+200, x_off:x_off+200] = crop
        
        # Get identification details
        grid_5x5 = get_piece_grid(piece_region)
        if grid_5x5 is not None:
            final_grid, score, name, match_info = template_manager.match_and_validate(grid_5x5)
            
            # Label
            label = f"{name}"
            raw_count = np.sum(grid_5x5)
            l2 = f"Match: {score:.2f} (Blocks: {raw_count})"
            color = (0, 255, 0) if score > 0.9 else (0, 255, 255) if score > 0.7 else (0, 165, 255)
            
            cv2.putText(vis, label, (x_off, y_off + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(vis, l2, (x_off, y_off + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            
            # Draw the 5x5 digital grid representation
            grid_y = y_off + 260
            cell_size = 25
            for r in range(5):
                for c in range(5):
                    gx = x_off + c * cell_size
                    gy = grid_y + r * cell_size
                    
                    # RAW DETECTED (Light Blue)
                    if grid_5x5[r, c]:
                        cv2.rectangle(vis, (gx, gy), (gx+cell_size-2, gy+cell_size-2), (100, 100, 100), -1)
                        
                    # SNAPPED / FINAL (Orange)
                    if final_grid[r, c]:
                        cv2.rectangle(vis, (gx, gy), (gx+cell_size-2, gy+cell_size-2), (0, 100, 255), -1)
                    
                    if not grid_5x5[r, c] and not final_grid[r, c]:
                        cv2.rectangle(vis, (gx, gy), (gx+cell_size-2, gy+cell_size-2), (40, 40, 40), 1)
        else:
            # Explicitly label empty slots
            cv2.putText(vis, "EMPTY / PLACED", (x_off, y_off + 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                        
    return vis


def draw_pause_status(vis: np.ndarray, is_paused: bool):
    """Overlay a large 'PAUSED' message if the bot is not running."""
    if is_paused:
        # Semi-transparent overlay
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        
        # Text
        text = "PAUSED (F10 to Resume)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 3
        size = cv2.getTextSize(text, font, scale, thickness)[0]
        tx = (vis.shape[1] - size[0]) // 2
        ty = (vis.shape[0] + size[1]) // 2
        
        # Shadow
        cv2.putText(vis, text, (tx+2, ty+2), font, scale, (0, 0, 0), thickness)
        # Main text (Yellow/Amber)
        cv2.putText(vis, text, (tx, ty), font, scale, (0, 190, 255), thickness)


def visualize_drag(frame: np.ndarray, piece: Piece, move: Move, start_xy: Tuple[int, int], click_xy: Tuple[int, int], end_xy: Tuple[int, int]) -> np.ndarray:
    """
    Visualize the drag move with:
    - Piece Blueprint: Transparent overlay of the piece at its target position.
    - Cell Highlights: Circles on the exact cells being filled.
    - Red Cross: Actual cursor destination (click_xy, where the mouse clicks).
    - Yellow Cross: Intended piece anchor center (end_xy).
    """
    vis = frame.copy()
    x1, y1 = config.GRID_TOP_LEFT
    
    # 1. DRAW PIECE BLUEPRINT & CELL HIGHLIGHTS
    # We draw where the bot thinks the pieces will land
    for dr, dc in piece.cells:
        r, c = move.row + dr, move.col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            # Calculate screen center of this specific cell
            cx = int(x1 + c * config.CELL_WIDTH + config.CELL_WIDTH / 2.0)
            cy = int(y1 + r * config.CELL_HEIGHT + config.CELL_HEIGHT / 2.0)
            
            # Draw a block blueprint (Semi-transparent orange)
            overlay = vis.copy()
            bw, bh = int(config.CELL_WIDTH * 0.8), int(config.CELL_HEIGHT * 0.8)
            cv2.rectangle(overlay, (cx - bw//2, cy - bh//2), (cx + bw//2, cy + bh//2), (0, 165, 255), -1)
            cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
            
            # Precise center dot for this cell
            cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)

    # 2. DRAW DRAG PATH
    cv2.line(vis, start_xy, click_xy, (200, 200, 200), 1, cv2.LINE_AA)
    
    # 3. ACTUAL CURSOR DESTINATION (RED CROSS - where it drags to)
    cv2.drawMarker(vis, click_xy, (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.putText(vis, "CURSOR", (click_xy[0] + 20, click_xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 4. EXPECTED BOARD ANCHOR (YELLOW CROSS - where the piece anchor should be)
    cv2.drawMarker(vis, end_xy, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 25, 2)
    cv2.putText(vis, "PIECE CENTER", (end_xy[0] + 15, end_xy[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return vis
    
if __name__ == "__main__":
    print("Testing vision module...")
    # ... rest of test code ...
