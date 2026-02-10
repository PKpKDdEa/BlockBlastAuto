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
                
    def match_and_snap(self, grid: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """
        Compare input grid against library and snap to the best match.
        If no match is good enough, we check if it's a 'reasonable' new piece.
        """
        if len(self.templates) == 0:
            return grid
            
        best_match = None
        best_score = 0.0
        best_name = "unknown"
        
        for category, pieces in self.data.items():
            for name, grid_list in pieces.items():
                template = np.array(grid_list, dtype=np.uint8)
                # Jaccard Similarity: Intersection over Union
                int_sum = np.logical_and(grid, template).sum()
                uni_sum = np.logical_or(grid, template).sum()
                
                if uni_sum == 0: continue
                score = int_sum / float(uni_sum)
                
                if score > best_score:
                    best_score = score
                    best_match = template
                    best_name = f"{category}/{name}"
                
        # 1. High Confidence SNAPPING
        if best_match is not None and best_score >= threshold:
            return best_match, best_score, best_name
            
        # 2. NOISE FILTERING
        block_count = np.sum(grid)
        if best_score < 0.3 and block_count > 12:
            return np.zeros((5, 5), dtype=np.uint8), 0.0, "noise"
            
        return grid, best_score, "new-pattern" if block_count > 0 else "empty"
        
    def learn_pattern(self, grid: np.ndarray):
        """Save a new verified pattern to the library."""
        # Check if we already have it
        if any(np.array_equal(grid, t) for t in self.templates):
            return
            
        self.templates.append(grid)
        
        # Save back to file (simple append to a 'learned' category)
        try:
            data = {}
            if os.path.exists(self.templates_path):
                with open(self.templates_path, 'r') as f:
                    data = json.load(f)
            
            if "learned" not in data:
                data["learned"] = {}
                
            pattern_id = f"pattern_{len(data['learned']) + 1}"
            data["learned"][pattern_id] = grid.tolist()
            
            with open(self.templates_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            if config.DEBUG:
                print(f"LEARNED new piece pattern: {pattern_id}")
        except Exception as e:
            if config.DEBUG:
                print(f"Error saving learned template: {e}")

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
    # Background tray is usually saturation < 160. Pieces are high.
    lower_vibrant = np.array([0, 160, 60])
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
    grid_5x5 = get_piece_grid(piece_region)
    if grid_5x5 is None:
        return None
        
    # SNAPPING: Use template manager to clean up the detection
    final_grid, score, name = template_manager.match_and_snap(grid_5x5)
    
    if config.DEBUG and np.sum(final_grid) > 0:
        print(f"  Grid Detected (Match: {name}@{score:.2f})")
        for row in final_grid:
            print("  " + "".join(["#" if x else "." for x in row]))
            
    return final_grid


def get_piece_grid(piece_region: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a raw 5x5 binary grid from a piece slot region.
    Samples at fixed calibrated offsets from the center of the region.
    """
    if piece_region.size == 0:
        return None
    
    # HSV for vibrancy check
    hsv = cv2.cvtColor(piece_region, cv2.COLOR_BGR2HSV)
    
    # Use unified mask logic
    mask = get_piece_vibrancy_mask(hsv)
    sh, sw = piece_region.shape[:2]
    cx, cy = sw // 2, sh // 2
    cw, ch = config.TRAY_CELL_SIZE
    
    grid_5x5 = np.zeros((5, 5), dtype=np.uint8)
    
    for r in range(5):
        for c in range(5):
            # Sample at (r-2, c-2) relative to center
            px = int(cx + (c - 2) * cw)
            py = int(cy + (r - 2) * ch)
            
            # Bound check
            if 0 <= px < sw and 0 <= py < sh:
                # Sample a small patch for robustness
                margin = 2
                patch = mask[max(0, py-margin):min(sh, py+margin+1), 
                             max(0, px-margin):min(sw, px+margin+1)]
                
                if patch.size > 0 and np.mean(patch) > 100:
                    grid_5x5[r, c] = 1
                    
    # Diagnostic: Print average HSV of center cell if we found "too many" or "no" blocks
    if config.DEBUG:
        center_patch = hsv[max(0, cy-5):min(sh, cy+5), max(0, cx-5):min(sw, cx+5)]
        avg_hsv = np.mean(center_patch, axis=(0, 1)).astype(int)
        # If the grid is totally full (25) or empty (0), log the center color
        block_count = np.sum(grid_5x5)
        if block_count == 25 or block_count == 0:
            print(f"  [Slot Diagnostic] Center HSV: {avg_hsv} -> Blocks: {block_count}")
            
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
        mask = get_piece_vibrancy_mask(hsv)
        
        sh, sw = slot.height, slot.width
        cx_rel, cy_rel = sw // 2, sh // 2
        cw, ch = config.TRAY_CELL_SIZE
        
        # Draw the 5x5 fixed sampling grid for verification
        for r in range(5):
            for c in range(5):
                px = slot.x + cx_rel + (c - 2) * cw
                py = slot.y + cy_rel + (r - 2) * ch
                
                # Highlight the calculated center of this cell
                cv2.circle(vis, (px, py), 2, (70, 70, 70), -1)
                
                # Draw sampling boxes (visual indicator of where we "look")
                box_w, box_h = max(2, cw // 4), max(2, ch // 4)
                cv2.rectangle(vis, (px-box_w, py-box_h), (px+box_w, py+box_h), (100, 100, 100), 1)
                
                # Check if this cell is filled in the detect grid
                # (Re-run raw detection for this specific cell for visual sync)
                patch = mask[max(0, cy_rel + (r-2)*ch - 2):min(sh, cy_rel + (r-2)*ch + 3), 
                             max(0, cx_rel + (c-2)*cw - 2):min(sw, cx_rel + (c-2)*cw + 3)]
                if patch.size > 0 and np.mean(patch) > 100:
                    cv2.rectangle(vis, (px-box_w, py-box_h), (px+box_w, py+box_h), (0, 255, 0), 1)
                    cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)

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
            final_grid, score, name = template_manager.match_and_snap(grid_5x5)
            
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
