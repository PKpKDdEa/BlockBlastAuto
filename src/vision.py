import cv2
import math
import numpy as np
import time
import json
import os
from typing import List, Tuple, Optional, Dict
from config import config
from model import Board, Piece, Move


class TemplateManager:
    """Manages canonical Block Blast piece templates and snapping logic."""
    
    def __init__(self, templates_path: str = "data/templates.json"):
        self.templates_path = templates_path
        self.templates: List[np.ndarray] = []
        self.template_entries: List[Dict] = []
        self.data: Dict = {}
        self._load_templates()
        
    def _load_templates(self):
        """Load templates from JSON into numpy arrays and cache metadata."""
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
            self.template_entries = []
            for category_name, category in self.data.items():
                for name, grid_list in category.items():
                    template = np.array(grid_list, dtype=np.uint8)
                    self.templates.append(template)
                    rows = int(np.any(template, axis=1).sum())
                    cols = int(np.any(template, axis=0).sum())
                    self.template_entries.append({
                        "category": category_name,
                        "name": name,
                        "full_name": f"{category_name}/{name}",
                        "template": template,
                        "mass": int(np.sum(template)),
                        "rows": rows,
                        "cols": cols,
                    })
        except Exception as e:
            if config.DEBUG:
                print(f"Error loading templates: {e}")

    def _score_template(self, grid: np.ndarray, template: np.ndarray) -> float:
        """Best IoU score over valid ±2 shifts without losing mass."""
        best_score = 0.0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                shifted = self.shift_grid(grid, dr, dc)
                if np.sum(shifted) == 0:
                    continue
                int_sum = np.logical_and(shifted, template).sum()
                uni_sum = np.logical_or(shifted, template).sum()
                if uni_sum == 0:
                    continue
                score = int_sum / float(uni_sum)
                if score > best_score:
                    best_score = score
        return best_score
                
    def _extract_tight_sub(self, grid: np.ndarray):
        """Extract the tight bounding-box sub-grid and its position."""
        filled_rows = np.where(np.any(grid, axis=1))[0]
        filled_cols = np.where(np.any(grid, axis=0))[0]
        if len(filled_rows) == 0 or len(filled_cols) == 0:
            return None, 0, 0
        r0, r1 = filled_rows[0], filled_rows[-1]
        c0, c1 = filled_cols[0], filled_cols[-1]
        return grid[r0:r1+1, c0:c1+1], r1 - r0 + 1, c1 - c0 + 1

    def match_and_validate(
        self,
        grid: np.ndarray,
        threshold: float = 0.80,
        expected_dims: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, float, str, Dict]:
        """
        Snap a detected 5x5 grid to the closest canonical Block Blast template.
        NEVER returns 'unknown' when blocks are detected — always finds the
        best-matching template.

        Algorithm:
          1. Extract the tight bounding-box sub-grid from detected grid.
          2. For EVERY template:
             a. If bbox dimensions match: cell-by-cell comparison (aligned).
             b. Otherwise: IoU-shift fallback (pad both to same size, try all shifts).
          3. Rank by: cell_score > lower mass_diff.
          4. Always return the best match.
        """
        if len(self.template_entries) == 0:
            return grid, 0.0, "unknown", {"is_new": True}

        current_mass = int(np.sum(grid))
        if current_mass == 0:
            return grid, 0.0, "empty", {"is_new": False}

        det_sub, det_h, det_w = self._extract_tight_sub(grid)
        if det_sub is None:
            return grid, 0.0, "empty", {"is_new": False}

        candidates = []
        for entry in self.template_entries:
            t = entry["template"]
            mass_diff = abs(current_mass - entry["mass"])

            t_sub, t_h, t_w = self._extract_tight_sub(t)
            if t_sub is None:
                continue

            bbox_match = (det_h == t_h and det_w == t_w)

            if bbox_match:
                # Direct cell-by-cell comparison (bounding boxes aligned).
                match_cells = int(np.sum(det_sub == t_sub))
                total_cells = det_h * det_w
                cell_score = match_cells / float(total_cells)
            else:
                # IoU-shift fallback: pad both to the same dimensions
                # and try all valid placements for best alignment.
                big_h, big_w = max(det_h, t_h), max(det_w, t_w)
                best_pad_score = 0.0
                for dr in range(big_h - det_h + 1):
                    for dc in range(big_w - det_w + 1):
                        padded_det = np.zeros((big_h, big_w), dtype=np.uint8)
                        padded_det[dr:dr+det_h, dc:dc+det_w] = det_sub
                        for tr in range(big_h - t_h + 1):
                            for tc in range(big_w - t_w + 1):
                                padded_t = np.zeros((big_h, big_w), dtype=np.uint8)
                                padded_t[tr:tr+t_h, tc:tc+t_w] = t_sub
                                mc = int(np.sum(padded_det == padded_t))
                                sc = mc / float(big_h * big_w)
                                if sc > best_pad_score:
                                    best_pad_score = sc
                cell_score = best_pad_score

            candidates.append({
                **entry,
                "score": cell_score,
                "mass_diff": mass_diff,
                "dim_diff": abs(det_h - t_h) + abs(det_w - t_w),
                "bbox_match": bbox_match,
            })

        if not candidates:
            return grid, 0.0, "unknown", {"best_score": 0.0, "is_valid": False, "category": "unknown"}

        # Rank by composite score that penalises dimensional distance.
        #
        # Problem: raw cell_score counts empty-empty matches, so a 2x5
        # detection (10 blocks) scores higher against 2x3 (0.60) than 1x5
        # (0.50) because they share the same width.  But the correct match
        # is 1x5 — the extra column is a detection artefact.
        #
        # Fix: subtract a penalty proportional to the sum of dimensional
        # differences (dim_diff).  0.12 per unit of dim_diff means a
        # template must outscore a closer-dimension alternative by >0.12
        # per dimension to win.
        #
        #   2x5 vs 1x5 (dim_diff=1): 0.50 - 0.12 = 0.38
        #   2x5 vs 2x3 (dim_diff=2): 0.60 - 0.24 = 0.36  → 1x5 wins ✓
        #
        # bbox_match (dim_diff=0) gets no penalty, preserving exact-match
        # priority.
        for c in candidates:
            c["composite"] = c["score"] - c["dim_diff"] * 0.12

        candidates.sort(key=lambda c: (
            c["bbox_match"],        # True (1) > False (0)  — exact dims first
            c["composite"],         # higher is better (score - dim penalty)
            -c["mass_diff"],        # lower is better (negated)
        ), reverse=True)

        best = candidates[0]

        is_valid = self.validate_shape(grid, best["full_name"], best["score"])
        match_info = {
            "best_score": best["score"],
            "is_valid": is_valid,
            "category": best["category"],
            "mass_diff": best["mass_diff"],
            "dim_diff": best["dim_diff"],
            "bbox_match": best["bbox_match"],
        }

        # Always return the best template match — never return 'unknown'.
        # The score indicates confidence; downstream logic can use it.
        return best["template"], best["score"], best["full_name"], match_info

    def shift_grid(self, grid: np.ndarray, dr: int, dc: int) -> np.ndarray:
        """Shift a 5x5 grid by (dr, dc), aborting if mass would be lost."""
        res = np.zeros_like(grid)
        original_mass = int(np.sum(grid))
        for r in range(5):
            for c in range(5):
                if grid[r, c] == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 5 and 0 <= nc < 5:
                    res[nr, nc] = 1
        if int(np.sum(res)) != original_mass:
            return np.zeros_like(grid)
        return res

    def validate_shape(self, grid: np.ndarray, name: str, score: float) -> bool:
        """Validate against broad canonical Block Blast families."""
        blocks = int(np.sum(grid))
        if blocks == 0:
            return True
        
        category = name.split('/')[0] if '/' in name else ""
        
        if category == "dots":
            return blocks == 1
        if category == "lines":
            rows = np.any(grid, axis=1)
            cols = np.any(grid, axis=0)
            return 1 <= blocks <= 5 and (np.sum(rows) == 1 or np.sum(cols) == 1)
        if category == "squares":
            return blocks in {4, 9}
        if category in {"corners", "l_shapes", "rectangles", "specials"}:
            return 2 <= blocks <= 6
            
        return score > 0.80

    def check_mass_mismatch(self, grid: np.ndarray, template: np.ndarray) -> bool:
        return int(np.sum(grid)) == int(np.sum(template))

    def learn_pattern(self, grid: np.ndarray):
        """[DISABLED in v2.1] - Managed templates only."""
        pass

# Global instance
template_manager = TemplateManager()


def read_board(frame: np.ndarray) -> Board:
    """
    Detect board state from game screenshot.
    v4.0: Improved robustness for Block Blast cells.
    """
    board = Board(config.GRID_ROWS, config.GRID_COLS)
    
    # Extract grid region
    x1, y1 = config.GRID_TOP_LEFT
    x2, y2 = config.GRID_BOTTOM_RIGHT
    grid_region = frame[y1:y2, x1:x2]
    
    # Pre-process board for vibrancy
    lab = cv2.cvtColor(grid_region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    board_en = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(board_en, cv2.COLOR_BGR2HSV)
    
    # Unified mask for all colorful blocks on board
    # v4.1: Pass is_board=True to lower saturation thresholds
    mask = get_piece_vibrancy_mask(hsv, is_board=True)
    
    # Process each cell
    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            # cell center in grid coordinates
            cx = int(col * config.CELL_WIDTH + config.CELL_WIDTH / 2)
            cy = int(row * config.CELL_HEIGHT + config.CELL_HEIGHT / 2)
            
            # v4.0 Multi-point "Cross" Sampling
            # Sample center + 4 offsets to handle highlights/textures
            offset = int(config.CELL_WIDTH * 0.2)
            points = [(cx, cy), (cx-offset, cy), (cx+offset, cy), (cx, cy-offset), (cx, cy+offset)]
            
            filled_points = 0
            for px, py in points:
                if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0]:
                    if mask[py, px] > 0:
                        filled_points += 1
            
            # v4.3 High-Agreement Consensus (at least 4 points on to count as filled)
            is_filled = filled_points >= 4
            
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
    v3.3: Uses Contrast-Boosted mask + adaptive threshold.
    """
    if patch.size == 0:
        return False
    
    # Pre-process with CLAHE for board (Value channel)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    patch_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(patch_enhanced, cv2.COLOR_BGR2HSV)
    # For board, we still use a broad exclusion since background is fixed navy
    mask = get_piece_vibrancy_mask(hsv)
    
    # v3.3: Average mask value check (Filled blocks are very bright/distinct after CLAHE)
    return np.mean(mask) > 160


def get_piece_vibrancy_mask(hsv_img: np.ndarray, bg_sample: Optional[np.ndarray] = None, is_board: bool = False) -> np.ndarray:
    """
    Unified vibrancy-aware mask for pieces. 
    v4.2: Separated Board vs Slot logic. Board uses S=30 floor and NO background subtraction.
    """
    # Stage 1: Absolute Vibrancy
    # v5.3: Relaxed floors for slots to catch cyan/vampire pieces
    s_floor = 50 if is_board else 80
    v_floor = 110 if is_board else 130
    
    lower_vibrant = np.array([0, s_floor, v_floor]) 
    upper_vibrant = np.array([180, 255, 255])
    mask_high_sat = cv2.inRange(hsv_img, lower_vibrant, upper_vibrant)
    
    # Stage 2: Color Selective (Boost for difficult segments)
    # Range: Purple/Lavender (H: 130-165, relaxed for board)
    lower_purp = np.array([130, 45 if is_board else 40, 90 if is_board else 100])
    upper_purp = np.array([165, 255, 255])
    mask_purp = cv2.inRange(hsv_img, lower_purp, upper_purp)
    
    # Range: Navy/Dark Blue (H: 100-130, board only boost)
    mask_navy = np.zeros_like(mask_purp)
    if is_board:
        lower_navy = np.array([100, 45, 90])
        upper_navy = np.array([140, 255, 255])
        mask_navy = cv2.inRange(hsv_img, lower_navy, upper_navy)
    
    # Combine Absolute + Selective
    mask = cv2.bitwise_or(mask_high_sat, cv2.bitwise_or(mask_purp, mask_navy))
    
    # Stage 3: ADAPTIVE BACKGROUND REJECTION (SLOTS ONLY)
    if not is_board:
        if bg_sample is not None:
            bg_h, bg_s, bg_v = bg_sample[0]
            lower_bg = np.array([max(0, bg_h-10), 0, 0])
            upper_bg = np.array([min(180, bg_h+10), min(255, bg_s+20), 255])
            mask_bg = cv2.inRange(hsv_img, lower_bg, upper_bg)
            
            # Dark Blue Boost for SLOTS ONLY
            is_blue_bg = (100 <= bg_h <= 145)
            if is_blue_bg:
                lower_db = np.array([bg_h-15, int(bg_s + 20), 80])
                upper_db = np.array([bg_h+15, 255, 255])
                mask_db_boost = cv2.inRange(hsv_img, lower_db, upper_db)
                mask_high_sat = cv2.bitwise_or(mask_high_sat, mask_db_boost)
                mask = cv2.bitwise_or(mask_high_sat, mask_purp)
        else:
            mask_bg = cv2.inRange(hsv_img, np.array([100, 0, 0]), np.array([145, 180, 255]))
        
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_bg))
    
    # v5.4: Soften cleanup to prevent thin piece fragmentation
    kernel = np.ones((3, 3), np.uint8)
    if not is_board:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # No extra dilation for slots — keeps bounding box tight for accurate
        # grid sizing.  Previous CLOSE(2)+DILATE(1) inflated the mask ~8px,
        # causing cols/rows overestimation and every cell to appear filled.
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask


def read_pieces(frame: np.ndarray) -> List[Piece]:
    """
    Detect available pieces from piece slots.
    """
    pieces = []
    
    for slot_idx, slot in enumerate(config.PIECE_SLOTS):
        piece_region = frame[slot.y:slot.y+slot.height, slot.x:slot.x+slot.width]
        mask, is_new, t_cx, t_cy = detect_piece_mask(piece_region)
        
        if mask is not None and np.sum(mask) > 0:
            # v3.9: Store the actual piece center in the tray
            piece = Piece.from_mask(piece_id=slot_idx, mask=mask, is_new=is_new)
            piece.tray_cx = t_cx
            piece.tray_cy = t_cy
            pieces.append(piece)
            if config.DEBUG:
                print(f"Piece {slot_idx} detected: {piece.width}x{piece.height} at ({t_cx:.1f}, {t_cy:.1f})")
        else:
            pieces.append(None)
    
    return pieces


def detect_piece_mask(piece_region: np.ndarray) -> Tuple[Optional[np.ndarray], bool, float, float]:
    """
    Detect the piece shape and identify if it is a new pattern.
    Returns (final_grid, is_new, piece_cx, piece_cy)
    """
    grid_data = get_piece_grid(piece_region)
    if grid_data is None:
        return None, False, 0.0, 0.0
        
    grid_5x5, p_cx, p_cy, _d, _ax, _ay, cols, rows, _bcols, _brows = grid_data
    
    # SNAPPING: Only canonical Block Blast templates are valid outputs.
    final_grid, score, name, match_info = template_manager.match_and_validate(
        grid_5x5,
    )
    
    if config.DEBUG and np.sum(final_grid) > 0:
        print(f"  Grid Detected (Match: {name}@{score:.2f})")
        for row in final_grid:
            print("  " + "".join(["#" if x else "." for x in row]))
            
    return final_grid, match_info.get("is_new", False), p_cx, p_cy


def _sample_grid_cells(
    mask: np.ndarray, hsv: np.ndarray,
    bx: int, by: int, bw: int, bh: int,
    cols: int, rows: int, d_base: float,
    sw: int, sh: int, debug: bool = False,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Sample a cols×rows cell grid centred on the bbox and determine which
    cells are filled.  Returns (grid_5x5, d, anchor_x, anchor_y).
    """
    d_x = bw / float(cols) if cols > 0 else d_base
    d_y = bh / float(rows) if rows > 0 else d_base
    d = (d_x + d_y) / 2.0

    center_x = bx + bw / 2.0
    center_y = by + bh / 2.0
    anchor_x = center_x - ((cols - 1) * d) / 2.0
    anchor_y = center_y - ((rows - 1) * d) / 2.0

    grid_5x5 = np.zeros((5, 5), dtype=np.uint8)
    start_r = (5 - rows) // 2
    start_c = (5 - cols) // 2
    is_elongated_line = (rows == 1 and cols >= 4) or (cols == 1 and rows >= 4)

    for r_idx in range(rows):
        for c_idx in range(cols):
            cx_cell = int(round(anchor_x + c_idx * d))
            cy_cell = int(round(anchor_y + r_idx * d))

            offset = max(1, int(d * 0.18))
            points_on = 0
            sat_on = 0
            for my in [-offset, 0, offset]:
                for mx in [-offset, 0, offset]:
                    px, py = cx_cell + mx, cy_cell + my
                    if 0 <= px < sw and 0 <= py < sh:
                        if mask[py, px] > 0:
                            points_on += 1
                        if hsv[py, px, 1] >= 100:
                            sat_on += 1

            half_span = max(4, int(d * 0.35))
            rx1 = max(0, cx_cell - half_span)
            rx2 = min(sw, cx_cell + half_span + 1)
            ry1 = max(0, cy_cell - half_span)
            ry2 = min(sh, cy_cell + half_span + 1)
            roi = mask[ry1:ry2, rx1:rx2]
            occupancy = float(np.mean(roi > 0)) if roi.size > 0 else 0.0
            center_on = 0 <= cx_cell < sw and 0 <= cy_cell < sh and mask[cy_cell, cx_cell] > 0

            mask_pass = (
                points_on >= 5
                or occupancy >= 0.40
                or (center_on and points_on >= 3)
            )
            filled = mask_pass and sat_on >= 4

            if debug:
                status = "FILL" if filled else "SKIP"
                print(f"    Cell({c_idx},{r_idx}) pts={points_on} occ={occupancy:.2f} sat={sat_on} mask={'Y' if mask_pass else 'N'} -> {status}")

            if filled:
                grid_5x5[start_r + r_idx, start_c + c_idx] = 1

    # Elongated-line gap bridging
    if is_elongated_line:
        if rows == 1:
            slice_arr = grid_5x5[start_r, start_c:start_c + cols]
            for k in range(1, len(slice_arr) - 1):
                if slice_arr[k] == 0 and slice_arr[k-1] == 1 and slice_arr[k+1] == 1:
                    grid_5x5[start_r, start_c + k] = 1
        else:
            slice_arr = grid_5x5[start_r:start_r + rows, start_c]
            for k in range(1, len(slice_arr) - 1):
                if slice_arr[k] == 0 and slice_arr[k-1] == 1 and slice_arr[k+1] == 1:
                    grid_5x5[start_r + k, start_c] = 1

    return grid_5x5, d, anchor_x, anchor_y


def get_piece_grid(piece_region: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a raw 5x5 binary grid from a piece slot region.
    Uses multi-candidate dimension estimation: tries the primary (ceil + clamp)
    estimate AND one-less in each dimension, picks the one with the best
    template match score.  This fixes cases where bbox overestimate causes
    wrong cell positions (e.g. 3-row piece estimated as 4 → cells land
    between blocks → wrong fill pattern).
    """
    if piece_region.size == 0:
        return None
    
    # 1. Enhanced CLAHE
    lab = cv2.cvtColor(piece_region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    piece_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(piece_enhanced, cv2.COLOR_BGR2HSV)
    
    # v3.6: Multi-Corner Background Sampling
    h, w = hsv.shape[:2]
    corners = [hsv[2:7, 2:7], hsv[2:7, w-7:w-2], hsv[h-7:h-2, 2:7], hsv[h-7:h-2, w-7:w-2]]
    bg_sample = np.mean(np.concatenate([c.reshape(-1, 3) for c in corners]), axis=0).reshape(1, 3)
    
    mask = get_piece_vibrancy_mask(hsv, bg_sample=bg_sample)
    sh, sw = piece_region.shape[:2]
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # v5.4 Contour Merging
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80: continue
        
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx_cnt = int(M["m10"] / M["m00"])
        cy_cnt = int(M["m01"] / M["m00"])
        
        dist_x = abs(cx_cnt - sw//2) / float(sw)
        dist_y = abs(cy_cnt - sh//2) / float(sh)
        if dist_x < 0.35 and dist_y < 0.35:
            valid_contours.append(cnt)
            
    if not valid_contours:
        return None
        
    all_points = np.concatenate(valid_contours)
    bx, by, bw, bh = cv2.boundingRect(all_points)
    
    d_base = float(config.TRAY_CELL_SIZE[0])

    # --- Multi-candidate dimension estimation ---
    # Primary estimate: ceil() to oversample, then clamp if pitch too small.
    max_cols = max(1, min(5, math.ceil(bw / d_base)))
    max_rows = max(1, min(5, math.ceil(bh / d_base)))
    min_d = d_base * 0.90
    while max_cols > 1 and bw / float(max_cols) < min_d:
        max_cols -= 1
    while max_rows > 1 and bh / float(max_rows) < min_d:
        max_rows -= 1

    # Generate candidates: primary + reduced in each dimension + both reduced.
    # When rows and/or cols is overestimated, a reduced candidate will
    # produce correct cell positions and a better template match.
    dim_candidates_set = {(max_cols, max_rows)}
    if max_rows > 1:
        dim_candidates_set.add((max_cols, max_rows - 1))
    if max_cols > 1:
        dim_candidates_set.add((max_cols - 1, max_rows))
    if max_cols > 1 and max_rows > 1:
        dim_candidates_set.add((max_cols - 1, max_rows - 1))
    dim_candidates = sorted(dim_candidates_set, reverse=True)

    # Estimate expected block count from mask area.
    # Used to penalize candidates that detect far fewer blocks than expected.
    mask_area = float(cv2.countNonZero(mask[by:by+bh, bx:bx+bw]))
    cell_area = d_base * d_base
    expected_blocks = max(1.0, mask_area / cell_area)

    best_grid = None
    best_composite = -1
    best_meta = None

    for try_cols, try_rows in dim_candidates:
        grid, d, ax, ay = _sample_grid_cells(
            mask, hsv, bx, by, bw, bh,
            try_cols, try_rows, d_base, sw, sh, debug=False,
        )
        n_blocks = int(np.sum(grid))
        if n_blocks == 0:
            continue
        _, score, name, _ = template_manager.match_and_validate(grid)

        # Composite score: template match quality × sqrt(block coverage).
        #
        # block_ratio = detected_blocks / expected_blocks, capped at 1.
        # We use sqrt() to soften the penalty.  The mask area overestimates
        # expected_blocks due to morphological fill (CLOSE fills gaps between
        # adjacent blocks), so a correct candidate with fewer-than-expected
        # blocks shouldn't be penalized too harshly.
        #
        # Linear (old, broken):
        #   (3,3) 5 blocks, expected≈7: ratio=0.68, comp = 1.00 × 0.68 = 0.68
        #   (3,4) 8 blocks:             ratio=1.00, comp = 0.75 × 1.00 = 0.75 ← wrong wins
        #
        # Sqrt (new, fixed):
        #   (3,3) 5 blocks: ratio=0.68, sqrt=0.82, comp = 1.00 × 0.82 = 0.82 ← correct wins
        #   (3,4) 8 blocks: ratio=1.00, sqrt=1.00, comp = 0.75 × 1.00 = 0.75
        #
        # But a 3-block candidate capturing half the piece still loses:
        #   (2,3) 3 blocks, expected≈5: ratio=0.60, sqrt=0.77, comp = 1.00 × 0.77 = 0.77
        #   (3,3) 5 blocks:             ratio=1.00, sqrt=1.00, comp = 0.90 × 1.00 = 0.90 ← correct wins
        block_ratio = min(1.0, n_blocks / expected_blocks)
        composite = score * math.sqrt(block_ratio)

        if config.DEBUG:
            print(f"  Candidate({try_cols}x{try_rows}): blocks={n_blocks}, match={name}@{score:.2f}, ratio={block_ratio:.2f}, composite={composite:.2f}")

        if composite > best_composite:
            best_composite = composite
            best_grid = grid.copy()
            best_meta = (d, ax, ay, try_cols, try_rows)

    if best_grid is None or best_meta is None:
        return None

    grid_5x5 = best_grid
    d, anchor_x, anchor_y, cols, rows = best_meta

    # Re-run with debug output for the winning candidate
    if config.DEBUG:
        print(f"  Winner: ({cols}x{rows}) composite={best_composite:.2f}")
        _sample_grid_cells(
            mask, hsv, bx, by, bw, bh,
            cols, rows, d_base, sw, sh, debug=True,
        )

    piece_cx = bx + bw / 2.0
    piece_cy = by + bh / 2.0
    
    # Recompute actual dimensions from the filled cells
    filled_rows_idx = np.where(np.any(grid_5x5, axis=1))[0]
    filled_cols_idx = np.where(np.any(grid_5x5, axis=0))[0]
    if len(filled_rows_idx) > 0 and len(filled_cols_idx) > 0:
        actual_rows = int(filled_rows_idx[-1] - filled_rows_idx[0] + 1)
        actual_cols = int(filled_cols_idx[-1] - filled_cols_idx[0] + 1)
    else:
        actual_rows = rows
        actual_cols = cols
    
    if config.DEBUG:
        block_count = np.sum(grid_5x5)
        if block_count > 0:
            print(f"  [v6] Pitch: {d:.1f}px, Blocks: {block_count}, Grid: {actual_cols}x{actual_rows} (est: {cols}x{rows})")
            
    # Return: grid, center, pitch, anchor, actual dims, bbox-estimated dims
    return grid_5x5, piece_cx, piece_cy, d, anchor_x, anchor_y, actual_cols, actual_rows, cols, rows


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


def visualize_detection(frame: np.ndarray, board: Board, pieces: List[Piece], scan_piece_slots: bool = True) -> np.ndarray:
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
    
    # Draw piece slots and their internal relative grids.
    # When scan_piece_slots is False, avoid expensive re-analysis and just draw
    # the cached slot boxes/labels so the UI remains live without nonstop scanning.
    for slot_idx, slot in enumerate(config.PIECE_SLOTS):
        cv2.rectangle(vis, (slot.x, slot.y), (slot.x + slot.width, slot.y + slot.height), (255, 0, 0), 1)
        if not scan_piece_slots:
            if pieces and slot_idx < len(pieces) and pieces[slot_idx] is not None:
                piece = pieces[slot_idx]
                cv2.putText(vis, f"{piece.width}x{piece.height}", (slot.x + 6, slot.y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            continue

        piece_region = frame[slot.y:slot.y+slot.height, slot.x:slot.x+slot.width]
        if piece_region.size == 0: continue
        
        # v5.6: Call get_piece_grid and reuse its exact metadata
        grid_data = get_piece_grid(piece_region)
        if grid_data is None: continue
        
        grid_5x5, _, _, d, anchor_x, anchor_y, actual_cols, actual_rows, bbox_cols, bbox_rows = grid_data
        
        # anchor_x/y is the center of the (start_c, start_r) cell in grid_5x5,
        # where start_r/c come from the bbox-estimated dims.
        start_r = (5 - bbox_rows) // 2
        start_c = (5 - bbox_cols) // 2
        
        # Draw dots for all cells within the filled extent
        filled_r = np.where(np.any(grid_5x5, axis=1))[0]
        filled_c = np.where(np.any(grid_5x5, axis=0))[0]
        if len(filled_r) == 0:
            continue
        
        r_min, r_max = filled_r[0], filled_r[-1]
        c_min, c_max = filled_c[0], filled_c[-1]
        
        block_count = 0
        for r_idx in range(r_min, r_max + 1):
            for c_idx in range(c_min, c_max + 1):
                # Map grid_5x5 (r_idx, c_idx) -> pixel position in piece_region
                px = int(anchor_x + (c_idx - start_c) * d)
                py = int(anchor_y + (r_idx - start_r) * d)
                # Convert from piece_region coords to frame coords
                abs_x = slot.x + px
                abs_y = slot.y + py
                
                if grid_5x5[r_idx, c_idx]:
                    cv2.circle(vis, (abs_x, abs_y), 5, (0, 255, 0), -1)  # green filled
                    block_count += 1
                else:
                    cv2.circle(vis, (abs_x, abs_y), 3, (100, 100, 100), -1)  # gray empty
        
        # Label with actual dimensions
        label = f"{actual_cols}x{actual_rows} ({block_count})"
        cv2.putText(vis, label, (slot.x + 6, slot.y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
        
        # Get identification details (v5.6: Correct 8-tuple unpacking)
        grid_data = get_piece_grid(piece_region)
        if grid_data is not None:
            grid_5x5, _, _, _d, _ax, _ay, cols, rows, _bcols, _brows = grid_data
            final_grid, score, name, match_info = template_manager.match_and_validate(
                grid_5x5,
                expected_dims=(rows, cols),
            )
            
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


def visualize_drag(frame: np.ndarray, piece: Piece, move: Move, 
                   start_xy: Tuple[int, int], click_xy: Tuple[int, int], end_xy: Tuple[int, int],
                   mult_x: float = 0.0, mult_y: float = 0.0) -> np.ndarray:
    """
    Visualize the drag move with:
    - Piece Blueprint: Transparent overlay of the piece at its target position.
    - Cell Highlights: Circles on the exact cells being filled.
    - Red Cross: Actual cursor destination (click_xy, where the mouse clicks).
    - Yellow Cross: Intended piece anchor center (end_xy).
    - v5.0: Enhanced labels for offsets and target coordinates.
    """
    vis = frame.copy()
    x1, y1 = config.GRID_TOP_LEFT
    
    # 1. DRAW PIECE BLUEPRINT & CELL HIGHLIGHTS
    for dr, dc in piece.cells:
        r, c = move.row + dr, move.col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            cx = int(x1 + c * config.CELL_WIDTH + config.CELL_WIDTH / 2.0)
            cy = int(y1 + r * config.CELL_HEIGHT + config.CELL_HEIGHT / 2.0)
            
            overlay = vis.copy()
            bw, bh = int(config.CELL_WIDTH * 0.8), int(config.CELL_HEIGHT * 0.8)
            cv2.rectangle(overlay, (cx - bw//2, cy - bh//2), (cx + bw//2, cy + bh//2), (0, 165, 255), -1)
            cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
            cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)

    # 2. DRAW DRAG PATH & CROSSES
    cv2.line(vis, start_xy, click_xy, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Yellow Cross: Visual piece center (anchor point)
    cv2.drawMarker(vis, end_xy, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
    
    # v5.0: Piece Center Labels
    anchor_dr, anchor_dc = piece.anchor_offset
    center_r = move.row + anchor_dr + piece.height/2.0
    center_c = move.col + anchor_dc + piece.width/2.0
    
    cv2.putText(vis, f"CENTER: {center_r:.1f}, {center_c:.1f}", (end_xy[0] + 15, end_xy[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, f"TARGET: ({move.row}, {move.col})", (end_xy[0] + 15, end_xy[1] + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Red Cross: Mouse Cursor (where it clicks + offset)
    cv2.drawMarker(vis, click_xy, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    
    # v5.0: Cursor and Offset Labels
    cv2.putText(vis, f"CURSOR (X_MULT: {mult_x:.2f})", (click_xy[0] + 20, click_xy[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis, f"OFFSETS (Y_MULT: {mult_y:.2f})", (click_xy[0] + 20, click_xy[1] + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return vis
