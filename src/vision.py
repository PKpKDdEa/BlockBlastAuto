"""
Computer vision module for detecting board state and pieces.
"""
import cv2
import numpy as np
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
            
            # Classify cell as filled or empty
            is_filled = classify_cell(patch)
            board.grid[row, col] = 1 if is_filled else 0
    
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
        else:
            # Empty slot - create a dummy piece that won't match anything
            # This prevents index errors when solver expects 3 pieces
            pieces.append(None)
    
    # Filter out None values
    pieces = [p for p in pieces if p is not None]
    
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
    
    # Create mask for colored regions (piece cells)
    # Adjust these thresholds based on actual piece appearance
    lower_bound = np.array([0, 30, 80])  # Low saturation/brightness = background
    upper_bound = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Clean up mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Resize to canonical small grid (e.g., 5x5)
    # This makes piece matching easier
    canonical_size = 5
    mask_resized = cv2.resize(mask, (canonical_size, canonical_size), interpolation=cv2.INTER_NEAREST)
    
    # Threshold to binary
    mask_binary = (mask_resized > 127).astype(np.uint8)
    
    # Check if mask is empty
    if np.sum(mask_binary) == 0:
        return None
    
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


if __name__ == "__main__":
    # Test vision module
    from window_capture import WindowCapture
    
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
