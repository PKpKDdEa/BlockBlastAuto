"""
Tool for collecting and saving piece templates.
"""
import cv2
import numpy as np
import os
from window_capture import WindowCapture
from vision import detect_piece_mask
from config import config


def collect_pieces():
    """Collect piece templates from current game state."""
    print("=" * 60)
    print("Piece Template Collection Tool")
    print("=" * 60)
    print()
    
    # Create templates directory
    templates_dir = "templates"
    os.makedirs(templates_dir, exist_ok=True)
    
    # Capture window
    print("Looking for LDPlayer window...")
    capture = WindowCapture()
    
    if not capture.find_window():
        print("ERROR: Could not find LDPlayer window!")
        return
    
    print(f"✓ Found window: {capture.window_title}")
    print()
    
    piece_count = 0
    
    print("Instructions:")
    print("- Position different pieces in the piece slots")
    print("- Press SPACE to capture current pieces")
    print("- Press 'q' to quit")
    print()
    
    while True:
        # Capture frame
        frame = capture.capture_frame()
        if frame is None:
            print("WARNING: Failed to capture frame")
            continue
        
        # Show frame
        display = frame.copy()
        
        # Draw piece slots
        for i, slot in enumerate(config.PIECE_SLOTS):
            cv2.rectangle(display, (slot.x, slot.y), 
                         (slot.x + slot.width, slot.y + slot.height),
                         (0, 255, 0), 2)
            cv2.putText(display, f"Slot {i}", (slot.x, slot.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(display, "Press SPACE to capture, 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Pieces collected: {piece_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Piece Collection", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Capture pieces
            print(f"\nCapturing pieces #{piece_count + 1}...")
            
            for slot_idx, slot in enumerate(config.PIECE_SLOTS):
                # Extract piece region
                piece_region = frame[slot.y:slot.y+slot.height, slot.x:slot.x+slot.width]
                
                # Detect mask
                mask = detect_piece_mask(piece_region)
                
                if mask is not None and np.sum(mask) > 0:
                    # Save piece image and mask
                    piece_filename = f"piece_{piece_count:03d}_slot{slot_idx}.png"
                    mask_filename = f"piece_{piece_count:03d}_slot{slot_idx}_mask.png"
                    
                    cv2.imwrite(os.path.join(templates_dir, piece_filename), piece_region)
                    cv2.imwrite(os.path.join(templates_dir, mask_filename), mask * 255)
                    
                    print(f"  ✓ Saved slot {slot_idx}: {piece_filename}")
                else:
                    print(f"  - Slot {slot_idx}: Empty or not detected")
            
            piece_count += 1
            print()
        
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"\n✓ Collection complete! Saved {piece_count} piece sets to '{templates_dir}/'")


if __name__ == "__main__":
    collect_pieces()
