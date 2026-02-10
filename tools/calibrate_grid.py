"""
Interactive calibration tool for setting up grid and piece slot coordinates.
"""
import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from window_capture import WindowCapture


class CalibrationTool:
    """Interactive tool for calibrating game coordinates."""
    
    def __init__(self):
        self.frame = None
        self.points = []
        self.current_step = 0
        self.steps = [
            "Click on TOP-LEFT corner of the grid",
            "Click on BOTTOM-RIGHT corner of the grid",
            "Click on center of FIRST piece slot (left)",
            "Click on center of SECOND piece slot (middle)",
            "Click on center of THIRD piece slot (right)",
            "Step A: Click the exact CENTER of a block in the tray",
            "Step B: Click the exact CENTER of the block NEXT to it",
        ]
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"✓ Point {len(self.points)}: ({x}, {y})")
            
            # Draw marker
            cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.frame, f"{len(self.points)}", (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Calibration", self.frame)
            
            self.current_step += 1
    
    def run(self):
        """Run calibration process."""
        print("=" * 60)
        print("Block Blast Calibration Tool")
        print("=" * 60)
        print()
        
        # Capture window
        print("Looking for LDPlayer window...")
        capture = WindowCapture()
        
        if not capture.find_window():
            print("ERROR: Could not find LDPlayer window!")
            return
        
        print(f"✓ Found window: {capture.window_title}")
        print()
        
        # Capture frame
        self.frame = capture.capture_frame()
        if self.frame is None:
            print("ERROR: Failed to capture frame!")
            return
        
        print("Frame captured successfully!")
        print()
        print("Instructions:")
        print("-" * 60)
        for i, step in enumerate(self.steps, 1):
            print(f"{i}. {step}")
        print("-" * 60)
        print()
        print("Click on the image to mark points. Press 'q' when done.")
        print()
        
        # Show frame
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        cv2.imshow("Calibration", self.frame)
        
        # Wait for user input
        while len(self.points) < len(self.steps):
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        # --- NEW: Live Preview Loop ---
        if len(self.points) == len(self.steps):
            print("\n" + "-" * 60)
            print("ENTERING PREVIEW MODE")
            print("Press 'S' to SAVE and generate config.")
            print("Press 'R' to RESET Step A & B (points 6 & 7) and click again.")
            print("-" * 60)
            
            while True:
                preview = self.frame.copy()
                
                # Calculate pitch from current points 6 and 7
                p6 = self.points[5]
                p7 = self.points[6]
                pitch = int(np.sqrt((p7[0]-p6[0])**2 + (p7[1]-p6[1])**2))
                
                # Draw the 5x5 grid preview for each slot
                piece_slots = self.points[2:5]
                for cx, cy in piece_slots:
                    # Draw blue square slot first (overlap prevention)
                    min_dist = 999
                    for i in range(len(piece_slots) - 1):
                        dist = abs(piece_slots[i+1][0] - piece_slots[i][0])
                        min_dist = min(min_dist, dist)
                    
                    size = int(pitch * 5.0)
                    if min_dist < size: size = min_dist - 2
                    
                    cv2.rectangle(preview, (cx - size//2, cy - size//2), 
                                 (cx + size//2, cy + size//2), (255, 0, 0), 1)
                    
                    # Draw the 5x5 probe grid (Zero Spacing)
                    for r in range(5):
                        for c in range(5):
                            px = cx + (c - 2) * pitch
                            py = cy + (r - 2) * pitch
                            
                            # Sampling box
                            m = pitch // 2
                            cv2.rectangle(preview, (px-m, py-m), (px+m, py+m), (80, 80, 80), 1)
                            cv2.circle(preview, (px, py), 2, (0, 0, 255), -1)

                cv2.imshow("Calibration", preview)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('s'):
                    break
                elif key == ord('r'):
                    print("Resetting points 6 & 7. Click them again!")
                    self.points = self.points[:5]
                    self.current_step = 5
                    self.frame = capture.capture_frame() # Refresh frame
                    cv2.imshow("Calibration", self.frame)
                    while len(self.points) < len(self.steps):
                        cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        
        # Generate config
        if len(self.points) >= 5:
            self.generate_config()
        else:
            print("\nCalibration incomplete. Please mark all 5 points.")
    
    def generate_config(self):
        """Generate configuration code from marked points."""
        grid_tl = self.points[0]
        grid_br = self.points[1]
        piece_slots = self.points[2:5]
        p6 = self.points[5]
        p7 = self.points[6]
        tray_cell_w = int(np.sqrt((p7[0]-p6[0])**2 + (p7[1]-p6[1])**2))
        tray_cell_h = tray_cell_w # Assume square blocks
        
        print("\n" + "=" * 60)
        print("Calibration Complete!")
        print("=" * 60)
        print()
        print("Add the following to your config.py:")
        print()
        print("-" * 60)
        print(f"GRID_TOP_LEFT = {grid_tl}")
        print(f"GRID_BOTTOM_RIGHT = {grid_br}")
        print(f"TRAY_CELL_SIZE = ({tray_cell_w}, {tray_cell_h})")
        print()
        # Calculate piece slot regions (Strict 5x5 bounds to avoid overlap)
        # 1. Base dimensions from single tray cell
        size = int(tray_cell_w * 5.0)
        
        # 2. Prevent horizontal overlap based on distance between centers
        min_dist = 999
        for i in range(len(piece_slots) - 1):
            dist = abs(piece_slots[i+1][0] - piece_slots[i][0])
            min_dist = min(min_dist, dist)
            
        if min_dist < size:
            size = min_dist - 2 # 2px gap
            print(f"! Warning: Square slot size capped to {size} to prevent overlap (min distance between slots: {min_dist})")

        slot_width = size
        slot_height = size
        
        print("TRAY_SLOT_CENTERS = [")
        for x, y in piece_slots:
            print(f"    ({x}, {y}),")
        print("]")
        
        print("\nPIECE_SLOTS = [")
        for i, (x, y) in enumerate(piece_slots):
            slot_x = x - slot_width // 2
            slot_y = y - slot_height // 2
            print(f"    GameRegion(x={slot_x}, y={slot_y}, width={slot_width}, height={slot_height}),")
        print("]")
        print("-" * 60)
        print()
        
        # Save to file
        config_path = "calibration_config.txt"
        with open(config_path, 'w') as f:
            f.write(f"GRID_TOP_LEFT = {grid_tl}\n")
            f.write(f"GRID_BOTTOM_RIGHT = {grid_br}\n")
            f.write(f"TRAY_CELL_SIZE = ({tray_cell_w}, {tray_cell_h})\n")
            f.write("\nTRAY_SLOT_CENTERS = [\n")
            for x, y in piece_slots:
                f.write(f"    ({x}, {y}),\n")
            f.write("]\n")
            f.write("\nPIECE_SLOTS = [\n")
            for i, (x, y) in enumerate(piece_slots):
                slot_x = x - slot_width // 2
                slot_y = y - slot_height // 2
                f.write(f"    GameRegion(x={slot_x}, y={slot_y}, width={slot_width}, height={slot_height}),\n")
            f.write("]\n")
        
        print(f"✓ Configuration saved to: {config_path}")
        print()


if __name__ == "__main__":
    tool = CalibrationTool()
    tool.run()
