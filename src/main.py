"""
Main orchestration loop for Block Blast automation.
"""
import time
import cv2
from window_capture import WindowCapture
from vision import read_board, read_pieces, visualize_detection
from solver import best_move
from controller import drag_piece
from config import config


def main():
    """Main bot loop."""
    print("=" * 50)
    print("Block Blast Automation Bot")
    print("=" * 50)
    print()
    
    # Initialize window capture
    print("Looking for LDPlayer window...")
    capture = WindowCapture()
    
    if not capture.find_window():
        print("ERROR: Could not find LDPlayer window!")
        print("Make sure LDPlayer is running with Block Blast open.")
        return
    
    print(f"✓ Found window: {capture.window_title}")
    print()
    
    # Calculate loop delay for target FPS
    loop_delay = 1.0 / config.CAPTURE_FPS
    
    print("Starting automation loop...")
    print("Press Ctrl+C to stop, or move mouse to top-left corner (failsafe)")
    print()
    
    move_count = 0
    
    try:
        while True:
            loop_start = time.time()
            
            # Capture frame
            frame = capture.capture_frame()
            if frame is None:
                print("WARNING: Failed to capture frame, retrying...")
                time.sleep(1)
                continue
            
            # Detect board state
            board = read_board(frame)
            
            # Detect available pieces
            pieces = read_pieces(frame)
            
            if len(pieces) == 0:
                if config.DEBUG:
                    print("No pieces detected, waiting...")
                time.sleep(0.5)
                continue
            
            if config.DEBUG:
                print(f"\nMove #{move_count + 1}")
                print("Board state:")
                print(board)
                print(f"Detected {len(pieces)} pieces")
            
            # Show visualization if debug enabled
            if config.DEBUG:
                vis = visualize_detection(frame, board, pieces)
                cv2.imshow("Bot Vision", vis)
                cv2.waitKey(1)
            
            # Compute best move
            move = best_move(board, pieces)
            
            if move is None:
                print("\n" + "=" * 50)
                print("GAME OVER: No legal moves available!")
                print(f"Total moves made: {move_count}")
                print("=" * 50)
                break
            
            # Execute move
            print(f"Move #{move_count + 1}: Placing piece {move.piece_index} at ({move.row}, {move.col}) [score: {move.score:.2f}]")
            drag_piece(move.piece_index, move.row, move.col)
            
            move_count += 1
            
            # Wait for animation and next frame
            time.sleep(0.2)  # Wait for piece placement animation
            
            # Maintain target FPS
            elapsed = time.time() - loop_start
            if elapsed < loop_delay:
                time.sleep(loop_delay - elapsed)
    
    except KeyboardInterrupt:
        print("\n\nBot stopped by user.")
        print(f"Total moves made: {move_count}")
    
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if config.DEBUG:
            cv2.destroyAllWindows()
        print("\nShutting down...")


if __name__ == "__main__":
    main()
