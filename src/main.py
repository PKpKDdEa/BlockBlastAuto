"""
Main orchestration loop for Block Blast automation.
"""
import time
import cv2
from window_capture import WindowCapture
from vision import read_board, read_pieces, visualize_detection
from model import Board
from solver import best_move
from controller import drag_piece
from config import config


from optimizer import WeightOptimizer

def main():
    """Main bot loop."""
    print("=" * 50)
    print("Block Blast Automation Bot (Enhanced)")
    print("=" * 50)
    print()
    
    # Initialize components
    config.load()
    optimizer = WeightOptimizer()
    current_weights = optimizer.get_current_weights()
    print("Loaded heuristic weights:")
    import json
    print(json.dumps(current_weights, indent=4))
    
    print("\nLooking for LDPlayer window...")
    capture = WindowCapture()
    
    if not capture.find_window():
        print("ERROR: Could not find LDPlayer window!")
        return
    
    print(f"✓ Found window: {capture.window_title}")
    
    # Simulation state
    sim_board = Board(config.GRID_ROWS, config.GRID_COLS)
    move_count = 0
    
    print("\nStarting automation loop...")
    
    try:
        while True:
            loop_start = time.time()
            
            # 1. Capture and Detect
            frame = capture.capture_frame()
            if frame is None:
                continue
                
            # Read actual board and pieces
            board = read_board(frame)
            pieces = read_pieces(frame)
            
            # Sync internal tracking with reality if needed
            # (Though the solver uses the frame-based detection)
            
            if not any(p is not None for p in pieces):
                if config.DEBUG:
                    print("Waiting for pieces...")
                time.sleep(1.0)
                continue
            
            # 2. Solve Best Move (Sequence optimized)
            move = best_move(board, pieces)
            
            if move is None:
                print("\n" + "=" * 50)
                print("GAME OVER: No legal moves!")
                print(f"Total Score: {board.total_score}")
                print(f"Total Moves: {move_count}")
                
                # SELF-IMPROVEMENT: Record and evolve
                optimizer.record_game(board.total_score, move_count, current_weights)
                new_w = optimizer.evolve_weights()
                optimizer.apply_weights(new_w)
                print("Evolved weights for next run.")
                print("=" * 50)
                break
            
            # 3. Execute Move
            piece = pieces[move.piece_index]
            print(f"Move #{move_count + 1}: Placing {piece.width}x{piece.height} at ({move.row}, {move.col}) [Streak: {board.combo_streak}]")
            drag_piece(move.piece_index, move.row, move.col)
            
            move_count += 1
            
            # 3. Wait for animation to finish
            from vision import wait_for_board_stable
            print("Done. Waiting for animation...")
            wait_for_board_stable(capture)
            
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
        print(f"Total moves made: {move_count}")
        print(f"Final Score: {board.total_score}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if config.DEBUG:
            cv2.destroyAllWindows()
        print("\nShutting down...")


if __name__ == "__main__":
    main()
