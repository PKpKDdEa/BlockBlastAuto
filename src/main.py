"""
Main orchestration loop for Block Blast automation.
"""
import time
import cv2
from window_capture import WindowCapture
from vision import read_board, read_pieces, visualize_detection, visualize_piece_analysis, draw_pause_status
from model import Board
from solver import best_move
from controller import drag_piece
from config import config
import keyboard


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
    print(f"Controls: Press [{config.HOTKEY_PAUSE.upper()}] to Pause/Resume, [{config.HOTKEY_AUTO_TOGGLE.upper()}] to toggle Auto-Play.")
    
    # Global state for controls
    state = {
        "paused": False,
        "auto_play": config.AUTO_PLAY,
        "pieces_last_seen": None,
        "diag_frame": None
    }
    
    def on_toggle_pause():
        state["paused"] = not state["paused"]
        status = "PAUSED" if state["paused"] else "RESUMED"
        print(f"\n[BOT {status}] - Controls suspended.")
        
    def on_toggle_auto():
        state["auto_play"] = not state["auto_play"]
        status = "AUTO-PLAY" if state["auto_play"] else "OBSERVATION-ONLY"
        print(f"\n[MODE: {status}] - Bot will {'now drag pieces' if state['auto_play'] else 'only observe'}.")
    
    keyboard.add_hotkey(config.HOTKEY_PAUSE, on_toggle_pause)
    keyboard.add_hotkey(config.HOTKEY_AUTO_TOGGLE, on_toggle_auto)
    
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
            
            # Show Live Vision (Always active)
            vis = visualize_detection(frame, board, pieces)
            draw_pause_status(vis, state["paused"])
            # Indicate current mode
            mode_text = "AUTO-PLAY" if state["auto_play"] else "OBSERVING ONLY"
            mode_color = (0, 255, 0) if state["auto_play"] else (0, 165, 255)
            cv2.putText(vis, mode_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            cv2.imshow("Bot Vision", vis)
            
            # Show Stable Piece Analysis (Update only on turn change)
            has_pieces = any(p is not None for p in pieces)
            if state["pieces_last_seen"] is None or pieces != state["pieces_last_seen"]:
                state["diag_frame"] = visualize_piece_analysis(frame, pieces)
                state["pieces_last_seen"] = pieces
            
            if state["diag_frame"] is not None:
                cv2.imshow("Piece Analysis", state["diag_frame"])
            
            cv2.waitKey(1)
            
            # If paused, skip the rest of the logic
            if state["paused"]:
                time.sleep(0.1)
                continue
                
            if not has_pieces:
                if config.DEBUG:
                    print("Waiting for pieces...")
                time.sleep(1.0)
                continue
            
            # 2. Solve Best Move (Sequence optimized)
            if config.DEBUG:
                print("\nBoard State:")
            if not state["auto_play"]:
                # In observation mode, we still show the best move for feedback
                move = best_move(board, pieces)
                if move is not None:
                    piece = pieces[move.piece_index]
                    from controller import cell_center
                    end_xy = cell_center(move.row, move.col)
                    # Show intended move with a small dot
                    cv2.circle(vis, end_xy, 10, (255, 255, 0), 2)
                    cv2.imshow("Bot Vision", vis)
                time.sleep(0.1)
                continue
                
            # 3. Solve & Execute (Auto-Play ONLY)
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
            
            if config.DEBUG:
                from controller import piece_slot_center, cell_center
                from vision import visualize_drag
                start_xy = piece_slot_center(move.piece_index)
                end_xy = cell_center(move.row, move.col)
                # Apply the same logic as controller.py for visualization
                y_top, y_bottom = config.GRID_TOP_LEFT[1], config.GRID_BOTTOM_RIGHT[1]
                progress = max(0, min(1, (y_bottom - end_xy[1]) / (y_bottom - y_top)))
                current_offset = int(config.DRAG_OFFSET_Y_BOTTOM + progress * (config.DRAG_OFFSET_Y_TOP - config.DRAG_OFFSET_Y_BOTTOM))
                click_xy = (end_xy[0], end_xy[1] + current_offset)
                
                vis_drag = visualize_drag(vis, move, start_xy, click_xy, end_xy)
                cv2.imshow("Bot Vision", vis_drag)
                cv2.waitKey(1)
                
            drag_piece(piece, move.row, move.col)
            
            # Verification: Check if piece actually landed
            time.sleep(1.0) # Wait for animation to finish
            frame_after = capture.capture_frame()
            board_after = read_board(frame_after)
            
            # Check the cells where the piece should be
            missed = False
            for dr, dc in piece.cells:
                r, c = move.row + dr, move.col + dc
                if r < 8 and c < 8 and board_after.grid[r, c] == 0:
                    missed = True
                    break
            
            if missed:
                print(f"  [!!!] PLACEMENT FAILURE: Piece {move.piece_index} missed ({move.row}, {move.col})")
                print(f"        Action: Adjust DRAG_OFFSET_Y values or Re-Calibrate Slots.")
            else:
                print(f"  [OK] Piece {move.piece_index} placed successfully at ({move.row}, {move.col})")
                
                # Success-Based Learning (Phase 8)
                if piece.is_new and piece.raw_mask is not None:
                    from vision import template_manager
                    template_manager.learn_pattern(piece.raw_mask)
                    print(f"  [LEARNING] Successfully verified and saved new pattern for piece {piece.id}")
            
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
