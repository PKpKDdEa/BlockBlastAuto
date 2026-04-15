"""
Mouse controller for executing moves via drag-and-drop.
"""
import pyautogui
import time
import random
import subprocess
import re
import os
import shutil
from typing import Tuple, List
from config import config
from model import Piece
from window_capture import find_window_handle, get_window_rect


# Configure pyautogui
pyautogui.PAUSE = 0.05  # Small pause between actions
pyautogui.FAILSAFE = True  # Move mouse to corner to abort


def _get_window_rect_safe():
    """Return current game window rect as (x, y, w, h), or None."""
    try:
        hwnd = find_window_handle(config.WINDOW_TITLE)
        if not hwnd:
            return None
        return get_window_rect(hwnd)
    except Exception:
        return None


def _to_screen_xy(x: int, y: int) -> Tuple[int, int]:
    """Convert configured coordinates to absolute screen coordinates for pyautogui."""
    if config.COORDINATE_SPACE == "screen-absolute":
        return int(x), int(y)

    wr = _get_window_rect_safe()
    if wr is None:
        return int(x), int(y)

    wx, wy, _ww, _wh = wr
    return int(wx + x), int(wy + y)


def _run_adb(args):
    adb_path = _resolve_adb_path()
    if not adb_path:
        raise FileNotFoundError(
            "ADB backend selected, but adb.exe was not found. "
            "Set ADB_PATH in calibration_config.txt/config, add adb to PATH, "
            "or switch CONTROL_BACKEND to pyautogui."
        )

    cmd = [adb_path]
    if config.ADB_DEVICE_ID:
        cmd += ["-s", config.ADB_DEVICE_ID]
    cmd += args
    timeout_s = 3.0
    if len(args) >= 7 and args[:3] == ["shell", "input", "swipe"]:
        try:
            duration_ms = int(args[-1])
            timeout_s = max(6.0, duration_ms / 1000.0 + 5.0)
        except Exception:
            timeout_s = 6.0
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ADB command timed out after {timeout_s:.1f}s: {' '.join(cmd)}"
        ) from exc


def _resolve_adb_path() -> str:
    """Resolve adb.exe from config, PATH, or common Windows locations."""
    configured = (config.ADB_PATH or "").strip()
    if configured:
        if os.path.isfile(configured):
            return configured
        found = shutil.which(configured)
        if found:
            return found

    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    user_profile = os.environ.get("USERPROFILE", "")
    android_home = os.environ.get("ANDROID_HOME", "")
    android_sdk_root = os.environ.get("ANDROID_SDK_ROOT", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

    for sdk_root in [android_home, android_sdk_root, os.path.join(local_app_data, "Android", "Sdk")]:
        if sdk_root:
            candidates.append(os.path.join(sdk_root, "platform-tools", "adb.exe"))

    candidates += [
        os.path.join(program_files, "MuMuPlayerGlobal-12.0", "shell", "adb.exe"),
        os.path.join(program_files_x86, "MuMuPlayer-12.0", "shell", "adb.exe"),
        os.path.join(program_files, "Netease", "MuMuPlayerGlobal-12.0", "shell", "adb.exe"),
        os.path.join(program_files, "BlueStacks_nxt", "HD-Adb.exe"),
        os.path.join(program_files_x86, "BlueStacks", "HD-Adb.exe"),
        os.path.join(program_files, "Microvirt", "MEmu", "adb.exe"),
        os.path.join(user_profile, "AppData", "Local", "Android", "Sdk", "platform-tools", "adb.exe"),
    ]

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            config.ADB_PATH = candidate
            return candidate

    return ""


def _run_adb_host(args, timeout_s: float = 5.0):
    """Run an ADB host command without device-specific shell args."""
    adb_path = _resolve_adb_path()
    if not adb_path:
        return None
    cmd = [adb_path] + args
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except Exception:
        return None


def _list_adb_devices() -> List[str]:
    """Return connected device ids from `adb devices`."""
    res = _run_adb_host(["devices"], timeout_s=5.0)
    if res is None:
        return []
    devices = []
    for line in (res.stdout or "").splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def ensure_control_backend_ready() -> bool:
    """Ensure selected input backend is usable; auto-fallback if ADB is missing."""
    if config.CONTROL_BACKEND != "adb":
        return True

    adb_path = _resolve_adb_path()
    if not adb_path:
        print("[CONTROL] adb.exe not found; falling back to pyautogui backend.")
        config.CONTROL_BACKEND = "pyautogui"
        return True

    _run_adb_host(["start-server"], timeout_s=5.0)
    devices = _list_adb_devices()
    if len(devices) == 1 and not config.ADB_DEVICE_ID:
        config.ADB_DEVICE_ID = devices[0]
    if config.DEBUG:
        print(f"[CONTROL] Using ADB executable: {config.ADB_PATH}")
        if devices:
            print(f"[CONTROL] Connected ADB devices: {', '.join(devices)}")

    if not devices:
        print("[CONTROL] No ADB device is connected. MuMu supports ADB, but the emulator must expose a device in `adb devices`. Falling back to pyautogui backend.")
        config.CONTROL_BACKEND = "pyautogui"
    return True


def _get_adb_screen_size() -> Tuple[int, int]:
    """Read Android device physical size via ADB (wm size)."""
    try:
        res = _run_adb(["shell", "wm", "size"])
        txt = (res.stdout or "") + "\n" + (res.stderr or "")
        m = re.search(r"(\d+)x(\d+)", txt)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    # Safe fallback often used by emulators
    return 1080, 1920


def _to_adb_xy(x: int, y: int) -> Tuple[int, int]:
    """Map configured coordinates to ADB device coordinates."""
    wr = _get_window_rect_safe()
    dev_w, dev_h = _get_adb_screen_size()

    # If already using absolute desktop coords, normalize into window first
    if config.COORDINATE_SPACE == "screen-absolute":
        if wr is not None:
            wx, wy, ww, wh = wr
            rx = max(0, min(ww - 1, x - wx))
            ry = max(0, min(wh - 1, y - wy))
        else:
            rx, ry = x, y
            ww, wh = dev_w, dev_h
    else:
        rx, ry = x, y
        if wr is not None:
            _wx, _wy, ww, wh = wr
        else:
            ww, wh = dev_w, dev_h

    sx = int(max(0, min(dev_w - 1, round(rx * (dev_w / float(max(1, ww)))))))
    sy = int(max(0, min(dev_h - 1, round(ry * (dev_h / float(max(1, wh)))))))
    return sx, sy


def cell_center(row: int, col: int) -> Tuple[int, int]:
    """
    Convert grid coordinates to screen coordinates (cell center).
    """
    grid_width = config.GRID_BOTTOM_RIGHT[0] - config.GRID_TOP_LEFT[0]
    grid_height = config.GRID_BOTTOM_RIGHT[1] - config.GRID_TOP_LEFT[1]
    cell_w = grid_width / config.GRID_COLS
    cell_h = grid_height / config.GRID_ROWS
    
    x = int(config.GRID_TOP_LEFT[0] + (col + 0.5) * cell_w)
    y = int(config.GRID_TOP_LEFT[1] + (row + 0.5) * cell_h)
    
    # Add small randomness for non-ADB mode only
    if config.CONTROL_BACKEND == "pyautogui" and config.MOUSE_MOVE_RANDOMNESS > 0:
        x += random.randint(-config.MOUSE_MOVE_RANDOMNESS, config.MOUSE_MOVE_RANDOMNESS)
        y += random.randint(-config.MOUSE_MOVE_RANDOMNESS, config.MOUSE_MOVE_RANDOMNESS)
    
    return (x, y)


def piece_slot_center(piece_index: int) -> Tuple[int, int]:
    """
    Get screen coordinates for piece slot center.
    
    Args:
        piece_index: Index of piece slot (0-2)
    
    Returns:
        Tuple of (x, y) screen coordinates
    """
    if piece_index < 0 or piece_index >= len(config.PIECE_SLOTS):
        raise ValueError(f"Invalid piece index: {piece_index}")
    
    slot = config.PIECE_SLOTS[piece_index]
    x = slot.x + slot.width // 2
    y = slot.y + slot.height // 2
    
    return (x, y)


def move_mouse_and_drag(start_xy: Tuple[int, int], end_xy: Tuple[int, int], duration_ms: int = None) -> None:
    """
    Move mouse and perform drag operation.
    
    Args:
        start_xy: Starting (x, y) coordinates
        end_xy: Ending (x, y) coordinates
        duration_ms: Duration of drag in milliseconds
    """
    if config.CONTROL_BACKEND == "adb":
        duration = int(duration_ms if duration_ms is not None else config.ADB_DRAG_DURATION_MS)
        sx, sy = _to_adb_xy(int(start_xy[0]), int(start_xy[1]))
        ex, ey = _to_adb_xy(int(end_xy[0]), int(end_xy[1]))
        _run_adb(["shell", "input", "swipe", str(sx), str(sy), str(ex), str(ey), str(duration)])
        time.sleep(0.08)
        return

    if duration_ms is None:
        duration_ms = config.MOUSE_DRAG_DURATION_MS

    duration_sec = duration_ms / 1000.0
    start_screen = _to_screen_xy(int(start_xy[0]), int(start_xy[1]))
    end_screen = _to_screen_xy(int(end_xy[0]), int(end_xy[1]))

    # Move to start position
    pyautogui.moveTo(start_screen[0], start_screen[1], duration=0.2)
    pyautogui.mouseDown()
    time.sleep(0.1) # Wait for click to register

    # Perform drag
    pyautogui.moveTo(end_screen[0], end_screen[1], duration=duration_sec)
    time.sleep(0.1) # Wait for movement to finish
    pyautogui.mouseUp()

    time.sleep(0.1)


def drag_piece(piece: Piece, target_row: int, target_col: int) -> None:
    """
    Drag a piece from its slot to a target position on the board.
    Optimized for MuMu emulator with dynamic line-based offsets.
    """
    # Use the ACTUAL detected coordinates in the tray for the starting grab
    slot = config.PIECE_SLOTS[piece.id]
    start_pos = (int(slot.x + piece.tray_cx), int(slot.y + piece.tray_cy))

    # Target screen positions for the pieces internal center
    anchor_center_x, anchor_center_y = cell_center(target_row, target_col)
    anchor_dr, anchor_dc = piece.anchor_offset
    piece_target_x = anchor_center_x + int(anchor_dc * config.CELL_WIDTH)
    piece_target_y = anchor_center_y + int(anchor_dr * config.CELL_HEIGHT)

    if config.DRAG_MODEL == "ratio":
        # Ratio model (conversation reference):
        # Cursor movement ≈ piece movement * ratio, with fixed lift.
        ratio = float(config.DRAG_DISTANCE_RATIO)
        lift_y = int(config.DRAG_LIFT_Y)

        dx = piece_target_x - start_pos[0]
        dy = piece_target_y - start_pos[1]

        dest_x = int(round(start_pos[0] + dx * ratio))
        dest_y = int(round(start_pos[1] + dy * ratio + lift_y))
        end_pos = (dest_x, dest_y)
        dx_cells = abs(dx) / max(1.0, config.CELL_WIDTH)
        dy_cells = abs(dy) / max(1.0, config.CELL_HEIGHT)
        model_info = f"ratio={ratio:.3f}, lift={lift_y}px"
    else:
        # Legacy table model
        start_x, start_y = piece_slot_center(piece.id)
        dx_cells = round(abs(piece_target_x - start_x) / config.CELL_WIDTH)
        dy_cells = round(abs(start_y - piece_target_y) / config.CELL_HEIGHT)

        mult_x = config.DISPLACEMENT_X_TABLE.get(min(8, int(dx_cells)), 0.6)
        mult_y = config.DISPLACEMENT_Y_TABLE.get(min(8, int(dy_cells)), 1.6)

        y_offset = int(mult_y * config.CELL_HEIGHT)
        x_pull = int(mult_x * config.CELL_WIDTH)

        dest_x = piece_target_x
        dest_y = piece_target_y + y_offset

        if piece_target_x > start_x: # Moving Right
            dest_x -= x_pull
        elif piece_target_x < start_x: # Moving Left
            dest_x += x_pull

        end_pos = (dest_x, dest_y)
        model_info = f"table mult_x={mult_x:.2f}, mult_y={mult_y:.2f}"
    
    if config.DEBUG:
        print(f"Dragging piece {piece.id} to board ({target_row}, {target_col})")
        print(f"  Backend: {config.CONTROL_BACKEND}")
        print(f"  Distance Cells: X={dx_cells:.2f}, Y={dy_cells:.2f}")
        print(f"  Model: {model_info}")
        print(f"  Final Cursor Target: {end_pos}")
    
    move_mouse_and_drag(start_pos, end_pos)


def click_position(x: int, y: int) -> None:
    """
    Click at a specific screen position.
    
    Args:
        x: X coordinate
        y: Y coordinate
    """
    pyautogui.click(x, y)


if __name__ == "__main__":
    # Test controller (be careful - this will move your mouse!)
    print("Testing controller...")
    print("Move your mouse to the top-left corner to abort!")
    time.sleep(3)
    
    print("\nTesting cell_center calculations:")
    for row in [0, 3, 7]:
        for col in [0, 3, 7]:
            x, y = cell_center(row, col)
            print(f"Cell ({row}, {col}) -> Screen ({x}, {y})")
    
    print("\nTesting piece_slot_center:")
    for i in range(3):
        x, y = piece_slot_center(i)
        print(f"Piece slot {i} -> Screen ({x}, {y})")
    
    print("\nTest complete!")
