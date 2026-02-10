"""
Window capture module for capturing screenshots from LDPlayer.
Uses mss for fast screen capture and pywin32 for window detection.
"""
import numpy as np
import mss
import win32gui
import win32ui
import win32con
from typing import Optional, Tuple
from config import config, GameRegion


def find_window_handle(window_title: str) -> Optional[int]:
    """
    Find window handle by title (partial match).
    
    Args:
        window_title: Window title to search for (case-insensitive, partial match)
    
    Returns:
        Window handle (hwnd) or None if not found
    """
    windows = []
    visible_titles = []
    
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                visible_titles.append(title)
            if window_title.lower() in title.lower():
                windows.append(hwnd)
        return True
    
    win32gui.EnumWindows(callback, windows)
    
    if not windows and config.DEBUG:
        print(f"Window '{window_title}' not found. Visible windows:")
        for t in visible_titles[:10]: # Top 10
            print(f"  - {t}")
    
    if windows:
        return windows[0]
    return None


def get_window_rect(hwnd: int) -> Tuple[int, int, int, int]:
    """
    Get window rectangle coordinates.
    
    Args:
        hwnd: Window handle
    
    Returns:
        Tuple of (x, y, width, height)
    """
    rect = win32gui.GetWindowRect(hwnd)
    x = rect[0]
    y = rect[1]
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    return (x, y, width, height)


def capture_region(region: GameRegion) -> np.ndarray:
    """
    Capture a region of the screen.
    
    Args:
        region: GameRegion defining the area to capture
    
    Returns:
        numpy array in BGR format (OpenCV compatible)
    """
    with mss.mss() as sct:
        # mss uses a dict format for monitor region
        monitor = {
            "top": region.y,
            "left": region.x,
            "width": region.width,
            "height": region.height
        }
        
        # Capture the screen
        screenshot = sct.grab(monitor)
        
        # Convert to numpy array (BGRA format)
        img = np.array(screenshot)
        
        # Convert BGRA to BGR (remove alpha channel)
        img_bgr = img[:, :, :3]
        
        return img_bgr


def capture_window(hwnd: int) -> Optional[np.ndarray]:
    """
    Capture entire window by handle.
    
    Args:
        hwnd: Window handle
    
    Returns:
        numpy array in BGR format or None if capture fails
    """
    try:
        x, y, width, height = get_window_rect(hwnd)
        region = GameRegion(x=x, y=y, width=width, height=height)
        return capture_region(region)
    except Exception as e:
        if config.DEBUG:
            print(f"Failed to capture window: {e}")
        return None


class WindowCapture:
    """
    Manages window detection and frame capture.
    """
    
    def __init__(self, window_title: str = None):
        """
        Initialize window capture.
        
        Args:
            window_title: Title of window to capture (defaults to config.WINDOW_TITLE)
        """
        self.window_title = window_title or config.WINDOW_TITLE
        self.hwnd = None
        self.game_region = None
        
    def find_window(self) -> bool:
        """
        Find and lock onto the game window.
        
        Returns:
            True if window found, False otherwise
        """
        self.hwnd = find_window_handle(self.window_title)
        if self.hwnd:
            x, y, width, height = get_window_rect(self.hwnd)
            self.game_region = GameRegion(x=x, y=y, width=width, height=height)
            if config.DEBUG:
                print(f"Found window: {self.window_title} at ({x}, {y}) size {width}x{height}")
            return True
        else:
            if config.DEBUG:
                print(f"Window not found: {self.window_title}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture current frame from the game window.
        
        Returns:
            BGR image as numpy array or None if capture fails
        """
        if not self.hwnd or not self.game_region:
            if not self.find_window():
                return None
        
        return capture_region(self.game_region)


if __name__ == "__main__":
    # Test window capture
    import cv2
    
    capture = WindowCapture()
    if capture.find_window():
        print("Window found! Capturing frame...")
        frame = capture.capture_frame()
        if frame is not None:
            print(f"Captured frame: {frame.shape}")
            cv2.imshow("Captured Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Failed to find window. Make sure LDPlayer is running.")
