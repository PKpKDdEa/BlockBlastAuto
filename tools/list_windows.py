import win32gui

def list_windows():
    print(f"{'HWND':<10} | {'Title'}")
    print("-" * 50)
    
    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                print(f"{hwnd:<10} | {title}")
        return True
    
    win32gui.EnumWindows(callback, None)

if __name__ == "__main__":
    list_windows()
