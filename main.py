import threading
import time
import tkinter as tk
from gui import GestureApp
from gesture_controller import GestureController

def main():
    gesture_controller = GestureController()
    gesture_thread = threading.Thread(target = gesture_controller.run, daemon = True)
    gesture_thread.start()

    print("Gesture controller started. Press Ctrl+C to exit.")

    root = tk.Tk()
    app = GestureApp(root, gesture_controller)

    def on_close():
        gesture_controller.stop()
        gesture_thread.join()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)  # Handle window closing properly
    root.mainloop()

if __name__ == "__main__":
    main()