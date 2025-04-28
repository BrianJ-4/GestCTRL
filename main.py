import threading
import time
import tkinter as tk
from camera_manager import CameraManager
from gui import GestureApp
from gesture_controller import GestureController
from settings_manager import SettingsManager

def main():
    settings_manager = SettingsManager()
    camera_manager = CameraManager(settings_manager.get_webcam_index_setting())
    camera_error = None

    try:
        camera_manager.start()
    except RuntimeError as e:
        print("Camera Error:", str(e))
        camera_error = str(e)

    gesture_controller = GestureController(camera_manager)
    gesture_thread = threading.Thread(target = gesture_controller.run, daemon = True)
    gesture_thread.start()

    print("Gesture controller started. Press Ctrl+C to exit.")

    root = tk.Tk()
    app = GestureApp(root, gesture_controller, camera_manager, settings_manager, camera_error)

    def on_close():
        gesture_controller.stop()
        gesture_thread.join()
        camera_manager.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)  # Handle window closing properly
    root.mainloop()

if __name__ == "__main__":
    main()