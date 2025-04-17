import threading
import time
from gesture_controller import GestureController

def main():
    gesture_controller = GestureController()
    gesture_thread = threading.Thread(target = gesture_controller.run, daemon = True)
    gesture_thread.start()

    print("Gesture controller started. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)  # This keeps the main thread alive
            # UI event loop goes here
    except KeyboardInterrupt:
        print("Stopping")
        gesture_controller.stop()
        gesture_thread.join()

if __name__ == "__main__":
    main()