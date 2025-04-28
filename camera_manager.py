import cv2
import threading
import time

class CameraManager:
    def __init__(self, camera_index = 0):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera: " + str(self.camera_index))
        self.running = True
        self.thread = threading.Thread(target = self.update_frames, daemon = True)
        self.thread.start()

    def update_frames(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.latest_frame = None