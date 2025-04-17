import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import tensorflow.lite as tflite
import time
import json

class CursorMovementThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.current_x, self.current_y = pyautogui.position()
        self.target_x, self.target_y = self.current_x, self.current_y
        self.running = True
        self.active = False
        self.smoothing = 0.2

    def run(self):
        while self.running:
            if self.active:
                dx = self.target_x - self.current_x
                dy = self.target_y - self.current_y
                self.current_x += dx * self.smoothing
                self.current_y += dy * self.smoothing
                pyautogui.moveTo(self.current_x, self.current_y, _pause=False)
            time.sleep(0.01)

    def update_target(self, x, y):
        self.target_x, self.target_y = x, y

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def stop(self):
        self.running = False

class HandGestureController:
    def __init__(self):
        self.running = False
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.interpreter = tflite.Interpreter(model_path = "model/gesture_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.GESTURE_LABELS = self.load_gesture_labels()
        self.screen_width, self.screen_height = pyautogui.size()
        self.movement_thread = CursorMovementThread()
        self.mouse_mode = False
        self.mouse_held = None

    def process_landmarks(self, hand_landmarks):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x = landmark.x - hand_landmarks.landmark[0].x
            y = landmark.y - hand_landmarks.landmark[0].y
            landmarks.append([x, y])
        flat = np.array(landmarks).flatten().tolist()
        max_val = max([abs(num) for num in flat]) or 1
        processed = (np.array(flat) / max_val).tolist()
        return np.array([processed], dtype=np.float32)
    
    def load_gesture_labels(self):
        try:
            with open("data/poses.txt", "r") as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            print("Warning: poses.txt not found. Using empty label list.")
            return []

    def run(self):
        self.running = True
        self.movement_thread.start()
        try:
            while self.cap.isOpened() and self.running:
                success, image = self.cap.read()
                if not success:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.hands.process(rgb)
                rgb.flags.writeable = True

                flipped_image = cv2.flip(image, 1)
                pose_name = "No Hand"
                confidence = None

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(
                        flipped_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    input_tensor = self.process_landmarks(hand_landmarks)
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
                    self.interpreter.invoke()
                    predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                    best_idx = np.argmax(predictions)
                    confidence = predictions[best_idx]

                    if confidence > 0.90:
                        pose_name = self.GESTURE_LABELS[best_idx]
                    else:
                        pose_name = "Unknown"

                    if pose_name in ["Peace", "Left Click", "Right Click"]:
                        self.mouse_mode = True
                        self.movement_thread.activate()
                        cursor_point = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                        finger_x = (1 - cursor_point.x) * self.screen_width
                        finger_y = cursor_point.y * self.screen_height
                        self.movement_thread.update_target(finger_x, finger_y)

                        if pose_name == "Left Click" and self.mouse_held != "left":
                            pyautogui.mouseDown()
                            self.mouse_held = "left"
                        elif pose_name == "Right Click" and self.mouse_held != "right":
                            pyautogui.mouseDown(button='right')
                            self.mouse_held = "right"
                        elif pose_name == "Peace":
                            if self.mouse_held == "left":
                                pyautogui.mouseUp()
                                self.mouse_held = None
                            elif self.mouse_held == "right":
                                pyautogui.mouseUp(button='right')
                                self.mouse_held = None

                    elif pose_name == "Closed":
                        self.movement_thread.deactivate()
                        self.mouse_mode = False
                        if self.mouse_held:
                            pyautogui.mouseUp()
                            pyautogui.mouseUp(button='right')
                            self.mouse_held = None

                cv2.putText(flipped_image, pose_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if confidence:
                    cv2.putText(flipped_image, f"{confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow('Hand Gesture Control', flipped_image)

                if cv2.waitKey(5) & 0xFF == 27:
                    self.stop()
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.movement_thread.stop()