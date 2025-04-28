import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import tensorflow.lite as tflite
import time
import json
from pose_action_manager import PoseActionManager
from action_controller import ActionController


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
                pyautogui.moveTo(self.current_x, self.current_y, _pause = False)
            time.sleep(0.01)

    def update_target(self, x, y):
        self.target_x, self.target_y = x, y

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def stop(self):
        self.running = False

class GestureController:
    def __init__(self, camera_manager):
        self.running = False
        self.camera_manager = camera_manager
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands = 1,
            model_complexity = 0,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
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
        self.pose_action_manager = PoseActionManager()
        self.action_controller = ActionController()

    def process_landmarks(self, hand_landmarks, handedness):
        landmarks = []

        # Extract x and y coordinates of each landmark normalized to wrist position
        for landmark in hand_landmarks.landmark:
            xCoordinate = landmark.x - hand_landmarks.landmark[0].x 
            yCoordinate = landmark.y - hand_landmarks.landmark[0].y

            if handedness == 'Left':
                landmarks.append([xCoordinate, yCoordinate])
            elif handedness == 'Right':
                landmarks.append([-xCoordinate, yCoordinate])

        # Flatten landmarks into 1D list
        flattened_landmarks = np.array(landmarks).flatten().tolist()

        # Normalize landmarks into -1 to 1 range based on max absolute value
        max_val = max([abs(num) for num in flattened_landmarks]) # Get the max absolute value
        if max_val == 0:
            max_val = 1
        processed_landmarks = (np.array(flattened_landmarks) / max_val).tolist() # Divide every number by the max absolute value
        
        return np.array([processed_landmarks], dtype = np.float32)
    
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
        while self.running:
            frame = self.camera_manager.get_frame()
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            rgb.flags.writeable = True

            pose_name = "No Hand"
            confidence = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label
                input_tensor = self.process_landmarks(hand_landmarks, handedness)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                best_idx = np.argmax(predictions)
                confidence = predictions[best_idx]

                if confidence > 0.90:
                    pose_name = self.GESTURE_LABELS[best_idx]
                else:
                    pose_name = "Unknown"
                
                if pose_name != "Unknown":
                    action = self.pose_action_manager.get_pose_action(pose_name)
                    if action in ["Mouse Mode", "Left Click", "Right Click"]:
                        self.mouse_mode = True
                        self.movement_thread.activate()
                        cursor_point = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                        finger_x = (1 - cursor_point.x) * self.screen_width
                        finger_y = cursor_point.y * self.screen_height
                        self.movement_thread.update_target(finger_x, finger_y)

                        if action == "Left Click" and self.mouse_held != "left":
                            pyautogui.mouseDown()
                            self.mouse_held = "left"
                        elif action == "Right Click" and self.mouse_held != "right":
                            pyautogui.mouseDown(button = 'right')
                            self.mouse_held = "right"
                        elif action == "Mouse Mode":
                            # Allow movement, but no clicking
                            if self.mouse_held == "left":
                                pyautogui.mouseUp()
                                self.mouse_held = None
                            elif self.mouse_held == "right":
                                pyautogui.mouseUp(button = 'right')
                                self.mouse_held = None

                    elif action == "Neutral":
                        self.movement_thread.deactivate()
                        self.mouse_mode = False
                        if self.mouse_held:
                            pyautogui.mouseUp()
                            pyautogui.mouseUp(button = 'right')
                            self.mouse_held = None
                    
                    elif action != "":
                        self.action_controller.perform_action(action)

    def stop(self):
        self.running = False
        self.movement_thread.stop()

    def reload_model(self):
        self.interpreter = tflite.Interpreter(model_path = "model/gesture_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.GESTURE_LABELS = self.load_gesture_labels()
        print("Gesture model reloaded")