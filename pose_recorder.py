import cv2
import mediapipe as mp
import numpy as np
from gesture_manager import GestureManager

class GestureRecorder:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands = 1,
            model_complexity = 0,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )
        self.cap = cv2.VideoCapture(0) # Might need to add logic to this to allow user to select webcam
        self.running = False
        self.reading = False
        self.pose_name = None
        self.gesture_manager = GestureManager()
        self.current_landmarks = None

    def start_recording(self, pose_name):
        self.pose_name = pose_name
        self.reading = True
        if not self.cap.isOpened():
            self.cap.open(0)

    def stop_recording(self):
        self.pose_name = None
        self.reading = False
        if self.cap.isOpened():
            self.cap.release()

    def record_frame(self):
        if self.reading and self.current_landmarks:
            processed = self.process_landmarks(self.current_landmarks)
            self.gesture_manager.add_pose(self.pose_name, processed)
    
    def run(self):
        self.running = True
        while self.running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Convert image to RGB (for MediaPipe)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True

            # Convert back to BGR for OpenCV display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.current_landmarks = hand_landmarks
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            else:
                self.current_landmarks = None

            # Flip image for mirror view
            cv2.imshow('Pose Recorder', cv2.flip(image, 1))
            key = cv2.waitKey(5)
    
    def stop(self):
        self.running = False
        self.reading = False
        self.cap.release()
        cv2.destroyAllWindows()

    def process_landmarks(self, hand_landmarks):
        landmarks = []
        
        # Extract x and y coordinates of each landmark normalized to wrist position
        for landmark in hand_landmarks.landmark:
            xCoordinate = landmark.x - hand_landmarks.landmark[0].x 
            yCoordinate = landmark.y - hand_landmarks.landmark[0].y
            landmarks.append([xCoordinate, yCoordinate])

        # Flatten landmarks into 1D list
        flattened_landmarks = np.array(landmarks).flatten().tolist()

        # Normalize landmarks into -1 to 1 range based on max absolute value
        processed_landmarks = []
        max_val = max([abs(num) for num in flattened_landmarks]) # Get the max absolute value
        if max_val == 0:
            max_val = 1
        processed_landmarks = (np.array(flattened_landmarks) / max_val).tolist() # Divide every number by the max absolute value

        return processed_landmarks