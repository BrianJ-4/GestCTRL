import csv
import cv2
import mediapipe as mp
import numpy as np
import json
from gesture_manager import GestureManager

gesture_manager = GestureManager()

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    try:
        with mp_hands.Hands(
            max_num_hands = 1,
            model_complexity = 0,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        ) as hands:
            reading = False
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Convert image to RGB (for MediaPipe)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                # Convert back to BGR for OpenCV display
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                # Flip image for mirror view
                cv2.imshow('Hand Tracking', cv2.flip(image, 1))
                        
                # Press Esc to exit
                key = cv2.waitKey(5)
                if key == 27:
                    break
                elif key == ord('n'):
                    name = input("Enter Gesture Name: ")
                    reading = True
                elif key == ord('q'):
                    name = None
                    reading = False
                elif key == 13 and reading and results.multi_hand_landmarks != None:
                    gesture_manager.add_pose(name, process_landmarks(hand_landmarks))
    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_landmarks(hand_landmarks):
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

if __name__ == "__main__":
    main()