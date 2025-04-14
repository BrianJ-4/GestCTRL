import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tensorflow.lite as tflite
import time

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize PyAutoGUI
screen_width, screen_height = pyautogui.size()

# Load the TFLite model
interpreter = tflite.Interpreter(model_path = "gesture_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

GESTURE_LABELS = ["Open", "Closed","Peace","Thumbs Up", "Okay", "RockNRoll", "Three", "Point", "LClick", "RClick"]

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    click_time, click_cooldown = 0, 1
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

                pose_name = "No Hand Detected"
                confidence = None
                # Convert back to BGR for OpenCV display
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                        #Problem with hand classification. Hands are switched
                        handedness = results.multi_handedness[0].classification[0].label
                        
                        x_coordinate = screen_width - (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * screen_width)
                        y_coordinate = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * screen_height)
                        pyautogui.moveTo(x_coordinate, y_coordinate)

                        processed_landmarks = process_landmarks(hand_landmarks, handedness)
                        interpreter.set_tensor(input_details[0]['index'], processed_landmarks)
                        interpreter.invoke()
                        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

                        best_idx = np.argmax(predictions)
                        confidence = predictions[best_idx]
                        if confidence > 0.90:
                            pose_name = GESTURE_LABELS[best_idx]
                        else:
                            pose_name = "Unknown"

                        cur_time = time.time()
                        if cur_time - click_time > click_cooldown:
                            process_click(pose_name)
                            click_time = cur_time
                        print(pose_name)


                # # Flip image for mirror view and display pose name
                flipped_image = cv2.flip(image, 1)
                cv2.putText(flipped_image, pose_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(flipped_image, str(confidence), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Hand Tracking', flipped_image)
                        
                # Press Esc to exit
                key = cv2.waitKey(5)
                if key == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_landmarks(hand_landmarks, handedness):
    landmarks = []

    # Extract x and y coordinates of each landmark normalized to wrist position
    for landmark in hand_landmarks.landmark:
        xCoordinate = landmark.x - hand_landmarks.landmark[0].x 
        yCoordinate = landmark.y - hand_landmarks.landmark[0].y

        if handedness == 'Left':
            landmarks.append([xCoordinate, yCoordinate])
        else:
            landmarks.append([-(xCoordinate), yCoordinate])

    # Flatten landmarks into 1D list
    flattened_landmarks = np.array(landmarks).flatten().tolist()

    # Normalize landmarks into -1 to 1 range based on max absolute value
    max_val = max([abs(num) for num in flattened_landmarks]) # Get the max absolute value
    if max_val == 0:
        max_val = 1
    processed_landmarks = (np.array(flattened_landmarks) / max_val).tolist() # Divide every number by the max absolute value
    
    return np.array([processed_landmarks], dtype = np.float32)

def process_click(pose_name):
    if pose_name == "LClick":
        pyautogui.click()
    if pose_name == "RClick":
        pyautogui.click(button='right')

if __name__ == "__main__":
    main()