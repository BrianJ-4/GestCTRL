import numpy as np
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_gesture_data():
    gestures = []       # Stores pose landmark coordinates
    labels = []         # Stores gesture numeric keys
    label_dict = {}     # Maps gesture names to numeric keys
    with open('data/gestures.csv', newline = '') as file:
        reader = csv.reader(file)
        label_counter = 0
        for row in reader:
            gesture_name = row[0]  # First column = gesture name
            features = list(map(float, row[1:]))  # Convert landmark values to floats

            # Assign a numeric label to each gesture name
            if gesture_name not in label_dict:
                label_dict[gesture_name] = label_counter
                label_counter += 1

            gestures.append(features)
            labels.append(label_dict[gesture_name])
    return np.array(gestures), np.array(labels), label_dict

def split_data(gestures, labels):
    x_training, x_testing, y_training, y_testing = train_test_split(gestures, labels, test_size = 0.25, random_state = 42)
    return x_training, x_testing, y_training, y_testing

def train_model():
    gestures, labels, label_dict = load_gesture_data()
    x_training, x_testing, y_training, y_testing = split_data(gestures, labels)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape = (x_training.shape[1],)),          # Input layer
        tf.keras.layers.Dense(64, activation = 'relu'),                 # Hidden layer 1
        tf.keras.layers.Dense(32, activation = 'relu'),                 # Hidden layer 2
        tf.keras.layers.Dense(len(label_dict), activation = 'softmax')  # Output layer
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x_training, y_training, epochs = 100, batch_size = 8, validation_data = (x_testing, y_testing))

    loss, accuracy = model.evaluate(x_testing, y_testing)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    model.save("model/gesture_model.keras")  # Save Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open("model/gesture_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Model saved as gesture_model.tflite")

train_model()