import sys
import numpy as np
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import resource_path

def load_gesture_data():
    gestures = []
    labels = []
    label_dict = {}
    with open(resource_path('data/gestures.csv'), newline='') as file:
        reader = csv.reader(file)
        label_counter = 0
        for row in reader:
            gesture_name = row[0]
            features = list(map(float, row[1:]))
            if gesture_name not in label_dict:
                label_dict[gesture_name] = label_counter
                label_counter += 1
            gestures.append(features)
            labels.append(label_dict[gesture_name])
    return np.array(gestures), np.array(labels), label_dict

def split_data(gestures, labels):
    x_training, x_testing, y_training, y_testing = train_test_split(gestures, labels, test_size=0.25, random_state=42)
    return x_training, x_testing, y_training, y_testing

def train_model():
    gestures, labels, label_dict = load_gesture_data()
    x_training, x_testing, y_training, y_testing = split_data(gestures, labels)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(x_training.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(label_dict), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_training, y_training, epochs=100, batch_size=8, validation_data=(x_testing, y_testing))
    loss, accuracy = model.evaluate(x_testing, y_testing)

    # print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    model.save(resource_path("model/gesture_model.keras"))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(resource_path("model/gesture_model.tflite"), "wb") as f:
        f.write(tflite_model)
    # print("Model saved as gesture_model.tflite")
