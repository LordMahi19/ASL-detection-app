# File 1: create_dataset.py
import os
import tensorflow as tf
import mediapipe as mp
import cv2
from multiprocessing import Pool
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

def process_image(args):
    dir_, img_path = args
    data_aux = []
    x_ = []
    y_ = []

    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) == 42:
            return data_aux, dir_
    return None

if __name__ == '__main__':
    data = []
    labels = []

    img_paths = [(dir_, img_path) for dir_ in os.listdir(DATA_DIR) for img_path in os.listdir(os.path.join(DATA_DIR, dir_))]

    with Pool() as pool:
        results = pool.map(process_image, img_paths)

    for result in results:
        if result:
            data.append(result[0])
            labels.append(result[1])

    # Export the dataset as .npz file for future use
    np.savez("sign_language_dataset.npz", data=np.array(data), labels=np.array(labels))

    print("Dataset saved as sign_language_dataset.npz")
