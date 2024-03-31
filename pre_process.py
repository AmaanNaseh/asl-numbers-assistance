import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

dataset = './dataset'

data = []
labels = []

for class_dir in os.listdir(dataset):
    for img_path in os.listdir(os.path.join(dataset, class_dir)):
        main_data = []

        x_ = []
        y_ = []

        image = cv2.imread(os.path.join(dataset, class_dir, img_path))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        final_images = hands.process(rgb_image)

        if final_images.multi_hand_landmarks:
            for hand_landmarks in final_images.multi_hand_landmarks:

                for i in range(len(hand_landmarks.landmark)):
                    
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):

                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    
                    main_data.append(x - min(x_))
                    main_data.append(y - min(y_))

            data.append(main_data)
            labels.append(class_dir)

f = open('preprocess_numbers.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
