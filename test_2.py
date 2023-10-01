
import cv2
from sign_classifier import SignClassifier
from hand_detector import HandDetector
import numpy as np
import math


cap = cv2.VideoCapture(0)
detector = HandDetector()
classifier = SignClassifier()
offset = 20
image_size = 300

labels = ['A', 'B', 'C', 'D', 'E', 'HI', 'OK']

while cv2.waitKey(1) != 27:

    _, image = cap.read()
    raw_image = image.copy()
    
    res = detector.find_hands(image)

    if res:
        bbox, _ = res
        x, y, w, h = bbox

        model_input = np.full((image_size, image_size, 3), fill_value=255, dtype=np.float32)
        cropped_image = image[
            y-offset : y+h+offset,
            x-offset : x+w+offset
        ]

        if cropped_image.size == 0:
            continue

        if h > w:
            # pad the width
            k = image_size / h
            wCal = math.ceil(k * w)
            resized_image = cv2.resize(cropped_image, (wCal, image_size))
            gap = math.ceil((image_size - wCal) / 2)
            model_input[:, gap: wCal+gap] = resized_image
        else:
            # pad the height
            k = image_size / w
            hCal = math.ceil(k * h)
            resized_image = cv2.resize(cropped_image, (image_size, hCal))
            gap = math.ceil((image_size - hCal) / 2)
            model_input[gap: hCal+gap, :] = resized_image

        index = classifier.classify(model_input)

        cv2.imshow("Cropped Image", cropped_image)
        cv2.imshow("Model Input", model_input.astype(np.uint8))
        cv2.putText(raw_image, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("Image", raw_image)
