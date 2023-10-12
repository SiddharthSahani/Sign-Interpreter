
from hand_detector import HandDetector
from keras.models import Sequential
from keras.layers import Dense
import cv2
import numpy as np


X = np.load("processed-dataset/processed-x-3000.npy")
Y = np.load("processed-dataset/processed-y-3000.npy")

## MAKE YOUR OWN MODEL
model = Sequential()

model.add(Dense(80, input_shape=(42,), activation="relu"))
model.add(Dense(80, activation="relu"))
model.add(Dense(27, activation="softmax"))

## TRY TO MAKE THE LOSS AS LOW AS POSSIBLE
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, Y, epochs=10)


classes = ["space"] + list(chr(i) for i in range(65, 91))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
det = HandDetector()

while cv2.waitKey(1) != 27:
    _, image = cap.read()
    res = det.find_hands(image)

    if res:
        _, (x_lms, y_lms) = res
        input_data = np.array(x_lms + y_lms) * 2 - 1

        if np.min(input_data) >= -1 and np.max(input_data) <= 1:
            prediction = model.predict(np.expand_dims(input_data, axis=0))
            pred_idx = np.argmax(prediction)
            cv2.putText(image, f"{classes[pred_idx]}: {prediction[0][pred_idx]:.3f}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0))

            for i in range(27):
                cv2.putText(image, f"{classes[i]}: {prediction[0][i]:.3f}", (10, 70 + i*20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    cv2.imshow("image", image)
