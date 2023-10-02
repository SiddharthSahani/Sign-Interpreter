
from hand_detector import HandDetector
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import cv2
import numpy as np


def load_dataset():
    train_path = "dataset/asl_alphabet_train/asl_alphabet_train/"
    classes = ["space"] + list(chr(i) for i in range(65, 91))

    detector = HandDetector()

    NUM_IMAGES_PER_CLASS = 500
    
    train_inputs = np.empty((NUM_IMAGES_PER_CLASS*len(classes), 42))
    train_outputs = np.empty(NUM_IMAGES_PER_CLASS*len(classes))

    bad_images = []

    for cls_idx, class_ in enumerate(classes):
        dir_path = train_path + class_

        for i in range(1, NUM_IMAGES_PER_CLASS+1):
            image_path = f"{dir_path}/{class_}{i}.jpg"
            print(image_path)
            img = cv2.imread(image_path)

            result = detector.find_hands(img)
            if result:
                bbox, (x_lms, y_lms) = result
                train_inputs[cls_idx*NUM_IMAGES_PER_CLASS + i-1] = np.array(x_lms + y_lms)
                train_outputs[cls_idx*NUM_IMAGES_PER_CLASS + i-1] = cls_idx
            else:
                bad_images.append(cls_idx*NUM_IMAGES_PER_CLASS + i)
    
    return train_inputs, to_categorical(train_outputs, num_classes=27), np.array(bad_images)


# takes too much time, so cache it somewhere with np.save
# and then load with np.load

## Comment these 5 lines after first run (unless u get an error)
X, Y, bad = load_dataset()
X = np.delete(X, bad, axis=0)
Y = np.delete(Y, bad, axis=0)
np.save("fast-x.npy", X)
np.save("fast-y.npy", Y)

X = np.load("fast-x.npy")
Y = np.load("fast-y.npy")

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
        input_data = np.array(x_lms + y_lms)

        if np.min(input_data) >= 0 and np.max(input_data) <= 1:
            prediction = model.predict(np.expand_dims(input_data, axis=0))
            pred_idx = np.argmax(prediction)
            cv2.putText(image, classes[pred_idx], (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0))

            for i in range(27):
                cv2.putText(image, f"{classes[i]}: {prediction[0][i]:.3f}", (10, 70 + i*20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    cv2.imshow("image", image)
