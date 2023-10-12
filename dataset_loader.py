
from hand_detector import HandDetector
import numpy as np
from keras.utils import to_categorical
import cv2


def load_dataset():
    # no use for now, since saved the ndarrays in processed-dataset/

    train_path = "dataset/asl_alphabet_train/asl_alphabet_train/"
    classes = ["space"] + list(chr(i) for i in range(65, 91))

    detector = HandDetector()

    NUM_IMAGES_PER_CLASS = 3000
    
    train_inputs = np.empty((NUM_IMAGES_PER_CLASS*len(classes), 42))
    train_outputs = np.empty(NUM_IMAGES_PER_CLASS*len(classes))

    bad_images = []

    for cls_idx, class_ in enumerate(classes):
        dir_path = train_path + class_

        for i in range(1, NUM_IMAGES_PER_CLASS+1):
            image_path = f"{dir_path}/{class_}{i}.jpg"
            if i % 100 == 0:
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
