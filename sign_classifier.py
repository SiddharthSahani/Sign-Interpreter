
import cv2
from keras.models import load_model
import numpy as np


class SignClassifier:

    def __init__(self):
        self.model = load_model("keras_model.h5")
    
    def classify(self, input_image):
        # uses the pretrained model to classify the image

        resized_image = cv2.resize(input_image, (224, 224))
        normalized_image = resized_image / 127.0 - 1.0  # from -1 to 1

        model_input = np.expand_dims(normalized_image, axis=0)

        model_prediction = self.model.predict(model_input)
        predicted_index = np.argmax(model_prediction)
        return predicted_index
