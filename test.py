
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Keras_model.h5","labels.txt")
offset = 20
imgSize = 300


labels = ["A", "B", "C", "D", "E", "HI", "OK"]     #LETTING THE system know the values


while cv2.waitKey(1) != 27:
    success,img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgSpiderman = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]


        imageCropShape = imgCrop.shape
        aspectRatio = h / w

        #height of the image setting
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imageResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)    #to fill up the white gap
            imgSpiderman[:,wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgSpiderman)
            print(prediction,index)

        #for setting up the width
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imageResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)    #to fill up the white gap
            imgSpiderman[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgSpiderman)

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageSpiderman",imgSpiderman)


    cv2.imshow("Image",imgOutput)
