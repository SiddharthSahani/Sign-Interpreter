import cv2

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
detector =HandDetector(maxHands=1)
classifier = Classifier("Model/Keras_model.h5","Model/labels.txt")
offset = 20
imgSize = 250

folder = "data/OK"
counter = 0

labels =["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","ROCK","OK","HI"]     #LETTING THE system know the values
#labels =["ROCK","FUCK YOU"]
while True:
    success,img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgSpiderman = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        if imgCrop.size==0:continue
        imageCropShape = imgCrop.shape
#height of the image setting
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imageResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)    #to fill up the white gap
            imgSpiderman[:,wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgSpiderman,draw=False)
            print(prediction,index)




        #for setting up the width
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imageResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)    #to fill up the white gap
            imgSpiderman[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgSpiderman,draw=False)

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.putText(imgOutput, f"{prediction[index]}", (x, y - 40), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        #print(prediction[index])
        cv2.imshow("ImageCrop", imgCrop)

        cv2.imshow("ImageSpiderman",imgSpiderman)


    cv2.imshow("Image",imgOutput)
    key = cv2.waitKey(1)

