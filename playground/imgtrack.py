import cv2
import numpy as np


def temp(x):
    pass


cap = cv2.VideoCapture(0)
cv2.namedWindow("Tracking")
cv2.resizeWindow("Tracking", 720, 300)

cv2.createTrackbar("Lower Hue", "Tracking", 0, 255, temp)
cv2.createTrackbar("Upper Hue", "Tracking", 255, 255, temp)
cv2.createTrackbar("Lower Saturation", "Tracking", 0, 255, temp)
cv2.createTrackbar("Upper Saturation", "Tracking", 255, 255, temp)
cv2.createTrackbar("Lower Value", "Tracking", 0, 255, temp)
cv2.createTrackbar("Upper Value", "Tracking", 255, 255, temp)

while True:
    rec, frame = cap.read()


    if rec:

        hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_hue = cv2.getTrackbarPos("Lower Hue", "Tracking")
        upper_hue = cv2.getTrackbarPos("Upper Hue", "Tracking")
        lower_sat = cv2.getTrackbarPos("Lower Saturation", "Tracking")
        upper_sat = cv2.getTrackbarPos("Upper Saturation", "Tracking")
        lower_val = cv2.getTrackbarPos("Lower Value", "Tracking")
        upper_val = cv2.getTrackbarPos("Upper Value", "Tracking")

        lower_bounds = np.array([lower_hue, lower_sat, lower_val])
        upper_bounds = np.array([upper_hue, upper_sat, upper_val])

        mask = cv2.inRange(hsvImg, lower_bounds, upper_bounds)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Image", frame)
        cv2.imshow("ResImg", res)
        cv2.imshow("MaskImg", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
        if cv2.waitKey(1) == 27:
            break
    else:
        break

cv2.destroyAllWindows()
