import cv2
import numpy as np

# Trying Event Handling
# Works but only for that event frame cant do it for all frames yet.
# Probably need to add another loop for checking the event and adding the events at each frame

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y, sep="-")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{x}-{y}", (x, y), font, 0.5, (255, 18, 18))
        cv2.imshow("Image", frame)


# Creating the camera

cap = cv2.VideoCapture(0)

# Setting Properties
cap.set(3, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # ID - 3
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # ID - 4

# Main Event Loop
while True:
    rec, frame = cap.read()
    if rec:
        cv2.imshow("Image", frame)
        cv2.setMouseCallback("Image", click_event)

        # Wait Key for closing idk how it works but this is how its supposed to be ig???
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# --------
''' Simple Video Input
# Getting all types of events in CV
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

capture = cv.VideoCapture(0)


capture.set(3, 1080)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # ID - 3
print(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))   # ID - 4


# Instance Loop
while (True):

    rect, frame = capture.read()

    # Rect returns if image rectangle was made or not

    if rect:

        if cv2.EVENT_FLAG_LBUTTON:
            print("Left Hello")

        cv.imshow('OpenCV Test', frame)
        # frame = cv.rectangle(frame, (200, 200), (100, 100), (18, 18, 18), 10)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


# Cleaning Up
capture.release()
cv2.destroyAllWindows()
'''
