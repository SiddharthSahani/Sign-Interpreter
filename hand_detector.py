
import cv2
import mediapipe


mp_hands = mediapipe.solutions.hands
mp_draw_utils = mediapipe.solutions.drawing_utils


class HandDetector:

    def __init__(self):
        self.mp_hands = mp_hands.Hands(max_num_hands=1)
    
    def find_hands(self, input_image):
        # returns bbox and landmarks (normalized from 0 to 1) of the detected hand and
        # draws landmarks on top of the input_image

        image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        mp_results = self.mp_hands.process(image_rgb)

        # no hand(s) were detected
        if not mp_results.multi_hand_landmarks:
            return None

        height, width, _ = input_image.shape
        
        hand_lms = mp_results.multi_hand_landmarks[0]

        mp_draw_utils.draw_landmarks(input_image, hand_lms, mp_hands.HAND_CONNECTIONS)

        lm_x_list = []
        lm_y_list = []
        for lm in hand_lms.landmark:
            lm_x_list.append(lm.x)
            lm_y_list.append(lm.y)
        
        # creating the bounding box
        x_min = min(lm_x_list) * width
        x_max = max(lm_x_list) * width
        y_min = min(lm_y_list) * height
        y_max = max(lm_y_list) * height
        box_w = x_max - x_min
        box_h = y_max - y_min

        bbox = (x_min, y_min, box_w, box_h)
        return bbox, (lm_x_list, lm_y_list)


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    detector = HandDetector()

    # Press ESCAPE key to quit
    while cv2.waitKey(1) != 27:
        _, image = capture.read()
        cv2.imshow("raw", image)

        res = detector.find_hands(image)
        if res:
            bbox, (x_lms, y_lms) = res
            cv2.rectangle(image, bbox, (255, 0, 0))
        cv2.imshow("processed", image)
