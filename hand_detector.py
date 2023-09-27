
import cv2
import mediapipe


mp_hands = mediapipe.solutions.hands
mp_draw_utils = mediapipe.solutions.drawing_utils


class HandDetector:

    def __init__(self, hand_count=1):
        self.hand_count = hand_count
        self.mp_hands = mp_hands.Hands(max_num_hands=self.hand_count)
    
    def find_hands(self, input_image):
        # returns bounding-box of the detected hand and
        # draws landmarks on top of the input_image

        image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        mp_results = self.mp_hands.process(image_rgb)

        # no hand(s) were detected
        if not mp_results.multi_hand_landmarks:
            return None

        height, width, _ = input_image.shape

        for hand_lms in mp_results.multi_hand_landmarks:

            mp_draw_utils.draw_landmarks(input_image, hand_lms, mp_hands.HAND_CONNECTIONS)

            lm_x_list = []
            lm_y_list = []
            for lm in hand_lms.landmark:
                lm_x_list.append(lm.x * width)
                lm_y_list.append(lm.y * height)
            
            # creating the bounding box
            x_min = min(lm_x_list)
            x_max = max(lm_x_list)
            y_min = min(lm_y_list)
            y_max = max(lm_y_list)
            box_w = x_max - x_min
            box_h = y_max - y_min

            return (
                int(x_min), int(y_min),
                int(box_w), int(box_h)
            )


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    detector = HandDetector()

    # Press ESCAPE key to quit
    while cv2.waitKey(1) != 27:
        _, image = capture.read()
        cv2.imshow("raw", image)

        detector.find_hands(image)
        cv2.imshow("processed", image)
