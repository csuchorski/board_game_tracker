import cv2
import numpy as np
# this is a leftover file from one of my previous approaches which was based on detecting
# the hand to safeguard the pipeline from making wrong detections because of the hand's disruption


def get_hand_mask(frame, border_margin=10):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = frame.shape[:2]

    lower_skin = np.array([0, 0, 3], dtype=np.uint8)
    upper_skin = np.array([40, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_hand_mask = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        touches_left = x < border_margin
        touches_top = y < border_margin
        touches_right = (x + w) > (w_img - border_margin)
        touches_bottom = (y + h) > (h_img - border_margin)

        if touches_left or touches_top or touches_right or touches_bottom:
            cv2.drawContours(final_hand_mask, [
                             cnt], -1, 255, thickness=cv2.FILLED)

    return final_hand_mask
