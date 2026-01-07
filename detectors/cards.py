import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def detect_cards(frame, debug=False):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.uint8(rescale_intensity(frame, out_range=(0, 255)))

    frame_blurred = cv2.medianBlur(frame_gray, 5)
    # frame_blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    frame_thresh = cv2.adaptiveThreshold(
        frame_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 10)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_closed = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        frame_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hand_discarded = []
    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000 or area > 50000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        if rect_area > 25000:
            continue

        cards.append((x, y, w, h))
    if debug:
        test_view = cv2.cvtColor(frame_closed, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(test_view, contours, -1, (0, 0, 255), 3)
        cv2.drawContours(test_view, hand_discarded, -1, (255, 0, 255), 3)
        cv2.imshow("Contours", test_view)
    return cards
