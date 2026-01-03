import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def detect_cards(frame_gray, debug=False):
    frame_gray = np.uint8(rescale_intensity(frame_gray, out_range=(0, 255)))

    frame_blurred = cv2.medianBlur(frame_gray, 5)
    frame_thresh = cv2.adaptiveThreshold(
        frame_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_closed = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        frame_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        test_view = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(test_view, contours, -1, (0, 0, 255), 3)
        cv2.imshow("Contours", test_view)

    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5000 < area < 100000:
            # rect = cv2.minAreaRect(cnt)
            # cards.append(rect)
            x, y, w, h = cv2.boundingRect(cnt)
            cards.append((x, y, w, h))

    return cards


def classify_card(extracted_rect):
    hsv = cv2.cvtColor(extracted_rect, cv2.COLOR_BGR2HSV)
    h, w, _ = extracted_rect.shape
    center_region = hsv[h//4:3*h//4, w//4:3*w//4]

    avg_color = np.mean(center_region, axis=(0, 1))
    H, S, V = avg_color[0], avg_color[1], avg_color[2]

    if V < 110:
        return "Black"
    if S < 40:
        return "White"
    if H < 16:
        return "Red"
    elif H < 24:
        return "Orange"
    elif H < 38:
        return "Quest" if S < 90 else "Yellow"
    elif H < 65:
        return "Green"
    elif H < 95:
        return "Joker" if S < 140 else "Blue"
    else:
        return "Pink"
