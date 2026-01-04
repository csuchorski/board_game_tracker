import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def detect_cards(frame, hand_mask=None, debug=False):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.uint8(rescale_intensity(frame, out_range=(0, 255)))

    frame_blurred = cv2.medianBlur(frame_gray, 5)
    # frame_blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    frame_thresh = cv2.adaptiveThreshold(
        frame_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 20)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_closed = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        frame_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hand_discarded = []
    cards = []
    print('___')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000 or area > 50000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 25000:
            continue
        if hand_mask is not None:
            hand_roi = hand_mask[y:y+h, x:x+w]

            hand_pixels = cv2.countNonZero(hand_roi)

            rect_area = w * h
            overlap_ratio = hand_pixels / rect_area

            # filter if a significant part of the detected rect overlaps with the hand_mask
            if overlap_ratio > 0.3:
                print(
                    f"Ignored contour with {overlap_ratio:.2f} hand overlap")
                hand_discarded.append(cnt)
                continue
        # rect = cv2.minAreaRect(cnt)

        # cards.append(rect)
        cards.append((x, y, w, h))
    if debug:
        test_view = frame_closed.copy()  # cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(test_view, contours, -1, (0, 0, 255), 3)
        # cv2.drawContours(test_view, hand_discarded, -1, (255, 0, 255), 3)
        cv2.imshow("Contours", frame_closed)
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
