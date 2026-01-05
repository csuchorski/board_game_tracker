import cv2
import numpy as np
from sklearn.cluster import KMeans


def detect_train_stacks(roi, min_area=500, blur_ksize=(5, 5), dilate_iters=1):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi_blur = cv2.GaussianBlur(roi_gray, blur_ksize, 0)

    thresh = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    if dilate_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=dilate_iters)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stack_contours = []
    rect_id = 1
    output_vis = roi.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) <= min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if w > 2*h or h > 2*w:
            continue
        if area < 15000 or area > 60000:
            continue
        cv2.drawContours(output_vis, [cnt], -1, (255, 0, 255),
                         thickness=cv2.FILLED)
        cv2.rectangle(output_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ID {rect_id}"
        cv2.putText(
            output_vis,
            label,
            (x + 5, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
        rect_id += 1  # type: ignore
        stack_contours.append(cnt)
    return stack_contours, output_vis, thresh


def get_dominant_color(roi, cnt):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    pixels = hsv[mask == 255]

    if len(pixels) == 0:
        return np.array([0, 0, 0])

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]

    hist = np.bincount(h, minlength=180)
    h_dom = np.argmax(hist)

    mask_h = h == h_dom
    s_dom = np.mean(s[mask_h])
    v_dom = np.mean(v[mask_h])

    return np.array([h_dom, int(s_dom), int(v_dom)])


def get_train_controls_vals():
    h_margin = cv2.getTrackbarPos("H_mar", "Controls")
    s_margin = cv2.getTrackbarPos("S_mar", "Controls")
    v_margin = cv2.getTrackbarPos("V_mar", "Controls")

    return h_margin, s_margin, v_margin


def get_train_mask(frame, dominant_colors, h_margin=15, s_margin=50, v_margin=50):
    h_margin, s_margin, v_margin = get_train_controls_vals()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)

    for color in dominant_colors:
        lower = np.clip(
            color - np.array([h_margin, s_margin, v_margin]), [0, 0, 0], [179, 255, 255])
        upper = np.clip(
            color + np.array([h_margin, s_margin, v_margin]), [0, 0, 0], [179, 255, 255])
        mask = cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8))
        mask_total = cv2.bitwise_or(mask_total, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)

    return mask_total
