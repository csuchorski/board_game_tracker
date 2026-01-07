import cv2
import numpy as np


def find_train_stacks(roi, min_area=5000, max_area=60000, max_aspect_ratio=2.0,
                      blur_ksize=(5, 5), dilate_iters=1, debug=False):

    h, w = roi.shape[:2]
    crop_y = int(h * 0.60)
    crop_x = int(w * 0.10)

    roi = roi[:crop_y, :]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, blur_ksize, 0)
    thresh = cv2.adaptiveThreshold(
        roi_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                              kernel=kernel, iterations=5)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w > max_aspect_ratio * h or h > max_aspect_ratio * w:
            continue
        area = w*h
        if area < 15000 or area > 60000:
            continue

        # smooth contour
        # cnt = cv2.convexHull(cnt)

        epsilon = 0.002 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)

        # compute center
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        center = np.array([cx, cy])

        candidates.append({'cnt': cnt, 'center': center})
    if debug:
        test_view = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(test_view, contours, -1, (0, 0, 255), 3)
        cv2.imshow("Contours", thresh)

    return candidates


def validate_train_stacks(stack_candidates, prev_stacks, roi, bottom_offset=0,
                          max_dist=150, max_h_dist=15, max_sv_dist=100):

    if len(stack_candidates) != 2:
        return None

    accepted_stacks = []

    # Apply bottom offset and get colors
    candidate_centers = []
    candidate_colors = []
    candidate_cnts_global = []

    for c in stack_candidates:
        cnt = c['cnt'].copy()
        center = c['center'].copy()
        color_hsv = get_dominant_color(roi, cnt)

        cnt[:, :, 1] += bottom_offset
        center[1] += bottom_offset

        candidate_centers.append(center)
        candidate_colors.append(color_hsv)
        candidate_cnts_global.append(cnt)

    candidate_centers = np.array(candidate_centers)
    candidate_colors = np.array(candidate_colors)

    if prev_stacks == [None, None]:
        for i in range(len(stack_candidates)):
            accepted_stacks.append({
                'cnt': candidate_cnts_global[i],
                'center': candidate_centers[i],
                'color': candidate_colors[i]
            })
        return accepted_stacks

    prev_centers = np.array([p['center'] for p in prev_stacks])
    prev_colors = np.array([p['color'] for p in prev_stacks])

    dists = np.linalg.norm(
        candidate_centers[:, None, :] - prev_centers[None, :, :], axis=2)

    # Compute hue differences
    dh = np.abs(candidate_colors[:, None, 0] - prev_colors[None, :, 0])
    dh = np.minimum(dh, 180 - dh)

    # Compute S/V differences
    sv_diff = np.linalg.norm(
        candidate_colors[:, None, 1:3] - prev_colors[None, :, 1:3], axis=2)

    # True if candidate matches previous stack thresholds
    valid_match = (dists <= max_dist) & (
        dh <= max_h_dist) & (sv_diff <= max_sv_dist)

    # Match candidates to previous stacks
    matched_prev = set()
    for i in range(len(stack_candidates)):
        # indices of previous stacks that match
        matches = np.where(valid_match[i])[0]
        for j in matches:
            if j not in matched_prev:
                matched_prev.add(j)
                accepted_stacks.append({
                    'cnt': candidate_cnts_global[i],
                    'center': candidate_centers[i],
                    'color': candidate_colors[i]
                })
                break
        else:
            return None

    return accepted_stacks


def get_dominant_color(roi, cnt):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    cv2.drawContours(mask, [cnt], -1, 255, -1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # mask = cv2.erode(mask, kernel, iterations=5)
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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_total = cv2.morphologyEx(
        mask_total, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_total
