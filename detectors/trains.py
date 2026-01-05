import cv2
import numpy as np


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
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
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
        if rect_id > 2:
            break
    return stack_contours, output_vis, thresh


def find_train_stacks(roi, min_area=500, max_area=60000, max_aspect_ratio=2.0,
                      blur_ksize=(5, 5), dilate_iters=1):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, blur_ksize, 0)
    thresh = cv2.adaptiveThreshold(
        roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    if dilate_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=dilate_iters)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)

        # compute center
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        center = np.array([cx, cy])

        candidates.append({'cnt': cnt, 'center': center})

    return candidates


def validate_train_stacks(stack_candidates, prev_stacks, roi, bottom_offset=0,
                          max_dist=200, max_h_dist=500, max_sv_dist=2000):
    # if prev_stacks == [None, None]:
    #     # no previous stacks, accept all candidates

    # # Track which prev_stacks have been matched
    # matched_prev = set()
    # accepted_stacks = []
    if len(stack_candidates) != 2:
        return None

    accepted_stacks = []
    for c in stack_candidates:
        cnt = c['cnt']
        center = c['center']
        color_hsv = get_dominant_color(roi, cnt)
        cnt_global = cnt.copy()
        cnt_global[:, :, 1] += bottom_offset
        accepted_stacks.append(
            {'cnt': cnt_global, 'center': center, 'color': color_hsv})
    return accepted_stacks
    # for candidate in stack_candidates:
    #     cnt = candidate['cnt']
    #     center = candidate['center']
    #     color_hsv = get_dominant_color(roi, cnt)
    #     cnt_global = cnt.copy()
    #     cnt_global[:, :, 1] += bottom_offset

    #     # find any prev stack that matches within thresholds and not yet matched
    #     match_found = False
    #     for i, prev in enumerate(prev_stacks):
    #         if i in matched_prev or prev is None:
    #             continue

    #         dist = np.linalg.norm(center - prev['center'])
    #         dh = min(abs(int(color_hsv[0]) - int(prev['color'][0])),
    #                  180 - abs(int(color_hsv[0]) - int(prev['color'][0])))
    #         sv_diff = np.linalg.norm(color_hsv[1:3] - prev['color'][1:3])

    #         if dist <= max_dist and dh <= max_h_dist and sv_diff <= max_sv_dist:
    #             # candidate is valid, mark prev as matched
    #             matched_prev.add(i)
    #             match_found = True
    #             break

    #     if not match_found:
    #         # candidate cannot be matched to any previous stack → reject frame
    #         print("stack validation failed: unmatched candidate")
    #         return None

    #     accepted_stacks.append(
    #         {'cnt': cnt_global, 'center': center, 'color': color_hsv})
    # return accepted_stacks


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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)

    return mask_total


# def get_dominant_color_lab(roi, cnt):
#     """
#     Get dominant color of a contour in LAB space for robustness under lighting.
#     Returns LAB color as np.array([L, A, B]).
#     """
#     lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

#     # Build mask for contour
#     mask = np.zeros(roi.shape[:2], dtype=np.uint8)
#     cv2.drawContours(mask, [cnt], -1, 255, -1)

#     pixels = lab[mask == 255]

#     if len(pixels) == 0:
#         return np.array([128, 128, 128])  # middle gray

#     # Cluster AB channels only (ignore L/lightness)
#     ab = pixels[:, 1:3]  # A,B channels
#     kmeans = KMeans(n_clusters=1, n_init=3).fit(ab)
#     a_dom, b_dom = kmeans.cluster_centers_[0]

#     # L = mean lightness in contour
#     l_dom = np.mean(pixels[:, 0])

#     return np.array([int(l_dom), int(a_dom), int(b_dom)], dtype=np.uint8)


# def get_train_mask_lab(frame, dominant_colors, l_margin=60, ab_margin=15):
#     """
#     Mask train regions using dominant LAB colors.
#     """
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)

#     for color in dominant_colors:
#         lower = np.clip(
#             color - np.array([l_margin, ab_margin, ab_margin]), 0, [255, 255, 255])
#         upper = np.clip(
#             color + np.array([l_margin, ab_margin, ab_margin]), 0, [255, 255, 255])
#         mask = cv2.inRange(lab, lower.astype(np.uint8), upper.astype(np.uint8))
#         mask_total = cv2.bitwise_or(mask_total, mask)

#     # Morphological cleanup
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
#     mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)

#     return mask_total
