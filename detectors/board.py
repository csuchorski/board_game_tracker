import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def sort_box_corners(box):
    """
    Returns corners in OpenCV homography order:
    TL, TR, BR, BL
    """
    box = np.asarray(box).reshape(4, 2)

    s = box.sum(axis=1)
    d = np.diff(box, axis=1).reshape(-1)

    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(d)]
    bl = box[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_board_orb(vid_frame_gray, ref_frame_gray, orb, bf):
    kp_board, des_board = orb.detectAndCompute(ref_frame_gray, None)
    kp_frame, des_frame = orb.detectAndCompute(vid_frame_gray, None)

    if des_frame is None or len(des_frame) < 10:
        return None, None

    matches = bf.knnMatch(des_board, des_frame, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 20:
        return None, None

    src_pts = np.float32(
        [kp_board[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, None

    h, w = ref_frame_gray.shape[:2]
    board_corners = np.float32(
        [[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    camera_corners = cv2.perspectiveTransform(board_corners, H)

    camera_corners = sort_box_corners(camera_corners)

    return camera_corners, H


# def detect_board_ctn(frame, debug=False):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     thresh = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         31, 10
#     )
#     blur = cv2.GaussianBlur(gray, (15, 15), 0)

#     edges = cv2.Canny(blur, 50, 120)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#     closed_canny = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#     contours, _ = cv2.findContours(
#         closed_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     best_cnt = None
#     best_score = 0
#     best_rect = None

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < 100_000:
#             continue

#         rect = cv2.minAreaRect(cnt)
#         (cx, cy), (w, h), angle = rect
#         rect_area = w * h
#         if rect_area < 200_000:
#             continue

#         fill_ratio = area / rect_area
#         score = area * fill_ratio

#         if score > best_score:
#             best_score = score
#             best_cnt = cnt
#             best_rect = rect

#     if best_rect is None:
#         return None, None

#     box = cv2.boxPoints(best_rect)
#     box = np.int32(box)

#     if debug:
#         vis = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
#         cv2.drawContours(vis, contours, -1, (100, 100, 100), 1)
#         cv2.drawContours(vis, [box], 0, (0, 0, 255), 3)
#         cv2.imshow("Contours", vis)

#     return best_rect, box
def detect_board_ctn(frame, debug=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print('boardnot contours')
        return None, None

    # 5) Board is the largest contour
    cnt = max(contours, key=cv2.contourArea)

    # 6) Approximate to 4 corners
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) != 4:
        return None, None

    corners = approx.reshape(4, 2).astype(np.float32)

    # 7) Order corners
    ordered = sort_box_corners(corners)

    if debug:
        vis = cv2.cvtColor(closed.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, [ordered.astype(int)], -1, (0, 0, 255), 3)
        cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 3)
        cv2.imshow("Contours", vis)

    return ordered, cnt
