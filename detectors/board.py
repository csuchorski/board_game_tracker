import cv2
import numpy as np


def detect_board(vid_frame_gray, ref_frame_gray, orb, bf):
    kp_board, des_board = orb.detectAndCompute(ref_frame_gray, None)
    kp_frame, des_frame = orb.detectAndCompute(vid_frame_gray, None)

    if des_frame is None or len(des_frame) < 2:
        return None

    matches = bf.knnMatch(des_board, des_frame, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > 20:
        src_pts = np.float32(
            [kp_board[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            return None

        h, w = ref_frame_gray.shape[:2]
        H_inv = np.linalg.inv(H)
        board_corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        camera_corners = cv2.perspectiveTransform(board_corners, H_inv)
        return camera_corners

    return None
