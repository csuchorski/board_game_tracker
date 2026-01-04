import numpy as np
import cv2


def sort_box_corners(box):
    """Sorts corners of a box: top-left, bottom-right, top-right, bottom-left."""
    s = box.sum(axis=1)
    d = np.diff(box, axis=1)

    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(d)]
    bl = box[np.argmax(d)]
    return tl, br, tr, bl


def calc_box_dimensions(tl, br, tr, bl):
    """Calculates max width and height from sorted corners."""
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight


def extract_rotated_card(box, vid_frame):
    """Warps perspective to extract a flat image of a rotated card."""
    tl, br, tr, bl = sort_box_corners(box)
    box_width, box_height = calc_box_dimensions(tl, br, tr, bl)

    src_pts = np.array([tl, tr, br, bl], dtype="float32")
    dst_pts = np.array([
        [0, 0],
        [box_width - 1, 0],
        [box_width - 1, box_height - 1],
        [0, box_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(vid_frame, M, (box_width, box_height))

    # Normalize orientation
    if box_width > box_height:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def get_aligned_frame(frame, H):
    h_img, w_img = frame.shape[:2]
    img_corners = np.float32(
        [[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(img_corners, H)

    [x_min, y_min] = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])

    H_final = H_translation.dot(H)

    output_width = x_max - x_min
    output_height = y_max - y_min
    table_view = cv2.warpPerspective(
        frame, H_final, (output_width, output_height), borderValue=(255, 255, 255))
    return table_view
