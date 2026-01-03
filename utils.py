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
