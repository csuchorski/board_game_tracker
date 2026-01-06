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
    return table_view, H_final


def boxes_to_points(boxes):
    """
    Converts (x, y, w, h) boxes to corner points for perspective transform.
    boxes: list of [x, y, w, h]
    returns: list of 4x1x2 float32 arrays
    """
    pts_list = []
    for box in boxes:
        x, y, w, h = box
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        pts_list.append(pts)
    return pts_list


def map_boxes_to_frame(boxes, H_inv):
    """
    Map a list of boxes [x, y, w, h] from table_view to camera frame
    """
    mapped_boxes = []
    for box in boxes:
        x, y, w, h = box
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)

        pts_cam = cv2.perspectiveTransform(pts, H_inv)
        x_min = int(np.min(pts_cam[:, 0, 0]))
        y_min = int(np.min(pts_cam[:, 0, 1]))
        x_max = int(np.max(pts_cam[:, 0, 0]))
        y_max = int(np.max(pts_cam[:, 0, 1]))
        mapped_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
    return mapped_boxes


def draw_on_table_view(table_view, detected_rects, tracked_objects, board_corners_table, train_stacks=None):
    table_overlay = table_view.copy()

    if board_corners_table is not None:
        cv2.polylines(table_overlay, [
                      board_corners_table.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
        for (x, y) in board_corners_table:
            cv2.circle(table_overlay, (x, y), 5, (0, 0, 255), -1)

    if detected_rects is not None:
        for box in detected_rects:
            x, y, w, h = box
            cv2.rectangle(table_overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)

    if tracked_objects is not None:
        for box in tracked_objects.values():
            x, y, w, h = box
            cv2.rectangle(table_overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if train_stacks is not None:
        for s in train_stacks:
            cnt = s['cnt'].copy()
            cv2.drawContours(table_overlay, [cnt], -1, (255, 0, 255), 3)

    return table_overlay


def draw_on_camera(overlay, detected_rects, tracked_objects, train_stacks, H_final):
    H_inv = np.linalg.inv(H_final)

    if detected_rects is not None:
        boxes_cam = map_boxes_to_frame(detected_rects, H_inv)
        for box in boxes_cam:
            x, y, w, h = box
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)

    if tracked_objects is not None:
        boxes_cam = map_boxes_to_frame(list(tracked_objects.values()), H_inv)
        for obj_id, box in zip(tracked_objects.keys(), boxes_cam):
            x, y, w, h = box
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(overlay, f"ID: {obj_id}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if train_stacks is not None:
        for s in train_stacks:
            cnt = s['cnt'].copy()
            cnt_cam = cv2.perspectiveTransform(
                cnt.astype(np.float32).reshape(-1, 1, 2), H_inv)
            cv2.drawContours(
                overlay, [cnt_cam.astype(np.int32)], -1, (255, 0, 255), 3)

    return overlay
