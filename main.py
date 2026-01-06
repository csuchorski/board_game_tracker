import cv2
import numpy as np

import config
from detectors.board import detect_board_ctn, sort_box_corners
from detectors.cards import detect_cards, classify_card
from detectors.hand import get_hand_mask
from detectors.trains import find_train_stacks, get_train_mask, validate_train_stacks
from trackers.tracker_manager import TrackerManager
from utils import boxes_to_points, draw_on_camera, draw_on_table_view, get_aligned_frame, map_boxes_to_frame


def nothing(x):
    pass


def main(save_video=False, save_path=''):
    SCALE = 1.0

    cap = cv2.VideoCapture(
        'data/MID_3.mp4')
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    # ret, frame = cap.read()
    # frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    # # board_img = cv2.imread('data/ref3.png', 0)
    # if board_img is None:
    #     raise ValueError("error with board reference img")
    BOARD_W, BOARD_H = 1200, 800
    dst_corners = np.array([
        [0, 0],
        [BOARD_W - 1, 0],
        [BOARD_W - 1, BOARD_H - 1],
        [0, BOARD_H - 1]
    ], dtype=np.float32)
    last_board_area = None
    prev_stacks = [None, None]
    train_stacks = None
    table_view = np.zeros((900, 600))
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 600, 900)
    cv2.namedWindow("Aligned view", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Aligned view", 600, 600)
    cv2.namedWindow("Board view", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Board view", 900, 900)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H_min", "Controls", 0, 179, nothing)
    cv2.createTrackbar("H_max", "Controls", 50, 179, nothing)
    cv2.createTrackbar("S_min", "Controls", 30, 255, nothing)
    cv2.createTrackbar("S_max", "Controls", 150, 255, nothing)
    cv2.createTrackbar("V_min", "Controls", 60, 255, nothing)
    cv2.createTrackbar("V_max", "Controls", 255, 255, nothing)
    cv2.createTrackbar("H_mar", "Controls", 15, 100, nothing)
    cv2.createTrackbar("S_mar", "Controls", 75, 100, nothing)
    cv2.createTrackbar("V_mar", "Controls", 60, 100, nothing)
    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contours", 600, 900)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.createTrackbar("Seek", "Tracking", 0, total_frames,
                       lambda x: cap.set(cv2.CAP_PROP_POS_FRAMES, x))

    board_corners = None
    tracker_manager = TrackerManager(max_disappeared=15)
    trains_ROI = np.zeros((600, 900))
    frame_idx = 0

    writer = None
    if save_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SCALE)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SCALE)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        print(f"Saving video to {save_path}...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break
            # frame = cv2.rotate(frame, cv2.)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            overlay = frame.copy()

            if frame_idx % 30 == 0:
                board_corners_tmp, board_cnt = detect_board_ctn(
                    frame, debug=False)

                if board_corners_tmp is not None:
                    board_corners = board_corners_tmp
                    # compute polygon area of detected board
                    new_area = cv2.contourArea(board_corners)

                    accept = True
                    if last_board_area is not None:
                        ratio = new_area / last_board_area
                        if ratio < 0.85 or ratio > 1.15:
                            accept = False
                            print("Rejected board: scale jump", ratio)

                    if accept:
                        last_board_area = new_area
                        H = cv2.getPerspectiveTransform(
                            board_corners, dst_corners)

            detected_rects = None
            if board_corners is not None:

                pts = board_corners.astype(int)
                cv2.polylines(overlay, [pts], isClosed=True,
                              color=(0, 0, 255), thickness=3)

                board_view = cv2.warpPerspective(
                    frame, H, (BOARD_W, BOARD_H))

                table_view, H_final = get_aligned_frame(frame, H)
                board_corners_h = board_corners.reshape(-1, 1, 2)
                board_corners_table = cv2.perspectiveTransform(
                    board_corners_h, H_final)
                board_corners_table = board_corners_table.reshape(
                    4, 2).astype(int)
                cv2.polylines(table_view, [board_corners_table], isClosed=True, color=(
                    0, 255, 0), thickness=3)

                y_min = int(board_corners_table[:, 1].min())
                y_max = int(board_corners_table[:, 1].max())
                # card roi stuff

                card_ROI = table_view[0:y_min, :]

                trains_ROI = table_view[y_max:table_view.shape[0], :]

                # train color detection and masking
                # Detect trains in the lower ROI
                stack_candidates = find_train_stacks(trains_ROI, debug=True)

                stacks_valid = validate_train_stacks(
                    stack_candidates, prev_stacks, trains_ROI, bottom_offset=y_max
                )

                if stacks_valid is not None:
                    train_stacks = stacks_valid
                    prev_stacks = train_stacks

                if train_stacks is not None:
                    for s in train_stacks:
                        cv2.drawContours(
                            table_view, [s['cnt']], -1, (255, 0, 255), 3)

                    train_masks = [get_train_mask(
                        board_view, [s['color']]) for s in train_stacks]
                    for mask in train_masks:
                        board_view[mask == 255] = (255, 255, 255)

                        # cv2.cvtColor(
                        #     np.uint8([[color]]), cv2.COLOR_HSV2BGR)[0][0]

                cv2.imshow("Board view", board_view)

                if card_ROI.size > 0:
                    # detect cards in the top_view
                    if frame_idx % 7 == 0:
                        # top_hand_mask = hand_mask[:, 0:min_x]

                        detected_rects = detect_cards(
                            card_ROI, hand_mask=None, debug=False)

                # Update tracker in table_view
                tracked_objects = tracker_manager.update(
                    table_view, detected_rects)

                # Draw both views
                table_view = draw_on_table_view(
                    table_view, detected_rects, tracked_objects, board_corners_table, train_stacks)
                overlay = draw_on_camera(
                    overlay, detected_rects, tracked_objects, train_stacks, H_final)

            frame_idx += 1
            cv2.putText(overlay, f"Frame: {frame_idx}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

            if 'writer' in locals() and writer is not None:
                writer.write(overlay)

            cv2.imshow("Tracking", overlay)
            cv2.imshow("Aligned view", table_view)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        if 'writer' in locals() and writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(save_video=False, save_path='results/ov_test_blue3.mp4')
