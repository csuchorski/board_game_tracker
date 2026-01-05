import cv2
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity


import config
from detectors.board import detect_board_ctn, sort_box_corners
from detectors.cards import detect_cards, classify_card
from detectors.hand import get_hand_mask
from detectors.trains import detect_train_stacks, find_train_stacks, get_dominant_color, get_dominant_color_lab, get_train_mask, get_train_mask_lab, validate_train_stacks
from trackers.tracker_manager import TrackerManager
from utils import extract_rotated_card, get_aligned_frame, opencv_autoexposure


def nothing(x):
    pass


def main(save_video=False, save_path=''):
    SCALE = 1.0

    cap = cv2.VideoCapture(
        'data/NEW_MID3.mp4')
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
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 600, 900)
    # cv2.namedWindow("Aligned view", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Aligned view", 600, 600)
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
    cv2.createTrackbar("S_mar", "Controls", 91, 100, nothing)
    cv2.createTrackbar("V_mar", "Controls", 50, 100, nothing)
    # cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Contours", 600, 900)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.createTrackbar("Seek", "Tracking", 0, total_frames,
                       lambda x: cap.set(cv2.CAP_PROP_POS_FRAMES, x))

    ordered_corners = None
    tracker_manager = TrackerManager(max_disappeared=30)
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
                continue
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            overlay = frame.copy()

            # hand_mask = get_hand_mask(frame)

            # board detection, every 30 frames
            if frame_idx % 30 == 0:
                rect, corners = detect_board_ctn(frame, debug=False)

                # found_corners, H_board = detect_board(
                #     frame_gray, board_img, config.ORB, config.BF)
                if rect is not None:
                    accept = True
                    (_, _), (w, h), _ = rect
                    new_area = w*h
                    if last_board_area is not None:
                        ratio = new_area / last_board_area
                        if ratio < (1 - 0.15) or ratio > (1 + 0.15):
                            accept = False
                            print("Rejected board: scale jump", ratio)
                    if accept:
                        last_board_area = new_area
                        ordered_corners = sort_box_corners(corners)

                        H = cv2.getPerspectiveTransform(
                            ordered_corners, dst_corners)

            detected_rects = None
            if ordered_corners is not None:
                # Draw board

                pts = ordered_corners.astype(int)
                cv2.polylines(overlay, [pts], isClosed=True,
                              color=(0, 0, 255), thickness=3)

                board_view = cv2.warpPerspective(
                    frame, H, (BOARD_W, BOARD_H))

                # table_view = get_aligned_frame(frame, H)

                # card roi stuff
                flat_pts = pts.reshape(-1, 2)
                x_coords = flat_pts[:, 0]

                # Sort x-coordinates from left to right
                x_coords_sorted = np.sort(x_coords)
                y_coords = flat_pts[:, 1]

                top_of_board = int(np.min(y_coords))
                bottom_of_board = int(np.max(y_coords))

                top_of_board = max(0, min(top_of_board, frame.shape[0]))
                bottom_of_board = max(0, min(bottom_of_board, frame.shape[0]))

                trains_ROI = frame[bottom_of_board:, :]
                card_ROI = frame[:top_of_board, :]

                # train color detection and masking
                stack_candidates = find_train_stacks(trains_ROI)

                stacks_valid = validate_train_stacks(
                    stack_candidates, prev_stacks, trains_ROI, bottom_offset=bottom_of_board)

                if stacks_valid is not None:
                    train_stacks = stacks_valid
                    prev_stacks = train_stacks

                for s in train_stacks:
                    cv2.drawContours(
                        overlay, [s['cnt']], -1, (255, 0, 255), 3)

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

            tracked_objects = tracker_manager.update(frame, detected_rects)

            if detected_rects is not None:
                for box in detected_rects:
                    p1 = (int(box[0]), int(box[1]))
                    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                    cv2.rectangle(overlay, p1, p2, (0, 0, 255), 2)

            for obj_id, box in tracked_objects.items():
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))

                cv2.rectangle(overlay, p1, p2, (0, 255, 0), 2)
                cv2.putText(overlay, f"ID: {obj_id}", (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame_idx += 1
            cv2.putText(overlay, f"Frame: {frame_idx}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

            if 'writer' in locals() and writer is not None:
                writer.write(overlay)

            cv2.imshow("Tracking", overlay)
            # cv2.imshow("Aligned view2", mask)

            if cv2.waitKey(25) & 0xFF == 27:  # ESC to exit
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
