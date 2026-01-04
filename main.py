import cv2
import numpy as np

import config
from detectors.board import detect_board
from detectors.cards import detect_cards, classify_card
from detectors.hand import get_hand_mask
from trackers.tracker_manager import TrackerManager
from utils import extract_rotated_card, get_aligned_frame


def main(save_video=False, save_path=''):
    SCALE = 0.75

    cap = cv2.VideoCapture(
        'data/cropped_blue4.mp4')
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 400)

    board_img = cv2.imread('data/ref2.png', 0)
    if board_img is None:
        raise ValueError("error with board reference img")

    BOARD_W, BOARD_H = board_img.shape[:2]

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 900, 600)
    cv2.namedWindow("Aligned view", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Aligned view", 900, 600)
    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contours", 900, 600)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.createTrackbar("Seek", "Tracking", 0, total_frames,
                       lambda x: cap.set(cv2.CAP_PROP_POS_FRAMES, x))

    board_corners = None
    tracker_manager = TrackerManager(max_disappeared=10)
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
                break
            frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            overlay = frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # adding a semi-transparent tint to the detected hand
            hand_mask = get_hand_mask(frame)
            red_layer = np.zeros_like(overlay)
            red_layer[:, :, 2] = hand_mask
            overlay = cv2.addWeighted(overlay, 1.0, red_layer, 0.5, 0)

            # board detection, every 100 frames
            if frame_idx % 100 == 0:
                found_corners = detect_board(
                    frame_gray, board_img, config.ORB, config.BF)
                if found_corners is not None:
                    board_corners = found_corners

            detected_rects = None

            if board_corners is not None:
                # Draw board
                pts = board_corners.astype(int)
                cv2.polylines(overlay, [pts], isClosed=True,
                              color=(0, 0, 255), thickness=3)

                flat_pts = pts.reshape(-1, 2)
                min_y = np.min(flat_pts[:, 1])

                crop_h = int(max(0, min_y))
                crop_h = min(crop_h, frame.shape[0])  # clamp to frame height

                top_view = frame[0:crop_h, :]

                if top_view.size > 0:
                    cv2.imshow("Aligned view", top_view)

                    # detect cards in the top_view
                    if frame_idx % 7 == 0:
                        top_view_gray = cv2.cvtColor(
                            top_view, cv2.COLOR_BGR2GRAY)
                        top_hand_mask = hand_mask[0:crop_h, :]

            detected_rects = detect_cards(
                top_view, hand_mask=None, debug=True)

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
    # try:
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    #         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #         # board detection, every 30 frames
    #         if frame_idx % 100 == 0:
    #             found_corners = detect_board(
    #                 frame_gray, board_img, config.ORB, config.BF)
    #             if found_corners is not None:
    #                 board_corners = found_corners

    #         # drawing the board outline
    #         if board_corners is not None:
    #             pts = board_corners.astype(int)
    #             # cv2.polylines(overlay, [pts], isClosed=True,
    #             #               color=(0, 0, 255), thickness=3)

    #         # board-based aligned view generation
    #         H, _ = cv2.findHomography(board_corners, np.float32(
    #             [[0, 0], [BOARD_W, 0], [BOARD_W, BOARD_H], [0, BOARD_H]]))
    #         # board_view = cv2.warpPerspective(frame, H, (BOARD_W, BOARD_H))

    #         table_view = get_aligned_frame(frame, H)
    #         frame_gray = cv2.cvtColor(table_view, cv2.COLOR_BGR2GRAY)

    #         overlay = table_view.copy()

    #         # adding a semi-transparent tint to the detected hand
    #         hand_mask = get_hand_mask(table_view)
    #         red_layer = np.zeros_like(overlay)
    #         red_layer[:, :, 2] = hand_mask
    #         overlay = cv2.addWeighted(overlay, 1.0, red_layer, 0.5, 0)

    #     # card detection
    #         detected_rects = None
    #         if frame_idx % 7 == 0:
    #             detected_rects = detect_cards(
    #                 frame_gray, hand_mask=hand_mask, debug=True)
    #         tracked_objects = tracker_manager.update(
    #             table_view, detected_rects)

    #         # drawing the card outlines
    #         # for rect in detected_rects:
    #         # box = cv2.boxPoints(rect)
    #         # box = np.int64(rect)
    #         # cv2.drawContours(overlay, [box], 0, (0, 255, 0), 2)
    #         if detected_rects is not None:
    #             for box in detected_rects:
    #                 p1 = (int(box[0]), int(box[1]))
    #                 p2 = (int(box[0] + box[2]), int(box[1] + box[3]))

    #                 cv2.rectangle(overlay, p1, p2, (0, 0, 255), 2)
    #         for obj_id, box in tracked_objects.items():
    #             p1 = (int(box[0]), int(box[1]))
    #             p2 = (int(box[0] + box[2]), int(box[1] + box[3]))

    #             cv2.rectangle(overlay, p1, p2, (0, 255, 0), 2)

    #             cv2.putText(overlay, f"ID: {obj_id}", (p1[0], p1[1] - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #         # classifying the cards
    #         # card_img = extract_rotated_card(box.astype("float32"), frame)
    #         # color_label = classify_card(card_img)
    #         # cv2.putText(overlay, color_label, (box[0][0], box[0][1]), ...

    #         frame_idx += 1
    #         cv2.putText(overlay, f"Frame: {frame_idx}", (10, 100),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    #         if writer is not None:
    #             writer.write(overlay)
    #         cv2.imshow("Tracking", overlay)
    #         # cv2.imshow("Aligned view", table_view)

    #         if cv2.waitKey(25) & 0xFF == 27:  # ESC to exit
    #             break

    # except Exception as e:
    #     print(f"Error: {e}")
    # finally:
    #     cap.release()
    #     if writer is not None:
    #         writer.release()
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    main(save_video=False, save_path='results/ov_test_blue3.mp4')
