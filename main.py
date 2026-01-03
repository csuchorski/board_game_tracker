import cv2
import numpy as np

import config
from detectors.board import detect_board
from detectors.cards import detect_cards, classify_card
from detectors.hand import get_hand_mask
from trackers.tracker_manager import TrackerManager
from utils import extract_rotated_card


def main():
    SCALE = 0.75

    cap = cv2.VideoCapture('data/easy1_start.mp4')
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 400)

    board_img = cv2.imread('data/board_reference.jpg', 0)
    if board_img is None:
        raise ValueError("error with board reference img")

    BOARD_W, BOARD_H = board_img.shape[:2]

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 900, 600)
    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contours", 900, 600)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.createTrackbar("Seek", "Tracking", 0, total_frames,
                       lambda x: cap.set(cv2.CAP_PROP_POS_FRAMES, x))

    board_corners = None
    tracker_manager = TrackerManager(max_disappeared=10)
    frame_idx = 0

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

            # hand_mask = cv2.bitwise_not(hand_mask)
            # masked_frame_gray = cv2.bitwise_and(
            #     frame_gray, frame_gray, mask=hand_mask)

            # board detection, every 30 frames
            if frame_idx % 100 == 0:
                found_corners = detect_board(
                    frame_gray, board_img, config.ORB, config.BF)
                if found_corners is not None:
                    board_corners = found_corners

            # drawing the board outline
            if board_corners is not None:
                pts = board_corners.astype(int)
                cv2.polylines(overlay, [pts], isClosed=True,
                              color=(0, 0, 255), thickness=3)

                # masking the board to make the card contour detection easier
                # mask = np.ones(frame_gray.shape, dtype=np.uint8) * 255
                # cv2.fillPoly(mask, [board_corners.astype(int)], 0)
                # masked_frame_gray = cv2.bitwise_and(
                #     masked_frame_gray, frame_gray, mask=mask)

                # card detection
            detected_rects = None
            if frame_idx % 7 == 0:
                detected_rects = detect_cards(
                    frame_gray, hand_mask=hand_mask, debug=True)
            tracked_objects = tracker_manager.update(frame, detected_rects)

            # drawing the card outlines
            # for rect in detected_rects:
            # box = cv2.boxPoints(rect)
            # box = np.int64(rect)
            # cv2.drawContours(overlay, [box], 0, (0, 255, 0), 2)
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

            # classifying the cards
            # card_img = extract_rotated_card(box.astype("float32"), frame)
            # color_label = classify_card(card_img)
            # cv2.putText(overlay, color_label, (box[0][0], box[0][1]), ...

            frame_idx += 1
            cv2.putText(overlay, f"Frame: {frame_idx}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            cv2.imshow("Tracking", overlay)

            if cv2.waitKey(25) & 0xFF == 27:  # ESC to exit
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
