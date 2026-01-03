import cv2
import numpy as np


class TrackerManager:
    def __init__(self, max_disappeared=10):
        # Dictionary to store trackers: { object_id: tracker_instance }
        self.trackers = {}
        # Dictionary to store lost frames count: { object_id: count }
        self.disappeared = {}

        self.next_object_id = 0
        self.max_disappeared = max_disappeared  # How many frames to keep a lost object

    def _create_tracker(self, frame, bbox):
        """Helper to initialize a single tracker"""
        tracker = cv2.legacy.TrackerCSRT.create()
        tracker.init(frame, bbox)
        return tracker

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, frame, detected_boxes=None):
        """
        frame: current video frame
        detected_boxes: list of (x, y, w, h) from your detect_cards function. 
                        Can be None if detection didn't run this frame.
        """
        active_ids = list(self.trackers.keys())
        current_tracked_boxes = {}

        to_delete = []

        for obj_id in active_ids:
            success, box = self.trackers[obj_id].update(frame)
            if success:
                current_tracked_boxes[obj_id] = tuple(map(int, box))
            else:
                to_delete.append(obj_id)

        # Remove failed trackers immediately
        for obj_id in to_delete:
            self.deregister(obj_id)

        if detected_boxes is not None:
            # Mark all current trackers as potentially disappeared
            # We will unmark them if we match them to a detection
            matched_tracker_ids = set()

            for det_box in detected_boxes:
                best_iou = 0
                best_match_id = -1

                # Find the tracker that overlaps most with this detection
                for obj_id, track_box in current_tracked_boxes.items():
                    iou = self._calculate_iou(det_box, track_box)
                    if iou > 0.5:
                        if iou > best_iou:
                            best_iou = iou
                            best_match_id = obj_id

                if best_match_id != -1:  # found a match
                    matched_tracker_ids.add(best_match_id)
                    self.disappeared[best_match_id] = 0

                else:
                    # no match so we start a new tracker
                    self.register(frame, det_box)

            # trackers that didnt match any new detection incrementation of the disappeared counter
            for obj_id in list(self.trackers.keys()):
                if obj_id not in matched_tracker_ids:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)

        return current_tracked_boxes

    def register(self, frame, bbox):
        print(f"Registered new object ID: {self.next_object_id}")
        self.trackers[self.next_object_id] = self._create_tracker(frame, bbox)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, obj_id):
        print(f"Deregistered object ID: {obj_id}")
        del self.trackers[obj_id]
        del self.disappeared[obj_id]
