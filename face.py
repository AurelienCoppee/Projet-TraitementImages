import cv2
from collections import deque
from enum import Enum
import time
import numpy as np

def update_history_eyes(circles, left, right):
    if circles is not None:
        avg_pos1_x, avg_pos1_y, avg_d1 = left.pos.get_avg_position()
        avg_pos2_x, avg_pos2_y, avg_d2 = right.pos.get_avg_position()
        circles = sorted(circles[0], key=lambda x: x[0])

        def compute_distances(avg_x, avg_y, circles):
            return [np.sqrt((c[0] - avg_x)**2 + (c[1] - avg_y)**2) for c in circles] if avg_x or avg_y else [float("inf")] * len(circles)

        def get_position_indices(avg_pos1, avg_pos2):
            dists_pos1 = compute_distances(*avg_pos1, circles)
            dists_pos2 = compute_distances(*avg_pos2, circles)

            pos1_idx = np.argmin(dists_pos1)
            pos2_idx = np.argmin(dists_pos2)

            if pos1_idx == pos2_idx:
                if dists_pos1[pos1_idx] <= dists_pos2[pos2_idx]:
                    pos2_idx = np.argsort(dists_pos2)[
                        1] if len(circles) > 1 else None
                else:
                    pos1_idx = np.argsort(dists_pos1)[1] if len(
                        circles) > 1 else None
            return pos1_idx, pos2_idx

        pos1_idx, pos2_idx = get_position_indices(
            (avg_pos1_x, avg_pos1_y), (avg_pos2_x, avg_pos2_y))

        if pos1_idx is not None and pos2_idx is not None:
            if circles[pos1_idx][0] > circles[pos2_idx][0]:
                pos1_idx, pos2_idx = pos2_idx, pos1_idx

        if pos1_idx is not None:
            left.pos.update_history(circles[pos1_idx])
        if pos2_idx is not None:
            right.pos.update_history(circles[pos2_idx])

class CircleTracker:
    def __init__(self, N):
        self.history_x = deque(maxlen=N)
        self.history_y = deque(maxlen=N)
        self.history_d = deque(maxlen=N)

    def draw(self, frame, color):
        avg_x, avg_y, avg_d = self.get_avg_position()
        cv2.circle(frame, (avg_x, avg_y), int(avg_d/2), color, 3)

    def get_avg_position(self):
        avg_x = int(sum(self.history_x) / len(self.history_x)) if self.history_x else 0
        avg_y = int(sum(self.history_y) / len(self.history_y)) if self.history_y else 0
        avg_d = int(sum(self.history_d) / len(self.history_d)) if self.history_d else 0
        return avg_x, avg_y, avg_d

    def update_history(self, circle):
        self.history_x.append(circle[0])
        self.history_y.append(circle[1])
        self.history_d.append(circle[2])

class Eye:
    def __init__(self, N):
        self.pos = CircleTracker(N)

    def draw(self, frame):
        self.pos.draw(frame, (255, 0, 0))

class Face:
    def __init__(self, N):
        self.pos = CircleTracker(N)
        self.left_eye = Eye(N)
        self.right_eye = Eye(N)
        self.status = FaceStatus.CREATED
        self.created_time = time.time()
        self.update_threshold = 8
        self.eyes_threshold = 8
        self.lost_threshold = 15
        self.update_interval = 1
        self.update_count = 0
        self.lost_count = 0

    def compute(self, frame):
        if self.status == FaceStatus.CREATED:
            current_time = time.time()
            if current_time - self.created_time > self.update_interval:
                self.status = FaceStatus.DESTROYED

            if self.update_count >= self.update_threshold:
                self.update_count = 0
                self.initialize_tracker(frame)
                self.created_time = time.time()
                self.status = FaceStatus.CIRCLE_VALIDATED
                return
        if self.status == FaceStatus.CIRCLE_VALIDATED:
            self.update_tracker(frame)

            avg_x, avg_y, avg_d = self.pos.get_avg_position()
            radius = int(avg_d / 2)
            center = (int(avg_x), int(avg_y))
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(mask, center, radius, 255, -1)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_masked, (5,5), 0)
            kernel = np.ones((3,3), np.uint8)
            erosion = cv2.erode(blurred, kernel, iterations=2)
            edges = cv2.Canny(erosion, cv2.getTrackbarPos("Min", "Edges"), cv2.getTrackbarPos("Max", "Edges"))

            kernel = np.ones((3,3), np.uint8)
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
            circle_mask = np.zeros_like(gray_masked)

            # center_x, center_y = masked_frame.shape[1] // 2, masked_frame.shape[0] // 2
            # cv2.circle(masked_frame, (center_x, center_y), int(avg_d/100), (0, 255, 0), 2)
            # cv2.circle(masked_frame, (center_x, center_y), int(avg_d/4), (0, 255, 0), 2)

            for contour in contours:
                area = cv2.contourArea(contour)
                if  avg_d/4 > area > avg_d/100:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter != 0:
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        if 0 < circularity < 0.1: 
                            cv2.drawContours(circle_mask, [contour], -1, 255, 2)
                            cv2.drawContours(masked_frame, [contour], -1, (0, 255, 0), 2)

            circles = cv2.HoughCircles(circle_mask, method=cv2.HOUGH_GRADIENT, dp=cv2.getTrackbarPos("dp", "Circle Settings")/10,
                                    minDist=cv2.getTrackbarPos("minDist", "Circle Settings"),
                                    param1=cv2.getTrackbarPos("param1", "Circle Settings"),
                                    param2=cv2.getTrackbarPos("param2", "Circle Settings"),
                                    minRadius=cv2.getTrackbarPos("minRadius", "Circle Settings"),
                                    maxRadius=cv2.getTrackbarPos("maxRadius", "Circle Settings"))

            if circles is not None:
                self.update_count += 1
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    print(circles)
                    print(self.update_count)
                    update_history_eyes(circles, self.left_eye, self.right_eye)
                    cv2.circle(masked_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(masked_frame, (i[0], i[1]), 2, (0, 0, 255), 3)

            cv2.imshow('Detected Circles on Contours', circle_mask)
            cv2.imshow("Edges", closed_edges)
            cv2.imshow("Masked Frame", masked_frame)
            current_time = time.time()
            print(self.update_count)
            if current_time - self.created_time > self.update_interval:
                self.status = FaceStatus.DESTROYED

            if self.update_count >= self.eyes_threshold:
                # self.initialize_tracker(frame)
                print(self.left_eye.pos.get_avg_position())
                print(self.right_eye.pos.get_avg_position())
                if self.left_eye.pos.get_avg_position() != (0,0,0) and self.right_eye.pos.get_avg_position() != (0,0,0):
                    self.status = FaceStatus.FACE_VALIDATED
                
            return
        
        if self.status == FaceStatus.FACE_VALIDATED:
            print("vali")
            self.update_tracker(frame)
            self.left_eye.draw(frame)
            self.right_eye.draw(frame)
    
    def update_history(self, circle):
        self.pos.update_history(circle)
        if self.status == FaceStatus.CREATED:
            self.update_count += 1

    def initialize_tracker(self, frame):
        avg_x, avg_y, avg_d = self.pos.get_avg_position()
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, (int(avg_x - avg_d), int(avg_y - avg_d), int(avg_d*2), int(avg_d*2)))

    def update_tracker(self, frame):
        success, box = self.tracker.update(frame)
        if success:
            x, y, w, h = map(int, box)

            new_x = x + w // 2
            new_y = y + h // 2
            new_d = (w + h) // 2

            self.update_history((new_x, new_y, new_d))
        else:
            if self.lost_count >= self.lost_threshold:
                self.update_count = 0
                self.status = FaceStatus.DESTROYED
            else:
                self.lost_count += 1

class FaceStatus(Enum):
    DESTROYED = 0
    CREATED = 1
    CIRCLE_VALIDATED = 2
    FACE_VALIDATED = 3
