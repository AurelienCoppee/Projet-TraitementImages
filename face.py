import cv2
from collections import deque
from enum import Enum
import time
import numpy as np

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
        self.sclera = CircleTracker(N)
        self.pupil = CircleTracker(N)
        self.dir_x = 0
        self.dir_y = 0
        self.distance = 0

    def draw(self, frame):
        self.sclera.draw(frame, 3, (0, 255, 0))
        self.pupil.draw(frame, 3, (0, 0, 255))
        self.get_vector()
        cv2.line(frame, self.sclera.get_avg_position(), self.pupil.get_avg_position(), (255, 0, 0), 2)

    def get_vector(self):
        start_point = self.sclera.get_avg_position()
        end_point = self.pupil.get_avg_position()

        self.dir_x = end_point[0] - start_point[0]
        self.dir_y = end_point[1] - start_point[1]

        self.distance = int((self.dir_x**2 + self.dir_y**2)**0.5)

class Face:
    def __init__(self, N):
        self.pos = CircleTracker(N)
        self.status = FaceStatus.CREATED
        self.created_time = time.time()
        self.update_threshold = 8
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
                print("test")
                self.initialize_tracker(frame)
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
            edges = cv2.Canny(gray_masked, cv2.getTrackbarPos("Min", "Edges"), cv2.getTrackbarPos("Max", "Edges"))

            circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=cv2.getTrackbarPos("dp", "Circle Settings")/10,
                                    minDist=cv2.getTrackbarPos("minDist", "Circle Settings"),
                                    param1=cv2.getTrackbarPos("param1", "Circle Settings"),
                                    param2=cv2.getTrackbarPos("param2", "Circle Settings"),
                                    minRadius=cv2.getTrackbarPos("minRadius", "Circle Settings"),
                                    maxRadius=cv2.getTrackbarPos("maxRadius", "Circle Settings"))

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(masked_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(masked_frame, (i[0], i[1]), 2, (0, 0, 255), 3)

            cv2.imshow("Edges", edges)
            cv2.imshow("Masked Frame", masked_frame)
            return
    
    def update_history(self, circle):
        self.pos.update_history(circle)
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
                self.status = FaceStatus.CREATED
            else:
                self.lost_count += 1

class FaceStatus(Enum):
    DESTROYED = 0
    CREATED = 1
    CIRCLE_VALIDATED = 2
