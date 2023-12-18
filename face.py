import cv2
from collections import deque

class CircleTracker:
    def __init__(self, N):
        self.history_x = deque(maxlen=N)
        self.history_y = deque(maxlen=N)

    def draw(self, frame, size, color):
        avg_x, avg_y = self.get_avg_position()
        cv2.circle(frame, (avg_x, avg_y), size, color, 3)

    def get_avg_position(self):
        avg_x = int(sum(self.history_x) / len(self.history_x)) if self.history_x else 0
        avg_y = int(sum(self.history_y) / len(self.history_y)) if self.history_y else 0
        return avg_x, avg_y

    def update_history(self, circle):
        self.history_x.append(circle[0])
        self.history_y.append(circle[1])


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
        self.dir_x = 0
        self.dir_y = 0
        self.pos = CircleTracker(N)
        self.left_eye = Eye(N)
        self.right_eye = Eye(N)
        self.vector_direction_history_x = deque(maxlen=N)
        self.vector_direction_history_y = deque(maxlen=N)
        self.vector_distance_history = deque(maxlen=N)

    def update_vector(self):
        avg_dir_x = (self.left_eye.dir_x + self.right_eye.dir_x) // 2
        avg_dir_y = (self.left_eye.dir_y + self.right_eye.dir_y) // 2
        avg_distance = (self.left_eye.distance + self.right_eye.distance) // 2

        self.vector_direction_history_x.append(avg_dir_x)
        self.vector_direction_history_y.append(avg_dir_y)
        self.vector_distance_history.append(avg_distance)

    def get_vector(self):
        avg_dir_x = int(sum(self.vector_direction_history_x) / len(self.vector_direction_history_x)) if self.vector_direction_history_x else 0
        avg_dir_y = int(sum(self.vector_direction_history_y) / len(self.vector_direction_history_y)) if self.vector_direction_history_y else 0
        avg_distance = int(sum(self.vector_distance_history) / len(self.vector_distance_history)) if self.vector_distance_history else 0

        start_x = (self.left_eye.sclera.get_avg_position()[0] + self.right_eye.sclera.get_avg_position()[0]) // 2
        start_y = (self.left_eye.sclera.get_avg_position()[1] + self.right_eye.sclera.get_avg_position()[1]) // 2
        end_x = start_x + avg_dir_x
        end_y = start_y + avg_dir_y

        return start_x, start_y, end_x, end_y

    def draw(self, frame):
        self.left_eye.draw(frame)
        self.right_eye.draw(frame)
        self.pos.draw(frame, 30, (255, 0, 0))
        self.update_vector()

        start_x, start_y, end_x, end_y = self.get_vector()

        cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 0), 2)