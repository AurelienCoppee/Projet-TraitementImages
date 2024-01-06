import cv2
from collections import deque
from enum import Enum
import time
import numpy as np

def update_history_eyes(circles, eyes, N):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0], key=lambda x: x[0])

        def associate_circle_with_eye(circle, eyes):
            for eye in eyes:
                if eye.status == FaceStatus.DESTROYED:
                    continue
                eye_pos = eye.pos.get_avg_position()
                distance = np.sqrt((circle[0] - eye_pos[0])**2 + (circle[1] - eye_pos[1])**2)
                if distance < eye_pos[2]*3:
                    return eye
            return None

        for circle in circles:
            eye = associate_circle_with_eye(circle, eyes)
            if eye is not None:
                if eye.status == EyeStatus.CREATED:
                    eye.update_history(circle)
            else:
                new_eye = Eye(N)
                new_eye.update_history(circle)
                eyes.append(new_eye)

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
        self.status = EyeStatus.CREATED
        self.update_count = 0
        self.created_time = time.time()
        self.update_interval = 1
        self.update_threshold = 4
        self.lost_threshold = 20
        self.lost_count = 0

    def draw(self, frame):
        self.pos.draw(frame, (255, 0, 0))

    def update_history(self, circle):
        self.pos.update_history(circle)
        if self.status == EyeStatus.CREATED:
            self.update_count += 1

    def compute(self, frame):
        if self.status == EyeStatus.CREATED:
            current_time = time.time()
            if current_time - self.created_time > self.update_interval:
                self.status = EyeStatus.DESTROYED

            if self.update_count >= self.update_threshold:
                self.update_count = 0
                self.initialize_tracker(frame)
                self.created_time = time.time()
                self.status = EyeStatus.CIRCLE_VALIDATED
                return
            
        if self.status == EyeStatus.CIRCLE_VALIDATED:
            self.update_tracker(frame)

            
    def initialize_tracker(self, frame):
        avg_x, avg_y, avg_d = self.pos.get_avg_position()
        enlargement_factor = 8
        enlarged_d = int(avg_d * enlargement_factor)
        new_x = int(avg_x - enlarged_d / 2)
        new_y = int(avg_y - enlarged_d / 2)
        new_width = new_height = enlarged_d
        self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.tracker.init(frame, (new_x, new_y, new_width, new_height))

    def update_tracker(self, frame):
        success, box = self.tracker.update(frame)
        if success:
            self.lost_count = 0
            x, y, w, h = map(int, box)

            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            new_x = x + w // 2
            new_y = y + h // 2
            new_d = (w + h) // 2

            self.update_history((new_x, new_y, new_d))
        else:
            if self.lost_count >= self.lost_threshold:
                self.update_count = 0
                self.status = EyeStatus.CREATED
            else:
                self.lost_count += 1

class EyeStatus(Enum):
    DESTROYED = 0
    CREATED = 1
    CIRCLE_VALIDATED = 2

class Face:
    def __init__(self, N):
        self.N = N
        self.pos = CircleTracker(N)
        self.left_eye = Eye(N)
        self.right_eye = Eye(N)
        self.status = FaceStatus.CREATED
        self.created_time = time.time()
        self.update_threshold = 5
        self.lost_threshold = 15
        self.update_interval = 0.25
        self.update_count = 0
        self.lost_count = 0
        self.eyes = []
        self.eyecount = 0

    def compute(self, frame_sent, display_frame):
        frame = frame_sent.copy()
        if self.status == FaceStatus.CREATED:
            # self.pos.draw(display_frame, (0, 0, 255))
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
            # self.pos.draw(display_frame, (0, 255, 0))
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
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    update_history_eyes(circles, self.eyes, self.N)
                    # cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

            # cv2.imshow("Edges", closed_edges)
            # cv2.imshow("Masked Frame", masked_frame)
            current_time = time.time()

            self.eyecount = 0

            for eye in self.eyes:
                eye.compute(frame)
                if eye.status == EyeStatus.CIRCLE_VALIDATED:
                    self.eyecount += 1

            if current_time - self.created_time > self.update_interval:
                self.status = FaceStatus.DESTROYED

            if self.eyecount >= 2:
                self.status = FaceStatus.FACE_VALIDATED
                
            return
        
        if self.status == FaceStatus.FACE_VALIDATED:
            # self.pos.draw(display_frame, (255, 0, 0))
            if self.left_eye.status == EyeStatus.DESTROYED and self.right_eye.status == EyeStatus.DESTROYED:
                self.status = FaceStatus.CREATED
                self.eyes = []
                return
            if self.left_eye.pos.get_avg_position() == (0, 0, 0) and self.right_eye.pos.get_avg_position() == (0, 0, 0):
                self.select_eyes()
                self.load_glasses_sprite()
            self.left_eye.compute(frame)
            self.right_eye.compute(frame)
            self.draw_glasses(display_frame)

    def load_glasses_sprite(self):
        self.glasses_sprite = cv2.imread('Projet-TraitementImages/glasses.png', cv2.IMREAD_UNCHANGED)

    def draw_glasses(self, frame):
            left_eye_center = self.left_eye.pos.get_avg_position()[:2]
            right_eye_center = self.right_eye.pos.get_avg_position()[:2]

            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = -np.degrees(np.arctan2(dy, dx))

            glasses_width = int(abs(dx) * 2.1)
            
            aspect_ratio = self.glasses_sprite.shape[1] / self.glasses_sprite.shape[0]
            glasses_height = int(glasses_width / aspect_ratio)

            resized_sprite = cv2.resize(self.glasses_sprite, (glasses_width, glasses_height))

            M = cv2.getRotationMatrix2D((glasses_width / 2, glasses_height / 2), angle, 1)
            rotated_sprite = cv2.warpAffine(resized_sprite, M, (glasses_width, glasses_height))

            center_x = int((left_eye_center[0] + right_eye_center[0]) / 2)
            center_y = int((left_eye_center[1] + right_eye_center[1]) / 2) + 10
            x_start = center_x - glasses_width // 2
            y_start = center_y - glasses_height // 2

            for y in range(rotated_sprite.shape[0]):
                for x in range(rotated_sprite.shape[1]):
                    if rotated_sprite[y, x, 3] != 0:
                        frame[y_start + y, x_start + x] = rotated_sprite[y, x][:3]

    def select_eyes(self):
        if len(self.eyes) >= 2:
            self.eyes.sort(key=lambda eye: eye.pos.get_avg_position()[0])
            if len(self.eyes) > 2:
                closest_eyes = sorted(self.eyes, key=lambda eye: abs(eye.pos.get_avg_position()[1] - self.eyes[0].pos.get_avg_position()[1]))[:2]
                closest_eyes.sort(key=lambda eye: eye.pos.get_avg_position()[0])
            else:
                closest_eyes = self.eyes

            self.left_eye, self.right_eye = closest_eyes[:2]
    
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
            self.lost_count = 0

            x, y, w, h = map(int, box)

            new_x = x + w // 2
            new_y = y + h // 2
            new_d = (w + h) // 2

            self.update_history((new_x, new_y, new_d))
        else:
            if self.lost_count >= self.lost_threshold:
                self.update_count = 0
                self.eyes = []
                self.status = FaceStatus.CREATED
            else:
                self.lost_count += 1

class FaceStatus(Enum):
    DESTROYED = 0
    CREATED = 1
    CIRCLE_VALIDATED = 2
    FACE_VALIDATED = 3
