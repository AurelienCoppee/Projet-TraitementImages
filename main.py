import cv2
from face import Face, FaceStatus
import numpy as np

N = 2
faces = []

edges = {
    "Min": [90, 1000], "Max": [59, 1000],
}

circle_settings = {
    "dp": [4, 50], "minDist": [50, 500], "param1": [100, 150],
    "param2": [16, 150], "minRadius": [5, 100], "maxRadius": [22, 100]
}

circle_settings_face = {
    "dp": [2, 50], "minDist": [400, 500], "param1": [100, 150],
    "param2": [31, 150], "minRadius": [60, 1000], "maxRadius": [300, 1000]
}

def create_trackbars(window_name, settings):
    cv2.namedWindow(window_name)
    for setting, value in settings.items():
        cv2.createTrackbar(setting, window_name,
                           value[0], value[1], lambda x: None)


def update_history_face(circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0], key=lambda x: x[0])

        def associate_circle_with_face(circle, faces):
            for face in faces:
                if face.status == FaceStatus.DESTROYED:
                    continue
                face_pos = face.pos.get_avg_position()
                distance = np.sqrt((circle[0] - face_pos[0])**2 + (circle[1] - face_pos[1])**2)
                if distance < face_pos[2]:
                    return face
            return None

        for circle in circles:
            face = associate_circle_with_face(circle, faces)
            if face is not None:
                if face.status == FaceStatus.CREATED:
                    face.update_history(circle)
            else:
                new_face = Face(N)
                new_face.update_history(circle)
                faces.append(new_face)


def detect_faces(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_frame, method=cv2.HOUGH_GRADIENT, dp=cv2.getTrackbarPos("dp", "Circle Settings Face")/10,
                                   minDist=cv2.getTrackbarPos("minDist", "Circle Settings Face"), param1=cv2.getTrackbarPos("param1", "Circle Settings Face"), param2=cv2.getTrackbarPos("param2", "Circle Settings Face"), minRadius=cv2.getTrackbarPos("minRadius", "Circle Settings Face"), maxRadius=cv2.getTrackbarPos("maxRadius", "Circle Settings Face"))

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # for i in circles[0, :]:
            #     cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #     cv2.circle(display_frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            update_history_face(circles)


create_trackbars("Edges", edges)
create_trackbars("Circle Settings", circle_settings)
create_trackbars("Circle Settings Face", circle_settings_face)

cam = cv2.VideoCapture(2)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920/2)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080/2)


while True:
    ret, frame = cam.read()

    if ret:
        display_frame = frame.copy()
        detect_faces(frame)

        for face in faces:
            face.compute(frame, display_frame)

        faces = [face for face in faces if face.status != FaceStatus.DESTROYED]

        cv2.imshow('Glasses fitting', display_frame)

        if cv2.waitKey(5) == 27 or cv2.getWindowProperty('Glasses fitting', cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()
cam.release()
