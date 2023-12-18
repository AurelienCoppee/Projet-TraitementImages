import cv2
import face
import numpy as np

N = 5
eye_settings = {
    # 146 0 66
    "Lower Eye H": [0, 255], "Lower Eye S": [0, 255], "Lower Eye V": [0, 255],
    # 255 111 203
    "Upper Eye H": [255, 255], "Upper Eye S": [26, 255], "Upper Eye V": [121, 255]
}

iris_settings = {
    # 0 0 38
    "Lower Iris H": [0, 255], "Lower Iris S": [0, 255], "Lower Iris V": [84, 255],
    "Upper Iris H": [255, 255], "Upper Iris S": [255, 255], "Upper Iris V": [255, 255]
}

circle_settings = {
    "dp": [25, 50], "minDist": [65, 500], "param1": [22, 150],
    "param2": [14, 150], "minRadius": [10, 100], "maxRadius": [22, 100]
}

circle_settings_face = {
    "dp": [18, 50], "minDist": [65, 500], "param1": [98, 150],
    "param2": [15, 150], "minRadius": [197, 1000], "maxRadius": [95, 1000]
}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def create_trackbars(window_name, settings):
    cv2.namedWindow(window_name)
    for setting, value in settings.items():
        cv2.createTrackbar(setting, window_name,
                           value[0], value[1], lambda x: None)


def hsv_threshold(frame, settings_name):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    lower = np.array([cv2.getTrackbarPos(f'Lower {settings_name} H', settings_name + ' Settings'),
                      cv2.getTrackbarPos(
        f'Lower {settings_name} S', settings_name + ' Settings'),
        cv2.getTrackbarPos(f'Lower {settings_name} V', settings_name + ' Settings')])

    upper = np.array([cv2.getTrackbarPos(f'Upper {settings_name} H', settings_name + ' Settings'),
                      cv2.getTrackbarPos(
        f'Upper {settings_name} S', settings_name + ' Settings'),
        cv2.getTrackbarPos(f'Upper {settings_name} V', settings_name + ' Settings')])

    return cv2.inRange(hsv_img, lower, upper)


def update_history(circles, left, right):
    if circles is not None:
        avg_pos1_x, avg_pos1_y = left.get_avg_position()
        avg_pos2_x, avg_pos2_y = right.get_avg_position()
        circles = np.uint16(np.around(circles))
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
            left.update_history(circles[pos1_idx])
        if pos2_idx is not None:
            right.update_history(circles[pos2_idx])


def update_history_face(circles):
    if circles is not None:
        avg_pos1_x, avg_pos1_y = face.pos.get_avg_position()
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0], key=lambda x: x[0])
        print("circles", circles)

        def compute_distances(avg_x, avg_y, circles):
            return [np.sqrt((c[0] - avg_x)**2 + (c[1] - avg_y)**2) for c in circles] if avg_x or avg_y else [float("inf")] * len(circles)

        def get_position_indices(avg_pos1):
            dists_pos1 = compute_distances(*avg_pos1, circles)

            pos1_idx = np.argmin(dists_pos1)

            pos1_idx = np.argsort(dists_pos1)[1] if len(
                circles) > 1 else None
            return pos1_idx

        pos1_idx = get_position_indices((avg_pos1_x, avg_pos1_y))

        if pos1_idx is not None:
            print("face :", circles[pos1_idx])
            face.pos.update_history(circles[pos1_idx])


def detect_eyes_and_faces(frame, last_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if last_frame is not None:
        frame_diff = cv2.absdiff(last_frame, frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        thresh_diff = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)[1]
        blur_diff = cv2.blur(thresh_diff, (5, 5))
        circles = cv2.HoughCircles(blur_diff, method=cv2.HOUGH_GRADIENT, dp=cv2.getTrackbarPos("dp", "Circle Settings Face")/10,
                                   minDist=cv2.getTrackbarPos("minDist", "Circle Settings Face"), param1=cv2.getTrackbarPos("param1", "Circle Settings Face"), param2=cv2.getTrackbarPos("param2", "Circle Settings Face"), minRadius=cv2.getTrackbarPos("minRadius", "Circle Settings Face"), maxRadius=cv2.getTrackbarPos("maxRadius", "Circle Settings Face"))

        if circles is not None:
            circles = np.uint16(np.around(circles))
            update_history_face(circles)
            for i in circles[0, :]:
                cv2.circle(frame_diff, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame_diff, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('Face detection', frame_diff)

    eyes_detected = []

    if len(faces) == 1:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                eyes_detected.extend([(ex + x, ey + y, ew, eh)
                                     for (ex, ey, ew, eh) in eyes])

    return eyes_detected


def process_and_mask_eyes(frame, eyes):
    eyes_only = np.zeros((1024, 1280, 3), dtype=np.uint8)

    for (ex, ey, ew, eh) in eyes:
        src_region = frame[ey:ey + eh, ex:ex + ew]
        dst_shape = eyes_only[ey:ey + eh, ex:ex + ew].shape

        if src_region.shape == dst_shape:
            eyes_only[ey:ey + eh, ex:ex + ew] = src_region

    mask_eye = hsv_threshold(eyes_only, "Eye")
    mask_iris = hsv_threshold(eyes_only, "Iris")

    return eyes_only, mask_eye, mask_iris


create_trackbars("Eye Settings", eye_settings)
create_trackbars("Iris Settings", iris_settings)
create_trackbars("Circle Settings", circle_settings)
create_trackbars("Circle Settings Face", circle_settings_face)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

face = face.Face(N)

last_frame = None

while True:
    print("Run")
    ret, frame = cam.read()

    if ret:
        # frame = cv2.flip(frame, -1)  # Flip camera vertically
        display_frame = frame.copy()
        eyes_detected = detect_eyes_and_faces(frame, last_frame)
        last_frame = display_frame

        if eyes_detected:
            eyes_only, mask_sclera, mask_pupil = process_and_mask_eyes(
                frame, eyes_detected)

            inverted_mask_pupil = cv2.bitwise_not(mask_pupil)
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(inverted_mask_pupil,
                                kernel, iterations=4)
            dilation = cv2.dilate(erosion, kernel, iterations=3)
            closing = cv2.morphologyEx(
                mask_sclera, cv2.MORPH_CLOSE, kernel)

            masked_sclera = cv2.bitwise_and(
                eyes_only, eyes_only, mask=closing)
            masked_pupil = cv2.bitwise_and(
                eyes_only, eyes_only, mask=dilation)

            grey_pupil = cv2.cvtColor(
                masked_pupil, cv2.COLOR_BGR2GRAY)
            blur_pupil = cv2.blur(grey_pupil, (5, 5))

            grey_sclera = cv2.cvtColor(
                masked_sclera, cv2.COLOR_BGR2GRAY)
            blur_sclera = cv2.blur(grey_sclera, (5, 5))

            circles_pupil = cv2.HoughCircles(blur_pupil, method=cv2.HOUGH_GRADIENT, dp=cv2.getTrackbarPos("dp", "Circle Settings")/10,
                                             minDist=cv2.getTrackbarPos("minDist", "Circle Settings"), param1=cv2.getTrackbarPos("param1", "Circle Settings"), param2=cv2.getTrackbarPos("param2", "Circle Settings"), minRadius=cv2.getTrackbarPos("minRadius", "Circle Settings"), maxRadius=cv2.getTrackbarPos("maxRadius", "Circle Settings"))

            circles_sclera = cv2.HoughCircles(blur_sclera, method=cv2.HOUGH_GRADIENT, dp=cv2.getTrackbarPos("dp", "Circle Settings")/10,
                                              minDist=cv2.getTrackbarPos("minDist", "Circle Settings"), param1=cv2.getTrackbarPos("param1", "Circle Settings"), param2=cv2.getTrackbarPos("param2", "Circle Settings"), minRadius=cv2.getTrackbarPos("minRadius", "Circle Settings"), maxRadius=cv2.getTrackbarPos("maxRadius", "Circle Settings"))

            if circles_pupil is not None:
                circles_pupil = np.uint16(np.around(circles_pupil))
                for i in circles_pupil[0, :]:
                    cv2.circle(
                        masked_pupil, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(masked_pupil, (i[0], i[1]), 2, (0, 0, 255), 3)

            if circles_sclera is not None:
                circles_sclera = np.uint16(np.around(circles_sclera))
                for i in circles_sclera[0, :]:
                    cv2.circle(masked_sclera,
                               (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(masked_sclera, (i[0], i[1]), 2, (0, 0, 255), 3)

            cv2.imshow('Masked Sclera', masked_sclera)
            cv2.imshow('Masked Pupil', masked_pupil)

            update_history(circles_pupil, face.left_eye.pupil,
                           face.right_eye.pupil)
            update_history(circles_sclera, face.left_eye.sclera,
                           face.right_eye.sclera)

        face.draw(display_frame)
        cv2.imshow('Detected', display_frame)

        if cv2.waitKey(5) == 27 or cv2.getWindowProperty('Detected', cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()
cam.release()
