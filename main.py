import cv2
from face import Face, FaceStatus
import numpy as np

N = 3
faces = []

edges = {
    "Min": [82, 1000], "Max": [116, 1000],
}

circle_settings = {
    "dp": [2, 50], "minDist": [100, 500], "param1": [100, 150],
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

def update_history(circles, left, right):
    if circles is not None:
        avg_pos1_x, avg_pos1_y = left.get_avg_position()
        avg_pos2_x, avg_pos2_y = right.get_avg_position()
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
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0], key=lambda x: x[0])

        def associate_circle_with_face(circle, faces):
            for face in faces:
                if face.status == FaceStatus.DESTROYED:
                    continue
                face_pos = face.pos.get_avg_position()
                distance = np.sqrt((circle[0] - face_pos[0])**2 + (circle[1] - face_pos[1])**2)
                if distance < face_pos[2]*1.5:
                    return face
            return None

        for circle in circles:
            face = associate_circle_with_face(circle, faces)
            if face is not None:
                if face.status != FaceStatus.CIRCLE_VALIDATED:
                    face.update_history(circle)
            else:
                new_face = Face(N)
                new_face.update_history(circle)
                faces.append(new_face)


def detect_faces(frame, last_frame):
    if last_frame is not None:
        frame_diff = cv2.absdiff(last_frame, frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        blur_diff = cv2.medianBlur(gray_diff, 1)
        erosion = cv2.erode(blur_diff, kernel, iterations=1)
        thresh_diff = cv2.threshold(erosion, 10, 255, cv2.THRESH_BINARY)[1]
        circles = cv2.HoughCircles(thresh_diff, method=cv2.HOUGH_GRADIENT, dp=cv2.getTrackbarPos("dp", "Circle Settings Face")/10,
                                   minDist=cv2.getTrackbarPos("minDist", "Circle Settings Face"), param1=cv2.getTrackbarPos("param1", "Circle Settings Face"), param2=cv2.getTrackbarPos("param2", "Circle Settings Face"), minRadius=cv2.getTrackbarPos("minRadius", "Circle Settings Face"), maxRadius=cv2.getTrackbarPos("maxRadius", "Circle Settings Face"))

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # for i in circles[0, :]:
            #     cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #     cv2.circle(display_frame, (i[0], i[1]), 2, (0, 0, 255), 3)

            update_history_face(circles)

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


create_trackbars("Edges", edges)
create_trackbars("Circle Settings", circle_settings)
create_trackbars("Circle Settings Face", circle_settings_face)

cam = cv2.VideoCapture(2)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920/2)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080/2)

last_frame = None

while True:
    print("Run")
    ret, frame = cam.read()

    if ret:
        display_frame = frame.copy()
        detect_faces(frame, last_frame)
        last_frame = frame.copy()

        # if eyes_detected:
        #     eyes_only, mask_sclera, mask_pupil = process_and_mask_eyes(
        #         frame, eyes_detected)

        #     inverted_mask_pupil = cv2.bitwise_not(mask_pupil)
        #     kernel = np.ones((3, 3), np.uint8)
        #     erosion = cv2.erode(inverted_mask_pupil,
        #                         kernel, iterations=4)
        #     dilation = cv2.dilate(erosion, kernel, iterations=3)
        #     closing = cv2.morphologyEx(
        #         mask_sclera, cv2.MORPH_CLOSE, kernel)

        #     masked_sclera = cv2.bitwise_and(
        #         eyes_only, eyes_only, mask=closing)
        #     masked_pupil = cv2.bitwise_and(
        #         eyes_only, eyes_only, mask=dilation)

        #     grey_pupil = cv2.cvtColor(
        #         masked_pupil, cv2.COLOR_BGR2GRAY)
        #     blur_pupil = cv2.blur(grey_pupil, (5, 5))

        #     grey_sclera = cv2.cvtColor(
        #         masked_sclera, cv2.COLOR_BGR2GRAY)
        #     blur_sclera = cv2.blur(grey_sclera, (5, 5))

        #     circles_pupil = cv2.HoughCircles(blur_pupil, method=cv2.HOUGH_GRADIENT, dp=cv2.getTrackbarPos("dp", "Circle Settings")/10,
        #                                      minDist=cv2.getTrackbarPos("minDist", "Circle Settings"), param1=cv2.getTrackbarPos("param1", "Circle Settings"), param2=cv2.getTrackbarPos("param2", "Circle Settings"), minRadius=cv2.getTrackbarPos("minRadius", "Circle Settings"), maxRadius=cv2.getTrackbarPos("maxRadius", "Circle Settings"))

        #     circles_sclera = cv2.HoughCircles(blur_sclera, method=cv2.HOUGH_GRADIENT, dp=cv2.getTrackbarPos("dp", "Circle Settings")/10,
        #                                       minDist=cv2.getTrackbarPos("minDist", "Circle Settings"), param1=cv2.getTrackbarPos("param1", "Circle Settings"), param2=cv2.getTrackbarPos("param2", "Circle Settings"), minRadius=cv2.getTrackbarPos("minRadius", "Circle Settings"), maxRadius=cv2.getTrackbarPos("maxRadius", "Circle Settings"))

        #     if circles_pupil is not None:
        #         circles_pupil = np.uint16(np.around(circles_pupil))
        #         for i in circles_pupil[0, :]:
        #             cv2.circle(
        #                 masked_pupil, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #             cv2.circle(masked_pupil, (i[0], i[1]), 2, (0, 0, 255), 3)

        #     if circles_sclera is not None:
        #         circles_sclera = np.uint16(np.around(circles_sclera))
        #         for i in circles_sclera[0, :]:
        #             cv2.circle(masked_sclera,
        #                        (i[0], i[1]), i[2], (0, 255, 0), 2)
        #             cv2.circle(masked_sclera, (i[0], i[1]), 2, (0, 0, 255), 3)

        #     cv2.imshow('Masked Sclera', masked_sclera)
        #     cv2.imshow('Masked Pupil', masked_pupil)

        #     update_history(circles_pupil, face.left_eye.pupil,
        #                    face.right_eye.pupil)
        #     update_history(circles_sclera, face.left_eye.sclera,
        #                    face.right_eye.sclera)
        for face in faces:
            face.compute(display_frame)

        cv2.imshow('Detected', display_frame)

        if cv2.waitKey(5) == 27 or cv2.getWindowProperty('Detected', cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()
cam.release()
