import cv2
import dlib

import numpy as np


def get_face_landmarks(photo_path, save_path):
    """
    generate and normalize landmarks from 2d image
    """
    photo = cv2.imread(photo_path, cv2.IMREAD_UNCHANGED)

    rbg_img = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    points_keys = []
    PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    rects = detector(gray, 1)
    assert len(rects) == 1

    landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, rects[0]).parts()])
    landmarks_colors = np.matrix([rbg_img[p.y, p.x, :3] for p in predictor(gray, rects[0]).parts()])
    # normalized_landmarks = np.matrix(
    #     [[float(p.x) * 0.18 - 105, -(float(p.y) * 0.18 - 140)] for p in predictor(gray, rects[0]).parts()])
    normalized_landmarks = (landmarks - np.mean(landmarks, axis=0))
    normalized_landmarks[:, 1] = -normalized_landmarks[:, 1]
    # normalized_landmarks = normalized_landmarks / max(gray.shape[0], gray.shape[1]) * 256
    normalized_landmarks = normalized_landmarks / np.max(normalized_landmarks)

    photo_with_lm = photo.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        points_keys.append(pos)
        cv2.circle(photo_with_lm, pos, 10, (255, 0, 0), -1)
    cv2.imwrite(save_path, photo_with_lm)
    return photo, photo_with_lm, normalized_landmarks, np.array(landmarks_colors).T
