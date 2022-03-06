"""
Use 3DMM to fit 2d pictures with BFM model
"""


import cv2
import dlib
import os
import subprocess
from skimage import img_as_ubyte

import numpy as np
from skimage import io

from face3d import mesh
from face3d.morphable_model import MorphabelModel


def generate_key_points(img):
    """generate key_points from 2d image"""
    rbg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    points_keys = []
    PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    rects = detector(gray, 1)
    assert len(rects) == 1

    for j in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, rects[j]).parts()])
        landmarks_colors = np.matrix([rbg_img[p.y, p.x, :3] for p in predictor(gray, rects[j]).parts()])
        normalized_img = np.matrix([[float(p.x) * 0.18 - 105,
                                   -(float(p.y) * 0.18 - 140)] for p in predictor(gray, rects[j]).parts()])
        img = img.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            points_keys.append(pos)
            cv2.circle(img, pos, 10, (255, 0, 0), -1)
        return img, normalized_img, np.array(landmarks_colors).T


def main():
    picture = cv2.imread("./data/2d_face/example.png", cv2.IMREAD_UNCHANGED)
    face_key, key_point, face_colors = generate_key_points(picture)
    cv2.imwrite('./results/3dmm/facekey.jpg', face_key)

    # ----- Forward: parameters(shape, expression, pose) --> 3D obj --> 2D image  ---------------
    # --- 1. load model
    bfm = MorphabelModel('data/BFM/Out/BFM.mat')
    print('init bfm model success')

    # --- 2. generate face mesh: vertices(represent shape) & colors(represent texture)
    sp = bfm.get_shape_para('zero')
    ep = bfm.get_exp_para('zero')
    vertices = bfm.generate_vertices(sp, ep)

    tp = bfm.get_tex_para('zero')
    colors = bfm.generate_colors(tp)
    colors = np.minimum(np.maximum(colors, 0), 1)

    # --- 3. transform vertices to proper position
    s = 8e-04
    angles = [0, 0, 0]
    t = [0, 0, 0]
    transformed_vertices = bfm.transform(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection

    # --- 4. render(3d obj --> 2d image)
    # set prop of rendering
    h = w = 256
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

    # ----- Back:  2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression) ------
    # only use 68 key points to fit
    # default_key_x = projected_vertices[bfm.kpt_ind, :2]  # 2d keypoint, which can be detected from image
    x = np.array(key_point)

    X_ind = bfm.kpt_ind  # index of keypoints in 3DMM. fixed.

    # fit
    fitted_sp, fitted_ep, fitted_tp, fitted_s, fitted_angles, fitted_t = bfm.fit(
        x, X_ind, face_colors, max_iter=20)

    colors = bfm.generate_colors(fitted_tp)
    # colors = np.minimum(np.maximum(colors, 0), 1)

    # verify fitted parameters
    fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
    transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)

    image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
    fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

    rotate_vertices = bfm.transform(transformed_vertices, 1, [10, 30, 10], [0, 0, 0])
    rotate_vertices = mesh.transform.to_image(rotate_vertices, h, w)
    rotated = mesh.render.render_colors(rotate_vertices, bfm.triangles, colors, h, w)

    # project picture color to face model
    img = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
    colors = np.zeros_like(colors)
    depth_buffer = np.zeros([h, w], dtype=np.float32) - 999999.
    index_buffer = np.zeros([h, w])
    for i, pos in enumerate(transformed_vertices):
        height, width, depth = int(pos[0]), int(pos[1]), pos[2]
        if depth >= depth_buffer[height, width]:
            index_buffer[height + h // 2, width + w // 2] = i
            depth_buffer[height + h // 2, width + w // 2] = depth
            transformed_w = int((height + 105.) / 0.18)
            transformed_h = int((-width + 140.) / 0.18)
            assert 0 <= transformed_h < img.shape[0] and 0 <= transformed_w < img.shape[1]
            colors[i] = img[transformed_h, transformed_w, :] / 255.
    real_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
    real_rotated_image = mesh.render.render_colors(rotate_vertices, bfm.triangles, colors, h, w)

    # --- print & show
    # print('pose, groudtruth:\n', s, angles[0], angles[1], angles[2], t[0], t[1])
    # print('pose, fitted:\n', fitted_s, fitted_angles[0], fitted_angles[1], fitted_angles[2], fitted_t[0], fitted_t[1])

    save_folder = 'results/3dmm'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    io.imsave('{}/generated.jpg'.format(save_folder), img_as_ubyte(image))
    io.imsave('{}/fitted.jpg'.format(save_folder), img_as_ubyte(fitted_image))
    io.imsave('{}/rotated.jpg'.format(save_folder), img_as_ubyte(rotated))
    io.imsave('{}/real.jpg'.format(save_folder), img_as_ubyte(real_image))
    io.imsave('{}/real_rotated.jpg'.format(save_folder), img_as_ubyte(real_rotated_image))

    # --- visualize fitting process
    # fit
    fitted_sp, fitted_ep, fitted_tp, fitted_s, fitted_angles, fitted_t = bfm.fit(
        x, X_ind, face_colors, max_iter=20, isShow=True)

    # verify fitted parameters
    for i in range(fitted_sp.shape[0]):
        fitted_vertices = bfm.generate_vertices(fitted_sp[i], fitted_ep[i])
        transformed_vertices = bfm.transform(fitted_vertices, fitted_s[i], fitted_angles[i], fitted_t[i])

        image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
        color = bfm.generate_colors(fitted_tp[i])
        fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, color, h, w)
        io.imsave('{}/show_{:0>2d}.jpg'.format(save_folder, i), img_as_ubyte(fitted_image))

    options = '-delay 20 -loop 0 -layers optimize'  # gif. need ImageMagick.
    subprocess.call('convert {} {}/show_*.jpg {}'.format(options, save_folder, save_folder + '/3dmm.gif'), shell=True)
    subprocess.call('rm {}/show_*.jpg'.format(save_folder), shell=True)


if __name__ == '__main__':
    main()
