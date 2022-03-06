import subprocess

import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage import io

from face3d import mesh


def project_img_color_to_mesh(photo, fitted_colors, transformed_vertices, h, w):
    img = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    fitted_colors = np.zeros_like(fitted_colors)
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
            fitted_colors[i] = img[transformed_h, transformed_w, :] / 255.
    return fitted_colors


def visualize_bfm_fit_progress(bfm, fit_progress, save_folder, h, w):
    fitted_sp, fitted_ep, fitted_tp, fitted_s, fitted_angle, fitted_t = fit_progress

    # verify fitted parameters
    for i in range(fitted_sp.shape[0]):
        fitted_vertices = bfm.generate_vertices(fitted_sp[i], fitted_ep[i])
        fitted_vertices = bfm.transform(fitted_vertices, fitted_s[i], fitted_angle[i], fitted_t[i])

        image_vertices = mesh.transform.to_image(fitted_vertices, h, w)
        colors = bfm.generate_colors(fitted_tp[i])
        fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
        io.imsave('{}/show_{:0>2d}.jpg'.format(save_folder, i), img_as_ubyte(fitted_image))

    options = '-delay 20 -loop 0 -layers optimize'  # gif. need ImageMagick.
    subprocess.call('convert {} {}/show_*.jpg {}'.format(options, save_folder, save_folder + '/3dmm.gif'), shell=True)
    subprocess.call('rm {}/show_*.jpg'.format(save_folder), shell=True)
