"""
Use 3DMM to fit 2d pictures with My own build model
"""

import os

import numpy as np
import torch
import tqdm
from pytorch3d.io import save_obj, load_obj

from face3d.my_3dmm import MyMorphableModel
from utils.preprocess import get_face_landmarks


IMG_PATH = ['./data/2d_face/photo1.png', './data/2d_face/photo2.png', './data/2d_face/photo3.png',
            './data/2d_face/photo4.png', './data/2d_face/photo5.jpg', './data/2d_face/photo6.jpg',
            './data/2d_face/photo7.jpg', './data/2d_face/photo8.jpg']
MESH_PATH = ['./data/3d_mesh/mesh1.obj', './data/3d_mesh/mesh2.obj']


def fit_2d_face(photo_path):
    """
    Get a 3d face model from a 2d photo by 3DMM.
    """

    max_iter = 10
    create_mode = 'zero'
    save_name = photo_path.split('/')[-1].split('.')[0] + '.obj'
    photo_save_path = './results/' + save_name.split('.')[0] + '_lm.jpg'
    model_path = './data/my_bfm.pkl'
    save_path = './results'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 1. load model
    bfm = MyMorphableModel(model_path)
    # 2. generate face mesh: vertices(represent shape) & colors(represent texture)
    vertices, sp = bfm.create_face_bfm(mode=create_mode)
    # 3. transform vertices to proper position
    template_vertices = bfm.transform(vertices, s=1, angles=[0, 0, 0], t3d=[0, 0, 0])
    save_obj('{}/template.obj'.format(save_path), torch.tensor(template_vertices), torch.tensor(bfm.triangles))

    # 4. get landmark and index
    photo, photo_with_lm, landmarks, landmark_colors = get_face_landmarks(photo_path, photo_save_path)
    landmarks = np.array(landmarks)
    landmark_index = bfm.kpt_ind  # index of key points in 3DMM. fixed.

    # 5. fit landmarks
    fit_result, fit_progress = bfm.fit(landmarks, landmark_index, max_iter=max_iter, isShow=True)
    fitted_sp, fitted_s, fitted_angle, fitted_t = fit_result
    # 6. use fitted param to generate new mesh
    fitted_vertices = bfm.generate_vertices(fitted_sp)
    fitted_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angle, fitted_t)
    # 7. save fitted mesh
    save_obj(os.path.join(save_path, save_name), torch.tensor(fitted_vertices), torch.tensor(bfm.triangles))

    # 8. plot fitted landmarks
    # f_lm = fitted_vertices[landmark_index, :]
    # import matplotlib.pyplot as plt
    # plt.plot(landmarks[:, 0], landmarks[:, 1])
    # plt.show()
    # plt.plot(f_lm[:, 0], f_lm[:, 1])
    # plt.show()

    return fit_result, fitted_vertices


def fit_3d_mesh(mesh_path):
    """
    Get a 3d face model from a 3d model by 3DMM.
    """

    max_iter = 10
    create_mode = 'zero'
    save_name = mesh_path.split('/')[-1]
    bfm_path = './data/my_bfm.pkl'
    save_path = './results'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 1. load model
    bfm = MyMorphableModel(bfm_path)
    # 2. generate face mesh: vertices(represent shape) & colors(represent texture)
    vertices, sp = bfm.create_face_bfm(mode=create_mode)
    # 3. transform vertices to proper position
    template_vertices = bfm.transform(vertices, s=1, angles=[0, 0, 0], t3d=[0, 0, 0])
    save_obj('{}/template.obj'.format(save_path), torch.tensor(template_vertices), torch.tensor(bfm.triangles))

    # 4. get landmark and index
    target_mesh = np.array(load_obj(mesh_path, load_textures=False)[0])
    landmark_index = bfm.kpt_ind  # index of landmark points
    # landmark_index = np.array(list(range(target_mesh.shape[0])))  # all points
    landmarks = target_mesh[landmark_index, :]

    # 5. fit landmarks
    fit_result, fit_progress = bfm.fit(landmarks, landmark_index, n_d='3d', max_iter=max_iter, isShow=True)
    fitted_sp, fitted_s, fitted_angle, fitted_t = fit_result
    # 6. use fitted param to generate new mesh
    fitted_vertices = bfm.generate_vertices(fitted_sp)
    fitted_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angle, fitted_t)

    # 7. save fitted mesh
    save_obj(os.path.join(save_path, save_name), torch.tensor(fitted_vertices), torch.tensor(bfm.triangles))
    # 8. plot fitted landmarks
    # f_lm = fitted_vertices[landmark_index, :]
    # import matplotlib.pyplot as plt
    # plt.plot(landmarks[:, 0], landmarks[:, 1])
    # plt.show()
    # plt.plot(f_lm[:, 0], f_lm[:, 1])
    # plt.show()

    return fit_result, fitted_vertices


def param_interpolation(p1, p2, step=10):
    bfm_path = './data/my_bfm.pkl'
    save_path = './results'
    bfm = MyMorphableModel(bfm_path)
    p1 = p1[0]
    p2 = p2[0]
    for i in range(step + 1):
        ratio1 = 1 - float(i) / step
        ratio2 = 1 - ratio1
        p = [ratio1 * np.array(p1[j]) + ratio2 * np.array(p2[j]) for j in range(len(p1))]
        vertices = bfm.generate_vertices(p[0])
        vertices = bfm.transform(vertices, p[1], p[2], p[3])
        save_obj(os.path.join(save_path, "interpolation_%d.obj" % i),
                 torch.tensor(vertices), torch.tensor(bfm.triangles))


if __name__ == '__main__':
    params = []
    for path in tqdm.tqdm(IMG_PATH):
        fit_2d_face(path)
    for path in tqdm.tqdm(MESH_PATH):
        param = fit_3d_mesh(path)
        params.append(param)
    param_interpolation(params[0], params[1])
