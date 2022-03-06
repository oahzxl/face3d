"""
Build my 3DMM with 3DCaricShop data
"""
import glob
import os
import pickle

import numpy as np
import tqdm
from sklearn.decomposition import PCA


def read_obj(file):
    m = []
    while True:
        line = file.readline()
        # end of file
        if not line:
            break
        # read vertices
        items = line.split(" ")
        if items[0] == "v":
            # remove some symbols
            if '\n' in items[-1]:
                items[-1] = items[-1].replace('\n', '')
            if len(items) != 4:
                for i in range(len(items) - 1, -1, -1):
                    if len(items[i]) < 1:
                        items.pop(i)

            m.append((float(items[1]), float(items[2]), float(items[3])))
        elif items[0] == "vt" or items[0] == 'f':
            break
    return m


def read_obj_list(tmesh_list):
    tmesh = []

    for t in tqdm.tqdm(tmesh_list):
        with open(t) as f:
            m = read_obj(f)
        if len(m) == 11551:
            tmesh.append(m)

    tmesh = np.array(tmesh)
    return tmesh


def read_faces(tmesh_list):
    mesh_count = 0

    for t in tmesh_list:
        with open(t) as f:
            face = []
            while True:
                line = f.readline()
                # end of file
                if not line:
                    break
                # read vertices
                if line[0] == "v":
                    mesh_count += 1
                elif line[0] == "f":
                    items = line.split(' ')
                    # remove some symbols
                    if '\n' in items[-1]:
                        items[-1] = items[-1].replace('\n', '')
                    try:
                        face.append((int(items[1]), int(items[2]), int(items[3])))
                    except ValueError:
                        break
            if mesh_count == 11551:
                return np.array(face)


def write_lm(path):
    x = y = z = 0

    label_points = open(r'./data/3DCaricShop/labelled_tMesh_picked_points.pp')
    lm_list = []
    for line in label_points:
        if "point" not in line:
            continue
        line = line.split(' ')
        for li in line:
            if "x=" in li:
                x = li.split('"')[1]
            elif "y=" in li:
                y = li.split('"')[1]
            elif "z=" in li:
                z = li.split('"')[1]
        lm_list.append((float(x), float(y), float(z)))
    lm_list = np.array(lm_list)

    points = []
    with open(path) as f:
        while True:
            line = f.readline()
            # end of file
            if not line:
                break
            # read vertices
            items = line.split(' ')
            if items[0] == "v":
                if items[-3][0] == '1' or True:
                    # remove some symbols
                    if '\n' in items[-1]:
                        items[-1] = items[-1].replace('\n', '')
                    points.append((float(items[1]), float(items[2]), float(items[3])))
    lm_arg_list = []
    points = np.array(points)
    for i in range(len(lm_list)):
        lm = lm_list[i].reshape(1, -1)
        dist = (lm - points) ** 2
        dist = np.sum(dist, axis=-1)
        lm_arg = np.argmin(dist)
        lm_arg_list.append(lm_arg)

    import matplotlib.pyplot as plt
    x = points[lm_arg_list, 0]
    y = points[lm_arg_list, 1]
    plt.scatter(x, y)
    plt.show()

    num_str = ""
    for lm in lm_arg_list:
        num_str = num_str + str(int(lm)) + '\n'
    with open(r'./data/3DCaricShop/labelled_tMesh_lm.txt', 'w+') as f:
        f.write(num_str)


def read_lm(path):
    if not os.path.exists(r'./data/3DCaricShop/labelled_tMesh_lm.txt'):
        write_lm(path)
    with open(r'./data/3DCaricShop/labelled_tMesh_lm.txt', 'r') as f:
        lm = f.readlines()
    lm = np.array([int(i[:-1]) for i in lm])
    return lm


def build_3dmm():
    print("Load meshes...")
    tmesh_path = r'./data/3DCaricShop/tMesh'
    tmesh_list = glob.glob(os.path.join(tmesh_path, r'*/*'))
    # tmesh_list = tmesh_list[:20]
    lm_path = r'./data/3DCaricShop/labelled_tMesh.obj'

    bfm = {'tri': None, 'kpt_ind': None, 'shapePC': None, 'shapeMU': None, 'shapeEV': None}
    tmesh = read_obj_list(tmesh_list)
    tmesh = np.reshape(tmesh, (tmesh.shape[0], -1)).T

    print("Perform PCA...")
    tmesh_mean = tmesh.mean(axis=1, keepdims=True)
    tmesh = tmesh - tmesh_mean
    pca = PCA(n_components=199)
    pca.fit(tmesh)
    fitted_tmesh = pca.transform(tmesh)

    print("Save model...")
    # tri: [ntri, 3] (start from 1, should sub 1 in python and c++)
    bfm['tri'] = read_faces(tmesh_list) - 1
    # shapeMU: average per component(face), [3*nver, 1]
    bfm['shapeMU'] = tmesh_mean
    # shapePC: principal component, [3*nver, n_shape_para]
    bfm['shapePC'] = fitted_tmesh
    # shapeEV: variance within each component(face), [n_shape_para, 1]
    bfm['shapeEV'] = np.var(fitted_tmesh, axis=0, keepdims=True).T
    # lm index: [68]
    bfm['kpt_ind'] = read_lm(lm_path)

    with open('./data/my_bfm.pkl', 'wb') as f:
        pickle.dump(bfm, f, pickle.HIGHEST_PROTOCOL)

    # from pytorch3d.io import save_obj, load_obj
    # import torch
    # save_obj("./results/2.obj", torch.tensor(bfm['shapeMU'].reshape(-1, 3)), torch.tensor(bfm['tri']))

    # with open('./data/my_bfm.pkl', 'rb') as f:
    #     bfm = pickle.load(f)

    return 0


if __name__ == '__main__':
    build_3dmm()
