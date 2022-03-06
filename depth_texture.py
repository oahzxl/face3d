"""
Generate 2d texture maps representing depth
"""
import os

import numpy as np
import torch
from pytorch3d.io import save_obj
from skimage import io

from face3d import mesh
from face3d.mesh.io import load_obj_mesh, write_obj_with_colors

# ------------------------------ load mesh data
vertices, triangles = load_obj_mesh("./data/3d_mesh/lincoln.obj")

# ------------------------------ modify vertices(transformation. change position of obj)
# set h, w of rendering
h = w = 512
# scale. target size=200 for example
s = h * 0.8 / (np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
# rotate 30 degree for example
R = mesh.transform.angle2matrix([0, 30, 0])
# no translation. center of obj:[0,0]
t = [0, 0, 0]
transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)

# ------------------------------ render settings(to 2d image)
# change to image coords for rendering
image_vertices = mesh.transform.to_image(transformed_vertices, h, w)

## --- start
save_folder = 'results/mesh_map'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)


# 1. depth map
# jpg
z = image_vertices[:, 2:]
z = z - np.min(z)
z = z / np.max(z)
colors = z
depth_image = mesh.render.render_colors(image_vertices, triangles, colors, h, w, c=1)
io.imsave('{}/depth.jpg'.format(save_folder), np.squeeze(depth_image))
# obj
z = vertices[:, 2:]
z = z - np.min(z)
z = z / np.max(z)
colors = np.repeat(z, 3, axis=1)
write_obj_with_colors('{}/depth.obj'.format(save_folder), vertices, triangles, colors)
# texture
colors = z
uv_map = np.repeat(np.array(range(h * w)).reshape(h, w, 1) / (h * w - 1), 3, axis=2)
uv_map = 1 - uv_map
uv_coords = ((colors + 0.01) / 1.02) * (h * w - 1)
uv_coords_int = np.array(uv_coords, dtype=np.int32)
uv_coords_float = uv_coords - uv_coords_int
uv_coords_int_y = uv_coords_int // h
uv_coords_int_x = uv_coords_int - uv_coords_int_y * h
uv_coords_int_x = np.clip(uv_coords_int_x + uv_coords_float, a_max=(w - 1), a_min=0)
uv_coords = np.concatenate((uv_coords_int_x / float(w - 1), uv_coords_int_y / float(h - 1)), axis=1)
save_obj('{}/lincoln.obj'.format(save_folder), torch.tensor(vertices * 70 + 70), torch.tensor(triangles),
         texture_map=torch.tensor(uv_map), verts_uvs=torch.tensor(uv_coords), faces_uvs=torch.tensor(triangles))

# uv_map = 1 - uv_map
# uv_coords = uv_coords * (h - 1)
# uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))
# rendering_tc = mesh.render.render_texture(image_vertices * 150, triangles, uv_map, uv_coords, triangles, h, w)
# io.imsave('{}/face3d_tex.jpg'.format(save_folder), np.squeeze(rendering_tc))

# uv_coords = uv_coords * (h - 1)
# index = 1940
# x = uv_coords[index, 0]
# y = uv_coords[index, 1]
# print(x, y)
# print(uv_map[int(y), int(x), 0])
# print(1 - colors[index])
