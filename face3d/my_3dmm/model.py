import pickle

import numpy as np

from . import fit
from .. import mesh


class MyMorphableModel(object):
    """docstring for  MorphableModel
    model:
        nver: number of vertices.
        ntri: number of triangles.
        *: must have.
        ~: can generate ones array for place holder.
            'shapeMU': average per component(face), [3*nver, 1]. *
            'shapePC': principal component, [3*nver, n_shape_para]. *
            'shapeEV': variance within each component(face), [n_shape_para, 1]. ~
            'expMU': [3*nver, 1]. ~ 
            'expPC': [3*nver, n_exp_para]. ~
            'expEV': [n_exp_para, 1]. ~
            'texMU': [3*nver, 1]. ~
            'texPC': [3*nver, n_tex_para]. ~
            'texEV': [n_tex_para, 1]. ~
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
            'kpt_ind': [68,] (start from 1). ~
    """

    def __init__(self, model_path):
        super(MyMorphableModel, self).__init__()
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # fixed attributes
        self.nver = self.model['shapePC'].shape[0] / 3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.kpt_ind = self.model['kpt_ind']
        self.triangles = self.model['tri']

    # shape: represented with mesh(vertices & triangles(fixed))
    def get_shape_para(self, dtype='random'):
        if dtype == 'zero':
            sp = np.zeros((self.n_shape_para, 1))
        elif dtype == 'random':
            sp = np.random.rand(self.n_shape_para, 1) * 0.1
        else:
            raise TypeError
        return sp

    def generate_vertices(self, shape_para):
        """
        Args:
            shape_para: (n_shape_para, 1)
        Returns:
            vertices: (nver, 3)
        """
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(shape_para)
        vertices = np.reshape(vertices, [-1, 3])

        return vertices

    # transformation
    @staticmethod
    def rotate(vertices, angles):
        """ rotate face
        Args:
            vertices: [nver, 3]
            angles: [3] x, y, z rotation angle(degree)
        Returns:
            vertices: rotated vertices
        """
        return mesh.transform.rotate(vertices, angles)

    @staticmethod
    def transform(vertices, s, angles, t3d):
        R = mesh.transform.angle2matrix(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)

    @staticmethod
    def transform_3ddfa(vertices, s, angles, t3d):  # only used for processing 300W_LP data
        R = mesh.transform.angle2matrix_3ddfa(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)

    # fit
    def fit(self, x, X_ind, max_iter=4, isShow=False, n_d='2d'):
        """ fit 3dmm & pose parameters
        Args:
            x: (n, 2) image points
            X_ind: (n,) corresponding Model vertex indices
            max_iter: iteration
            isShow: whether to reserve middle results for show
            n_d: 2d or 3d
        Returns:
            fitted_sp: (n_sp, 1). shape parameters
            fitted_ep: (n_ep, 1). exp parameters
            s, angles, t
        """
        assert isShow is True
        assert n_d in ('2d', '3d')

        result, show = fit.fit_points(
            x, X_ind, self.model, n_sp=self.n_shape_para, max_iter=max_iter, show=True, n_d=n_d)

        # turn rotate matrix to angle
        angles = np.zeros((show[-2].shape[0], 3))
        for i in range(show[-2].shape[0]):
            angles[i] = mesh.transform.matrix2angle(show[-2][i])
        show[-2] = angles

        angles = mesh.transform.matrix2angle(result[-2])
        result[-2] = angles

        return result, show

    # utils
    def create_face_bfm(self, mode='random'):
        """
        generate face mesh: vertices(represent shape) & colors(represent texture)
        """
        assert isinstance(mode, str)
        assert mode in ('zero', 'random')

        sp = self.get_shape_para(mode)
        vertices = self.generate_vertices(sp)

        return vertices, sp

    def mesh_to_image(self, vertices, colors, h, w, s=1, angle=(0, 0, 0), t=(0, 0, 0)):
        vertices = self.transform(vertices, s, angle, t)
        vertices = mesh.transform.to_image(vertices, h, w)
        image = mesh.render.render_colors(vertices, self.triangles, colors, h, w)
        return image
