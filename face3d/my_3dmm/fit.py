import numpy as np
from .. import mesh

''' TODO: a clear document. 
Given: image_points, 3D Model, Camera Matrix(s, R, t2d)
Estimate: shape parameters, expression parameters

Inference: 

    projected_vertices = s*P*R(mu + shape + exp) + t2d  --> image_points
    s*P*R*shape + s*P*R(mu + exp) + t2d --> image_points

    # Define:
    X = vertices
    x_hat = projected_vertices
    x = image_points
    A = s*P*R
    b = s*P*R(mu + exp) + t2d
    ==>
    x_hat = A*shape + b  (2 x n)

    A*shape (2 x n)
    shape = reshape(shapePC * sp) (3 x n)
    shapePC*sp : (3n x 1)

    * flatten:
    x_hat_flatten = A*shape + b_flatten  (2n x 1)
    A*shape (2n x 1)
    --> A*shapePC (2n x 199)  sp: 199 x 1
    
    # Define:
    pc_2d = A* reshape(shapePC)
    pc_2d_flatten = flatten(pc_2d) (2n x 199)

    =====>
    x_hat_flatten = pc_2d_flatten * sp + b_flatten ---> x_flatten (2n x 1)

    Goals:
    (ignore flatten, pc_2d-->pc)
    min E = || x_hat - x || + lambda*sum(sp/sigma)^2
          = || pc * sp + b - x || + lambda*sum(sp/sigma)^2

    Solve:
    d(E)/d(sp) = 0
    2 * pc' * (pc * sp + b - x) + 2 * lambda * sp / (sigma' * sigma) = 0

    Get:
    (pc' * pc + lambda / (sigma'* sigma)) * sp  = pc' * (x - b)

'''


def estimate_2d_shape(img_lm, generated_lm, shapeMU, shapePC, shapeEV, s, R, t2d, lamb=3000):
    """
    Args:
        img_lm: (2, n). image points (to be fitted)
        generated_lm: (3, n). 3d model lm points
        shapeMU: (3n, 1)
        shapePC: (3n, n_sp)
        shapeEV: (n_sp, 1)
        s: scale
        R: (3, 3). rotation matrix
        t2d: (2,). 2d translation
        lamb: regulation coefficient

    Returns:
        shape_para: (n_sp, 1) shape parameters(coefficients)
    """
    img_lm = img_lm.copy()
    assert (shapeMU.shape[0] == shapePC.shape[0])
    assert (shapeMU.shape[0] == img_lm.shape[1] * 3)

    dof = shapePC.shape[1]

    n = img_lm.shape[1]
    sigma = shapeEV
    t2d = np.array(t2d)
    P = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    A = s * P.dot(R)

    # --- calc pc
    pc_3d = np.resize(shapePC.T, [dof, n, 3])  # 199 x n x 3
    pc_3d = np.reshape(pc_3d, [dof * n, 3])
    pc_2d = pc_3d.dot(A.T.copy())  # 199 x n x 2

    pc = np.reshape(pc_2d, [dof, -1]).T  # 2n x 199

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T  # 3 x n

    b = A.dot(mu_3d) + np.tile(t2d[:, np.newaxis], [1, n])  # 2 x n
    b = np.reshape(b.T, [-1, 1])  # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma ** 2)
    img_lm = np.reshape(img_lm.T, [-1, 1])
    equation_right = np.dot(pc.T, img_lm - b)
    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para


def estimate_3d_shape(img_lm, generated_lm, shapeMU, shapePC, shapeEV, s, R, t3d, lamb=3000):
    """
    Args:
        img_lm: (2, n). image points (to be fitted)
        generated_lm: (3, n). 3d model lm points
        shapeMU: (3n, 1)
        shapePC: (3n, n_sp)
        shapeEV: (n_sp, 1)
        s: scale
        R: (3, 3). rotation matrix
        t3d: (2,). 2d translation
        lamb: regulation coefficient

    Returns:
        shape_para: (n_sp, 1) shape parameters(coefficients)
    """
    img_lm = img_lm.copy()
    assert (shapeMU.shape[0] == shapePC.shape[0])
    assert (shapeMU.shape[0] == img_lm.shape[1] * 3)

    dof = shapePC.shape[1]

    n = img_lm.shape[1]
    sigma = shapeEV
    t3d = np.array(t3d)
    A = s * R

    # --- calc pc
    pc_3d = np.resize(shapePC.T, [dof, n, 3])  # 199 x n x 3
    pc_3d = np.reshape(pc_3d, [dof * n, 3])
    pc_3d = pc_3d.dot(A.T.copy())  # 199 x n x 3

    pc = np.reshape(pc_3d, [dof, -1]).T  # 3n x 199

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T  # 3 x n

    b = A.dot(mu_3d) + np.tile(t3d[:, np.newaxis], [1, n])  # 2 x n
    b = np.reshape(b.T, [-1, 1])  # 2n x 1

    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1 / sigma ** 2)
    img_lm = np.reshape(img_lm.T, [-1, 1])
    equation_right = np.dot(pc.T, img_lm - b)
    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para


# ---------------- fit 
def fit_points(lm, lm_index, model, n_sp, max_iter=4, show=False, n_d='2d'):
    """
    Args:
        lm: (n, 2) image points
        n_sp: shape param
        show: show progress
        lm_index: (n,) corresponding Model vertex indices
        model: 3DMM
        max_iter: iteration
        n_d: 2d or 3d
    Returns:
        sp: (n_sp, 1). shape parameters
        s, R, t
    """
    lm = lm.copy().T

    # -- init
    sp = np.zeros((n_sp, 1), dtype=np.float32)

    # -------------------- estimate
    X_ind_all = np.tile(lm_index[np.newaxis, :], [3, 1]) * 3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    # MU: 3n*1, average per face
    # EV: p*1 , variance within face
    # PC: 3n*p, principal component
    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]

    # init settings for show
    s = 1
    R = mesh.transform.angle2matrix([0, 0, 0])
    t = [0, 0, 0]
    lsp, ls, lR, lt = [], [], [], []

    for i in range(max_iter):
        X = shapeMU + shapePC.dot(sp)
        X = np.reshape(X, [int(len(X) / 3), 3]).T

        if show:
            lsp.append(sp), ls.append(s), lR.append(R), lt.append(t)

        # estimate pose
        if n_d == '2d':
            P = mesh.transform.estimate_affine_matrix_3d22d(X.T, lm.T)
            s, R, t = mesh.transform.P2sRt(P)
        else:
            P = mesh.transform.estimate_affine_matrix_3d23d(X.T, lm.T)
            s, R, t = mesh.transform.P2sRt(P)

        if show:
            lsp.append(sp), ls.append(s), lR.append(R), lt.append(t)

        if n_d == '2d':
            # estimate 2d shape
            sp = estimate_2d_shape(lm, X, shapeMU, shapePC, model['shapeEV'][:n_sp, :], s, R, t[:2], lamb=20)
        else:
            sp = estimate_3d_shape(lm, X, shapeMU, shapePC, model['shapeEV'][:n_sp, :], s, R, t, lamb=20)

        if show:
            lsp.append(sp), ls.append(s), lR.append(R), lt.append(t)

    result = [sp, s, R, t]
    show = [np.array(lsp), np.array(ls), np.array(lR), np.array(lt)]
    return result, show
