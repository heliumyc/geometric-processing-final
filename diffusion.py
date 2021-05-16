import numpy as np
import numba as nb
import scipy.sparse as sp
import scipy.optimize as so
from diffusion_curve import DiffusionCurve
from PIL import Image
import utils


def downsample_gradient(x: np.array) -> np.array:
    n = x.shape[1]
    m = n // 2
    y = np.zeros((m, m, 3), dtype=x.dtype)
    for i in range(m):
        for j in range(m):
            count = 4
            y[i, j] = (x[2 * i, 2 * j] + x[2 * i + 1, 2 * j] + x[2 * i, 2 * j + 1] + x[2 * i + 1, 2 * j + 1]) / count
    return y


def downsample_pixels(x: np.array, mask: np.array) -> (np.array, np.array):
    n = x.shape[1]
    m = n // 2
    y = np.zeros((m, m, 3), dtype=x.dtype)
    new_mask = np.full((m, m), False)
    for i in range(m):
        for j in range(m):
            sub_idx = np.ix_([2 * i, 2 * i + 1], [2 * j, 2 * j + 1])
            count = np.sum(1 * mask[sub_idx])
            new_mask[i, j] = count > 0
            if count > 0:
                y[i, j] = (x[2 * i, 2 * j] + x[2 * i + 1, 2 * j] + x[2 * i, 2 * j + 1] + x[
                    2 * i + 1, 2 * j + 1]) / count
    return y, new_mask


def upsample(x: np.array) -> np.array:
    m = x.shape[0]
    n = m * 2
    y = np.zeros((n, n, 3), dtype=x.dtype)
    for i in range(m):
        for j in range(m):
            y[2 * i, 2 * j] = y[2 * i, 2 * j + 1] = y[2 * i + 1, 2 * j] = y[2 * i + 1, 2 * j + 1] = x[i, j]

    return y


@nb.njit(nb.boolean(nb.int32, nb.int32, nb.int32))
def is_valid_idx(i, j, n) -> bool:
    return 0 <= i < n and 0 <= j < n


@nb.guvectorize([(nb.float32[:, :, :], nb.float32[:, :, :], nb.float32[:, :, :])],
                '(n,n,m),(n,n,m)->(n,n,m)')
def jacobi(x0, divw, x):
    n = x0.shape[0]
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            count = 0
            for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor_idx = (i + offset[0], j + offset[1])
                if is_valid_idx(neighbor_idx[0], neighbor_idx[1], n):
                    count += 1
                    x[i, j] += x0[neighbor_idx]

            x[i, j] -= divw[i, j]
            x[i, j] /= count


def jacobi_relaxation(x0: np.array, divw: np.array, constraint, mask, max_iter=5):
    n = x0.shape[0]
    x = np.zeros(x0.shape, x0.dtype)
    for k in range(max_iter):
        jacobi(x0, divw, x)
        x0, x = x, x0

        # for i in range(n):
        #     for j in range(n):
        #         count = 0
        #         for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        #             neighbor_idx = (i + offset[0], j + offset[1])
        #             if is_valid_idx(neighbor_idx[0], neighbor_idx[1], n):
        #                 count += 1
        #                 x[i, j] += x0[neighbor_idx]
        #
        #         x[i, j] -= divw[i, j]
        #         x[i, j] /= count
        # x0, x = x, x0

        x.fill(0)
        x0[mask] = constraint[mask]
    return x0


def solve_exact(pixel, mask, div):
    n = pixel.shape[0]
    laplacian = utils.laplacian(n, n)
    flat_div = div.reshape(-1, 3)
    min_constraint = np.zeros((n, n, 3), dtype=np.float32)
    max_constraint = np.full((n, n, 3), np.inf, dtype=np.float32)

    # build constraint
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                min_constraint[i, j] = pixel[i, j] - 1e-5
                max_constraint[i, j] = pixel[i, j] + 1e-5

    min_constraint = min_constraint.reshape(-1, 3)
    max_constraint = max_constraint.reshape(-1, 3)

    x = np.zeros((n, n, 3), dtype=np.float32)

    for i in range(3):
        res = so.lsq_linear(laplacian, flat_div[:, i], method='trf',
                            bounds=(min_constraint[:, i], max_constraint[:, i]),
                            max_iter=20, tol=1e-6, verbose=0)
        x[:, :, i] = res.x.reshape(n, n)

    return x


def diffuse(curves: [DiffusionCurve], canvas_size: int) -> np.array:
    # build a global color source
    global_color_source = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    global_color_mask = np.full((canvas_size, canvas_size), False)
    global_gradx = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    global_grady = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    global_div = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    for c in curves:
        global_color_source += c.color_source
        global_color_mask = np.logical_or(global_color_mask, c.color_source_mask)
        global_gradx += c.wx
        global_grady += c.wy
        global_div += utils.calc_curve_divergence(c.wx, c.wy)

    # build downsample coarse levels
    pyramid_gradx = [global_gradx]
    pyramid_grady = [global_grady]
    pyramid_div = [global_div]
    pyramid_color_source = [global_color_source]
    pyramid_color_mask = [global_color_mask]

    least_solve_size = 16
    pyramid_sizes = [least_solve_size * (2 ** l) for l in
                     reversed(range(int(np.log2(canvas_size / least_solve_size)) + 1))]
    levels = len(pyramid_sizes)

    for l in range(1, levels):
        gradx = downsample_gradient(pyramid_gradx[l - 1])
        pyramid_gradx.append(gradx)
        grady = downsample_gradient(pyramid_grady[l - 1])
        pyramid_grady.append(grady)
        div = utils.calc_curve_divergence(gradx, grady)
        pyramid_div.append(div)
        pixel, mask = downsample_pixels(pyramid_color_source[l - 1], pyramid_color_mask[l - 1])
        pyramid_color_source.append(pixel)
        pyramid_color_mask.append(mask)

    # solve exact at least size
    pyramid_pixels = [solve_exact(pyramid_color_source[-1], pyramid_color_mask[-1], pyramid_div[-1])]

    # upsample and relaxation
    x0 = pyramid_pixels[0]
    for l in reversed(range(0, levels - 1)):
        x = jacobi_relaxation(upsample(x0), pyramid_div[l], pyramid_color_source[l], pyramid_color_mask[l])
        x0 = x
        pyramid_pixels.append(x0)

    pyramid_pixels.reverse()
    return pyramid_pixels
