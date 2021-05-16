import tkinter as tk
import numpy as np
import numba as nb
import scipy.sparse as sp


def create_circle(canvas: tk.Canvas, center: (int, int), radius: int, color='') -> int:
    return canvas.create_oval(center[0] - radius, center[1] - radius,
                              center[0] + radius, center[1] + radius,
                              fill=color, width=2, outline='black')


def rasterize_curve(x0, y0, x1, y1, x2, y2, x3, y3, resolution_x, resolution_y, frequency) -> np.array:
    # (x, y, r, g, b, a)
    rasterize_pixel = np.zeros((frequency, 6), dtype=np.int32)
    for i in range(frequency):
        t = i / frequency
        x, y = bezier_curve_interpolate(t, x0, y0, x1, y1, x2, y2, x3, y3)
        px = int(x * resolution_x)
        py = int(y * resolution_y)
        rasterize_pixel[i, 0] = np.clip(px, 0, resolution_x)
        rasterize_pixel[i, 1] = np.clip(py, 0, resolution_y)
        rasterize_pixel[i, 2] = 0
        rasterize_pixel[i, 3] = 0
        rasterize_pixel[i, 4] = 0
        rasterize_pixel[i, 5] = 1
    return rasterize_pixel


# @nb.njit(fastmath=True)
def calc_sqr_dist_from_points(x: int, y: int, curve_pixel: np.array) -> (int, int):
    dist = np.empty((curve_pixel.shape[0],), dtype=np.int64)
    for i in range(len(curve_pixel)):
        dist[i] = np.square(curve_pixel[i][0] - x) + np.square(curve_pixel[i][1] - y)
    return np.min(dist), np.argmin(dist)


# @nb.njit(nb.int32(nb.int32, nb.int32, nb.int32, nb.int32))
def calc_sqr_dist_from_pt(x: int, y: int, px: int, py: int) -> int:
    return np.square(x - px) + np.square(y - py)


def flatten(l: list) -> list:
    res = []
    for x in l:
        if type(x) is list or type(x) is tuple:
            sub = [y for y in flatten(x)]
            res.extend(sub)
        else:
            res.append(x)
    return res


@nb.njit()
def interpolate_color(t, color_pts_left, color_pts_right):
    # find the color segment, could be optimized with binary search
    seg_id = -1
    for i in range(1, len(color_pts_left)):
        if color_pts_left[i][-1] >= t:
            seg_id = i - 1
            break
    if seg_id < 0:
        seg_id = len(color_pts_left) - 1
    t1 = color_pts_left[seg_id][-1]
    t2 = color_pts_right[seg_id + 1][-1]
    ratio = (t - t1) / (t2 - t1)
    color_left_1, color_left_2 = color_pts_left[seg_id], color_pts_left[seg_id + 1]
    color_right_1, color_right_2 = color_pts_right[seg_id], color_pts_right[seg_id + 1]

    color_left = color_left_1 + (color_left_2 - color_left_1) * ratio
    color_right = color_right_1 + (color_right_2 - color_right_1) * ratio
    return color_left[:-1], color_right[:-1]


@nb.njit(nb.int32(nb.float64))
def get_int_color(c: float) -> int:
    r = int(c * 255)
    if r < 0:
        return 0
    elif r > 255:
        return 255
    else:
        return r


'''
this bezier algorithm is adapted from the link below
https://gist.github.com/PM2Ring/d6a19f5062b39467ac669a4fb4715779
'''


def bezier_curve_interpolate(p: float, x0: float, y0: float, x1: float, y1: float,
                             x2: float, y2: float, x3: float, y3: float) -> (float, float):
    q = 1 - p
    t0 = q * q * q  # (1-t)^3
    t1 = q * q * p  # 3*(1-t)^2*t
    t2 = q * p * p  # 3*(1-t)*t^2
    t3 = p * p * p  # t^3
    x = t0 * x0 + 3 * t1 * x1 + 3 * t2 * x2 + t3 * x3
    y = t0 * y0 + 3 * t1 * y1 + 3 * t2 * y2 + t3 * y3
    return x, y


def tangent_vector(ctx_pts: [(float, float)], p: float) -> (float, float):
    q = 1 - p
    d0 = 3 * q * q
    d1 = 6 * q * p
    d2 = 3 * p * p
    x_1_0 = ctx_pts[1][0] - ctx_pts[0][0]
    y_1_0 = ctx_pts[1][1] - ctx_pts[0][1]
    x_2_1 = ctx_pts[2][0] - ctx_pts[1][0]
    y_2_1 = ctx_pts[2][1] - ctx_pts[1][1]
    x_3_2 = ctx_pts[3][0] - ctx_pts[2][0]
    y_3_2 = ctx_pts[3][1] - ctx_pts[2][1]
    x = d0 * x_1_0 + d1 * x_2_1 + d2 * x_3_2
    y = d0 * y_1_0 + d1 * y_2_1 + d2 * y_3_2
    return x, y


# in our case, the image coordinate is conjugate with normal 2d coordinate system
def rotate_90(x: float, y: float, is_right_hand_side) -> (float, float):
    if is_right_hand_side:
        return -y, x
    else:
        return y, -x


def normalize(x: float, y: float) -> (float, float):
    vec = np.array([x, y])
    norm = np.linalg.norm(vec)
    return x / norm, y / norm


def rgb_to_hex(r: float, g: float, b: float) -> str:
    def clamp(x):
        return max(0, min(x, 255))

    return '#%02x%02x%02x' % (clamp(int(r * 255)), clamp(int(g * 255)), clamp(int(b * 255)))


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def padding(x):
    if len(x.shape) == 3:
        return np.pad(x, ((1, 1), (1, 1), (0, 0)), constant_values=(0, 0))
    elif len(x.shape) == 2:
        return np.pad(x, ((1, 1), (1, 1)), constant_values=(0, 0))


def laplacian(m: int, n: int) -> np.array:
    size = m * n

    def flat_index(i, j):
        if i < 0 or j < 0 or i >= m or j >= n:
            return -1
        return i * n + j

    ii = []
    jj = []
    data = []

    def addto(row, col, ele):
        ii.append(row)
        jj.append(col)
        data.append(ele)

    for x in range(m):
        for y in range(n):
            cur_row = flat_index(x, y)
            around = [flat_index(x + a, y + b) for a, b in [(0, -1), (0, 1), (-1, 0), (1, 0)]]
            addto(cur_row, cur_row, -4)
            for neighbor in around:
                if neighbor >= 0:
                    addto(cur_row, neighbor, 1)
    return sp.coo_matrix((data, (ii, jj)), shape=(size, size), dtype=np.float32)


def calc_curve_divergence(wx, wy):
    padded_wx = padding(wx)
    padded_wy = padding(wy)
    # get div
    conv_y = np.array([
        [0, -0.5, 0],
        [0, 0, 0],
        [0, 0.5, 0]
    ], dtype=np.float32)
    conv_x = np.array([
        [0, 0, 0],
        [-0.5, 0, 0.5],
        [0, 0, 0]
    ], dtype=np.float32)
    divergence = np.zeros(wx.shape, dtype=wx.dtype)
    for k in range(3):
        div_x = conv2d(padded_wx[:, :, k], conv_x)
        div_y = conv2d(padded_wy[:, :, k], conv_y)
        divergence[:, :, k] = div_x + div_y
    return divergence
# def flat(x0, y0, x1, y1, tol):
#     return abs(x0*y1 - x1*y0) < tol * abs(x0 * x1 + y0 * y1)
#
# ''' Draw a cubic Bezier curve by recursive subdivision
#     The curve is subdivided until each of the 4 sections is
#     sufficiently flat, determined by the angle between them.
#     tol is the tolerance expressed as the tangent of the angle
# '''
# def bezier_points(x0:float, y0:float, x1:float, y1:float, x2:float, y2:float, x3:float, y3:float,
#                   tol:float=0.001) -> [(float, float)]:
#
#     if (flat(x1-x0, y1-y0, x2-x0, y2-y0, tol) and
#         flat(x2-x1, y2-y1, x3-x1, y3-y1, tol)):
#         return [x0, y0, x3, y3]
#
#     x01, y01 = (x0 + x1) / 2., (y0 + y1) / 2.
#     x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
#     x23, y23 = (x2 + x3) / 2., (y2 + y3) / 2.
#     xa, ya = (x01 + x12) / 2., (y01 + y12) / 2.
#     xb, yb = (x12 + x23) / 2., (y12 + y23) / 2.
#     xc, yc = (xa + xb) / 2., (ya + yb) / 2.
#
#     # Double the tolerance angle
#     tol = 2. / (1. / tol - tol)
#     return (bezier_points(x0, y0, x01, y01, xa, ya, xc, yc, tol)[:-2] +
#         bezier_points(xc, yc, xb, yb, x23, y23, x3, y3, tol))
