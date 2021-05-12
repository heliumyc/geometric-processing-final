import tkinter as tk
import numpy as np
import numba as nb


def create_circle(canvas: tk.Canvas, center: (int, int), radius: int) -> int:
    return canvas.create_oval(center[0] - radius, center[1] - radius,
                              center[0] + radius, center[1] + radius,
                              fill='black', width=10)


def rasterize_curve(x0, y0, x1, y1, x2, y2, x3, y3, resolution_x, resolution_y, frequency) -> np.array:
    # (x, y, r, g, b, a)
    rasterize_pixel = np.zeros((frequency, 6), dtype=np.int32)
    for i in range(frequency):
        t = i / frequency
        x, y = bezier_curve_interpolate(t, x0, y0, x1, y1, x2, y2, x3, y3)
        px = int(x * resolution_x)
        py = int(y * resolution_y)
        rasterize_pixel[i, 0] = px
        rasterize_pixel[i, 1] = py
        rasterize_pixel[i, 2] = 0
        rasterize_pixel[i, 3] = 0
        rasterize_pixel[i, 4] = 0
        rasterize_pixel[i, 5] = 255
    return rasterize_pixel


@nb.njit(fastmath=True)
def calc_sqr_dist_from_curve(x: int, y: int, curve_pixel: np.array) -> int:
    dist = np.empty((curve_pixel.shape[0],), dtype=np.int64)
    for i in range(len(curve_pixel)):
        dist[i] = np.square(curve_pixel[i][0] - x) + np.square(curve_pixel[i][1] - y)
    return np.min(dist)


@nb.njit(nb.int32(nb.int32, nb.int32, nb.int32, nb.int32))
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
