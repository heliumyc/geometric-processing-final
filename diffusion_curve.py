import numpy as np
import tkinter as tk
import utils


class DiffusionCurve:
    def __init__(self):
        self.ctx_points: [(float, float)] = []
        # rgb from 0 to 1, and t, default is black at begin and end
        self.color_pts_left: [(float, float, float, float)] = [(0., 0., 0., 0.), (0., 0., 0., 1.)]
        self.color_pts_right: [(float, float, float, float)] = [(0., 0., 0., 0.), (0., 0., 0., 1.)]

        self.rasterized_pixels: np.array = None

    def rasterize_ctx_pts(self, resolution_x, resolution_y):
        return [(int(cx * resolution_x), int(cy * resolution_y)) for cx, cy in self.ctx_points]

    def rasterize_color_pts(self, resolution_x, resolution_y) -> [((int, int), (int, int))]:
        

    def rasterize(self, resolution_x, resolution_y, frequency=3000):
        self.rasterized_pixels = utils.rasterize_curve(self.ctx_points[0][0], self.ctx_points[0][1],
                                                       self.ctx_points[1][0], self.ctx_points[1][1],
                                                       self.ctx_points[2][0], self.ctx_points[2][1],
                                                       self.ctx_points[3][0], self.ctx_points[3][1],
                                                       resolution_x, resolution_y, frequency)
