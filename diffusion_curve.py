from typing import Optional

import numpy as np
import utils


class DiffusionCurve:
    def __init__(self, width, height):
        self.ctx_points: [(float, float)] = []
        # rgb from 0 to 1, and t, default is black at begin and end
        # left means the left hand side of tangent vector
        # right means the right hand side of tangent vector

        # self.color_pts_left: [(float, float, float, float)] = [(3 / 255., 157 / 255., 252 / 255., 0.),
        #                                                        (3 / 255., 157 / 255., 252 / 255., 1.)]
        # self.color_pts_right: [(float, float, float, float)] = [(252 / 255., 15 / 255., 3 / 255., 0.),
        #                                                         (252 / 255., 15 / 255., 3 / 255., 1.)]
        self.color_pts_left: [(float, float, float, float)] = [(0., 0., 0., 0.),
                                                               (0., 0., 0., 1.)]
        self.color_pts_right: [(float, float, float, float)] = [(0., 0., 0., 0.),
                                                                (0., 0., 0., 1.)]

        self.width = width
        self.height = height

        self.rasterized_pixels: Optional[np.array] = None

        self.color_source: np.array = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.color_source_mask: np.array = np.full((self.height, self.width), False)

        self.wx: np.array = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.wy: np.array = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.divergence: np.array = np.zeros((self.height, self.width, 3), dtype=np.float32)

    def rasterize_ctx_pts(self):
        return [(int(cx * self.width), int(cy * self.height)) for cx, cy in self.ctx_points]

    def __sort_color_pts(self):
        self.color_pts_left = sorted(self.color_pts_left, key=lambda x: x[-1])
        self.color_pts_right = sorted(self.color_pts_right, key=lambda x: x[-1])

    def __rasterized_tan_vec(self, t) -> (float, float):
        tangent_vec = utils.tangent_vector(self.ctx_points, t)
        tangent_vec = utils.normalize(tangent_vec[0] * self.width, tangent_vec[1] * self.height)  # len of 1 pixel
        return tangent_vec

    def rasterize_color_pts(self) -> [(int, int)]:
        self.__sort_color_pts()
        result = []
        scale = 12
        for left_p, right_p in zip(self.color_pts_left, self.color_pts_right):
            t = left_p[-1]
            tangent_vec = self.__rasterized_tan_vec(t)
            # tangent_vec = utils.tangent_vector(self.ctx_points, t)
            # tangent_vec = utils.normalize(tangent_vec[0] * self.width, tangent_vec[1] * self.height)  # len of 1 pixel
            left_vec = utils.rotate_90(tangent_vec[0], tangent_vec[1], is_right_hand_side=False)
            right_vec = utils.rotate_90(tangent_vec[0], tangent_vec[1], is_right_hand_side=True)
            cur_p = self.get_coord_at_t(t)
            cl = (cur_p[0] * self.width + left_vec[0] * scale, cur_p[1] * self.height + left_vec[1] * scale)
            cr = (cur_p[0] * self.width + right_vec[0] * scale, cur_p[1] * self.height + right_vec[1] * scale)
            result.append((int(cl[0]), int(cl[1])))
            result.append((int(cr[0]), int(cr[1])))
        return result

    def get_coord_at_t(self, t) -> (float, float):
        return utils.bezier_curve_interpolate(t, self.ctx_points[0][0], self.ctx_points[0][1],
                                              self.ctx_points[1][0], self.ctx_points[1][1],
                                              self.ctx_points[2][0], self.ctx_points[2][1],
                                              self.ctx_points[3][0], self.ctx_points[3][1])

    def rasterize_curve(self, frequency=3000):
        self.rasterized_pixels = utils.rasterize_curve(self.ctx_points[0][0], self.ctx_points[0][1],
                                                       self.ctx_points[1][0], self.ctx_points[1][1],
                                                       self.ctx_points[2][0], self.ctx_points[2][1],
                                                       self.ctx_points[3][0], self.ctx_points[3][1],
                                                       self.height, self.width, frequency)

    def __clamp(self, coord):
        coord[0] = np.clip(coord[0], 0, self.width)
        coord[1] = np.clip(coord[1], 0, self.height)
        return coord

    def rasterize_color_source(self):
        self.__sort_color_pts()
        freq = self.rasterized_pixels.shape[0]

        self.color_source.fill(0)
        self.color_source_mask.fill(False)
        self.wx.fill(0)
        self.wy.fill(0)

        color_ctx_left = np.array(self.color_pts_left)
        color_ctx_right = np.array(self.color_pts_right)
        for i, pixel in enumerate(self.rasterized_pixels):
            px, py, r, g, b, a = pixel
            t = i / freq
            scale = 3
            tangent_vec = self.__rasterized_tan_vec(t)
            left_vec = np.array(utils.rotate_90(tangent_vec[0], tangent_vec[1], is_right_hand_side=False),
                                dtype=np.float32)
            right_vec = np.array(utils.rotate_90(tangent_vec[0], tangent_vec[1], is_right_hand_side=True),
                                 dtype=np.float32)

            cur_coord = np.array([px, py], dtype=np.int32)
            # shift
            p_left = cur_coord + (scale * left_vec).astype(np.int32)
            p_right = cur_coord + (scale * right_vec).astype(np.int32)

            p_left = self.__clamp(p_left)
            p_right = self.__clamp(p_right)
            color_left, color_right = utils.interpolate_color(t, color_ctx_left, color_ctx_right)
            self.color_source[p_left[1], p_left[0]] = color_left
            self.color_source_mask[p_left[1], p_left[0]] = True
            self.color_source[p_right[1], p_right[0]] = color_right
            self.color_source_mask[p_right[1], p_right[0]] = True

            # gradient
            self.wx[cur_coord[1], cur_coord[0]] = (color_left - color_right) * (right_vec[0])
            self.wy[cur_coord[1], cur_coord[0]] = (color_left - color_right) * (right_vec[1])

    def calc_divergence(self):
        def pad(mat):
            return np.pad(mat, ((1, 1), (1, 1), (0, 0)), constant_values=(0, 0))

        padded_wx = pad(self.wx)
        padded_wy = pad(self.wy)
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
        for k in range(3):
            div_x = utils.conv2d(padded_wx[:, :, k], conv_x)
            div_y = utils.conv2d(padded_wy[:, :, k], conv_y)
            self.divergence[:, :, k] = div_x + div_y
