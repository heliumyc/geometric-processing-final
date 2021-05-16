from diffusion_curve import DiffusionCurve
from diffusion import diffuse
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
import utils


def paint(data, mask):
    PIL_data = 255 * np.ones((data.shape[1], data.shape[0], 3), dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                for k in range(3):
                    PIL_data[j, i, k] = utils.get_int_color(data[i, j, k])
    img = Image.fromarray(PIL_data, 'RGB')
    img.save('./test.jpg')


if __name__ == '__main__':
    n = 512
    curve = DiffusionCurve(n, n)
    curve.ctx_points = [(0.3, 0.3), (0.4, 0.4), (0.5, 0.5), (0.7, 0.7)]
    left_color = [(255, 0, 255, 0.0), (255, 0, 0, 1.)]
    right_color = [(0, 0, 255, 0.0), (255, 255, 0, 1.)]

    curve.color_pts_left = [(x[0] / 255, x[1] / 255, x[2] / 255, x[3]) for x in left_color]
    curve.color_pts_right = [(x[0] / 255, x[1] / 255, x[2] / 255, x[3]) for x in right_color]

    curve.rasterize_curve()
    curve.rasterize_color_source()
    curve.calc_divergence()

    import time
    s = time.time()
    pyramid = diffuse([curve], 512)
    t = time.time()
    print(t-s)
    paint(pyramid[0], np.full((n, n), True))
