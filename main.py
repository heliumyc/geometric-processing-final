import tkinter as tk
import tkinter.colorchooser
from enum import Enum, auto

from diffusion_curve import DiffusionCurve
import utils
import PIL
from PIL import Image, ImageDraw, ImageTk
import numpy
import numpy as np
import numba as nb
from typing import Optional
from diffusion import diffuse


class Mode(Enum):
    CreateMode = auto()
    EditShapeMode = auto()
    CreateColorMode = auto()
    EditColorMode = auto()
    CreateBlurMode = auto()
    EditBlurMode = auto()
    DeleteMode = auto()


class Application(tk.Frame):

    def __init__(self):
        super().__init__()

        self.width = 512  # canvas width
        self.height = 512  # canvas height
        self.center = self.height // 2
        self.white = (255, 255, 255)  # canvas back
        self.button_edit_shape_text = 'Edit shape'
        self.button_add_text = 'Create Shape'
        self.button_add_done_text = 'Finish Creation'
        self.button_create_color_text = 'Add color'
        self.button_edit_color_text = 'Edit color'
        self.button_create_blur_text = 'Add blur'
        self.button_edit_blur_text = 'Edit blur'
        self.button_delete_text = 'Delete a Curve'
        self.button_render_text = 'Render!'
        self.circle_radius = 5

        self.curves: [DiffusionCurve] = []
        self.current_curve: Optional[DiffusionCurve] = None
        self.current_handler_idx: int = -1
        self.handler_buf: [(int, int, int)] = []
        self.mode = Mode.EditShapeMode
        self.curve_selection_threshold = 40 ** 2  # 5px of dist from selection curve
        self.pts_selection_threshold = 10 ** 2  # 5px of dist from selection point

        self.canvas_pixels = 255 * np.ones((self.height, self.width, 4), dtype=np.uint8)
        self.init_widget()

    def init_widget(self):
        self.pack()
        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg='white')
        self.canvas.bind("<Button-1>", self.on_click_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.on_release_canvas)
        self.canvas.bind("<B1-Motion>", self.on_move_canvas)
        self.canvas.pack(side=tk.RIGHT)

        control_frame = tk.Frame(self, borderwidth=1, width=150)
        control_frame.pack(side=tk.LEFT)
        self.button_select = tk.Button(control_frame, width=10, text=self.button_edit_shape_text,
                                       command=self.on_click_edit_shape)
        self.button_select.pack(side=tk.TOP)
        self.button_create = tk.Button(control_frame, width=10, text=self.button_add_text,
                                       command=self.on_click_create_shape)
        self.button_create.pack(side=tk.TOP)
        self.button_add_color = tk.Button(control_frame, width=10, text=self.button_create_color_text,
                                          command=self.on_click_add_color)
        self.button_add_color.pack(side=tk.TOP)
        self.button_edit_color = tk.Button(control_frame, width=10, text=self.button_edit_color_text,
                                           command=self.on_click_edit_color)
        self.button_edit_color.pack(side=tk.TOP)
        self.button_delete_curve = tk.Button(control_frame, width=10, text=self.button_delete_text,
                                             command=self.on_click_delete_curve)
        self.button_delete_curve.pack(side=tk.TOP)
        self.button_render = tk.Button(control_frame, width=10, text=self.button_render_text,
                                       command=self.on_click_render_diffusion)
        self.button_render.pack(side=tk.TOP)

        self.canvas_image = Image.fromarray(self.canvas_pixels, mode='RGB')
        self.tkImg = ImageTk.PhotoImage(self.canvas_image)
        self.sprite = self.canvas.create_image(self.width / 2, self.height / 2, image=self.tkImg)

        # test
        self.handler_buf = [(27, 443, 2), (191, 181, 3), (415, 352, 4), (450, 99, 5)]
        # self.handler_buf = [(100, 100, 2), (200, 200, 3), (300, 300, 4), (400, 400, 5)]
        curve = DiffusionCurve(self.width, self.height)

        curve.ctx_points = [(x / self.width, y / self.height) for x, y, _ in self.handler_buf]
        left_color = [(255, 0, 255, 0.0), (255, 0, 0, 1.)]
        right_color = [(0, 0, 255, 0.0), (255, 255, 0, 1.)]

        curve.color_pts_left = [(x[0] / 255, x[1] / 255, x[2] / 255, x[3]) for x in left_color]
        curve.color_pts_right = [(x[0] / 255, x[1] / 255, x[2] / 255, x[3]) for x in right_color]

        curve.rasterize_curve()
        self.curves.append(curve)
        self.cleanup_buf()
        curve.rasterize_color_source()
        curve.calc_divergence()
        self.paint_curves()

    def on_click_edit_shape(self):
        self.reset_style()
        self.cleanup_buf()
        self.paint_curves()
        self.mode = Mode.EditShapeMode

    def on_click_create_shape(self):
        self.cleanup_buf()
        self.paint_curves()
        if self.mode is not Mode.CreateMode:
            self.button_create['fg'] = '#43454a'
            self.button_create['font'] = '-weight bold'
            self.button_create['text'] = self.button_add_done_text
            self.mode = Mode.CreateMode
        else:
            self.reset_style()
            self.mode = Mode.EditShapeMode

    def on_click_add_color(self):
        self.reset_style()
        self.cleanup_buf()
        for c in self.curves:
            c.rasterize_color_source()
        self.paint_color_source()
        self.mode = Mode.CreateColorMode

    def on_click_edit_color(self):
        self.reset_style()
        self.cleanup_buf()
        for c in self.curves:
            c.rasterize_color_source()
        self.paint_color_source()
        self.mode = Mode.EditColorMode

    def on_click_canvas(self, event):
        # print('click canvas')
        if self.mode is Mode.CreateMode:
            circle_idx = utils.create_circle(self.canvas, (event.x, event.y), self.circle_radius)
            self.handler_buf.append((event.x, event.y, circle_idx))
            if len(self.handler_buf) == 4:
                curve = DiffusionCurve(self.width, self.height)
                curve.ctx_points = [(x / self.width, y / self.height) for x, y, _ in self.handler_buf]
                curve.rasterize_curve()
                self.curves.append(curve)
                self.cleanup_buf()
                self.paint_curves()
        elif self.mode is Mode.DeleteMode:
            self.handle_delete(event)
        elif self.mode is Mode.EditShapeMode:
            self.select_on_canvas(event)
        elif self.mode is Mode.CreateColorMode:
            self.handle_create_color_click(event)
        elif self.mode is Mode.EditColorMode:
            self.handle_edit_color_click(event)
        elif self.mode is Mode.CreateBlurMode:
            self.handle_create_blur_click(event)
        elif self.mode is Mode.EditBlurMode:
            self.handle_edit_blur_click(event)

    def on_click_delete_curve(self):
        self.reset_style()
        self.cleanup_buf()
        self.paint_curves()
        self.mode = Mode.DeleteMode

    def handle_delete(self, event):
        has_sth_selected = self.select_curve(event)
        # draw ctx points
        if has_sth_selected and self.current_curve:
            self.curves.remove(self.current_curve)
        self.paint_curves()

    def draw_color_ctx(self):
        def get_color_fn(i, curve):
            if i % 2 == 0:
                return curve.color_pts_left[i // 2]
            else:
                return curve.color_pts_right[i // 2]

        color_handlers = self.current_curve.rasterize_color_pts()
        colors = [get_color_fn(i, self.current_curve)[:-1] for i, c in enumerate(color_handlers)]
        self.draw_circle_buf(color_handlers, colors)

    def handle_create_color_click(self, event):
        self.cleanup_buf()
        is_select_near_curve = self.select_curve(event)
        if not is_select_near_curve:
            self.cleanup_buf()
            return
        _, discrete_id = utils.calc_sqr_dist_from_points(event.x, event.y, self.current_curve.rasterized_pixels)
        freq = self.current_curve.rasterized_pixels.shape[0]
        t = discrete_id / freq

        self.current_curve.color_pts_left.append((0., 0., 0., t))
        self.current_curve.color_pts_right.append((0., 0., 0., t))

        self.draw_color_ctx()
        self.current_curve.rasterize_color_source()
        self.paint_color_source()

    def handle_edit_color_click(self, event):
        self.cleanup_buf()
        has_sth_selected = self.select_curve(event)
        if has_sth_selected:
            self.draw_color_ctx()
        has_sth_selected = self.select_handler(event) or has_sth_selected
        if has_sth_selected and self.current_handler_idx >= 0:
            color, _ = tk.colorchooser.askcolor(title="Choose color")
            if not color:
                return
            color = tuple([x / 255.0 for x in color])
            if self.current_handler_idx % 2 == 0:
                _, _, _, t = self.current_curve.color_pts_left[self.current_handler_idx // 2]
                self.current_curve.color_pts_left[self.current_handler_idx // 2] = (color[0], color[1], color[2], t)
            else:
                _, _, _, t = self.current_curve.color_pts_right[self.current_handler_idx // 2]
                self.current_curve.color_pts_right[self.current_handler_idx // 2] = (color[0], color[1], color[2], t)
        if self.current_curve:
            self.draw_color_ctx()
            self.current_curve.rasterize_color_source()
            self.paint_color_source()

    def on_click_render_diffusion(self):
        print('start render')
        for c in self.curves:
            c.rasterize_curve()
            c.rasterize_color_source()
            c.calc_divergence()
        pyramid_pixels = diffuse(self.curves, self.width)
        pixels = pyramid_pixels[0]
        self.clear_pixels()
        self.canvas_pixels[:, :, 0] = np.clip(255 * pixels[:, :, 0], 0, 255).astype(np.uint8)
        self.canvas_pixels[:, :, 1] = np.clip(255 * pixels[:, :, 1], 0, 255).astype(np.uint8)
        self.canvas_pixels[:, :, 2] = np.clip(255 * pixels[:, :, 2], 0, 255).astype(np.uint8)
        self.refresh_canvas_image()
        print('done render')

    def handle_create_blur_click(self, event):
        pass

    def handle_edit_blur_click(self, event):
        pass

    def on_move_canvas(self, event):
        # print('move')
        if not self.mode == Mode.EditShapeMode:
            return
        if self.current_handler_idx >= 0 and self.handler_buf:
            cid = self.handler_buf[self.current_handler_idx][-1]
            # self.canvas.moveto(cid, event.x - 6 * self.circle_radius, event.y - 6 * self.circle_radius)
            self.canvas.moveto(cid, event.x, event.y)
            self.canvas.itemconfig(cid, outline='red')

            # update curve
            self.current_curve.ctx_points[self.current_handler_idx] = (event.x / self.width, event.y / self.height)
            self.current_curve.rasterize_curve()
            # self.current_curve.rasterize_color_source()
            self.paint_curves()

    def on_release_canvas(self, event):
        # print('release')
        if self.mode == Mode.EditShapeMode:
            if self.current_curve:
                self.draw_circle_buf(self.current_curve.rasterize_ctx_pts())
            self.current_handler_idx = -1

    def select_curve(self, event) -> bool:
        has_sth_selected = False
        # select curve
        if self.curves:
            dist = [utils.calc_sqr_dist_from_points(event.x, event.y, c.rasterized_pixels)[0] for c in self.curves]
            min_dist = np.min(dist)
            min_curve_id = np.argmin(dist)
            if min_dist < self.curve_selection_threshold:
                self.current_curve = self.curves[min_curve_id]
                self.current_handler_idx = -1
                has_sth_selected = True
        return has_sth_selected

    def select_handler(self, event) -> bool:
        has_sth_selected = False
        # select ctx points
        if self.handler_buf:
            dist = [utils.calc_sqr_dist_from_pt(event.x, event.y, p[0], p[1]) for p in self.handler_buf]
            min_dist = np.min(dist)
            if min_dist < self.pts_selection_threshold:
                self.current_handler_idx = np.argmin(dist)
                has_sth_selected = True
        return has_sth_selected

    def select_on_canvas(self, event):
        has_sth_selected = self.select_curve(event)
        # draw ctx points
        if self.current_curve:
            ctx_pts = self.current_curve.rasterize_ctx_pts()
            self.draw_circle_buf(ctx_pts)
        has_sth_selected = self.select_handler(event) or has_sth_selected
        # nothing is selected (too far from ctx pts and curves)
        if not has_sth_selected:
            self.cleanup_buf()
            self.canvas.update()

    def reset_style(self):
        self.button_create['fg'] = 'black'
        self.button_create['font'] = '-weight normal'
        self.button_create['text'] = self.button_add_text
        self.mode = Mode.EditShapeMode

    def draw_circle_buf(self, pts: [(int, int)], colors: [(float, float, float)] = None):
        _ = [self.canvas.delete(idx) for _, _, idx in self.handler_buf]
        # self.canvas.update()
        if colors:
            circle_idx = [utils.create_circle(self.canvas, (c[0], c[1]), self.circle_radius,
                                              color=utils.rgb_to_hex(colors[i][0], colors[i][1], colors[i][2])) for i, c
                          in enumerate(pts)]

        else:
            circle_idx = [utils.create_circle(self.canvas, (cx, cy), self.circle_radius) for cx, cy in pts]

        self.handler_buf = [(p[0], p[1], circle_idx[i]) for i, p in enumerate(pts)]

    def cleanup_buf(self):
        for _, _, idx in self.handler_buf:
            self.canvas.delete(idx)
        # self.canvas.update()
        self.handler_buf.clear()
        self.current_curve = None
        self.current_handler_idx = -1

    def clear_pixels(self):
        self.canvas_pixels[:, :, :-1] = 255
        self.canvas_pixels[:, :, -1] = 255

    def paint_curves(self):
        self.clear_pixels()
        for curve in self.curves:
            for pixel in curve.rasterized_pixels:
                px, py, r, g, b, a = pixel.tolist()
                self.canvas_pixels[py, px, 0] = utils.get_int_color(r)
                self.canvas_pixels[py, px, 1] = utils.get_int_color(g)
                self.canvas_pixels[py, px, 2] = utils.get_int_color(b)
                self.canvas_pixels[py, px, 3] = utils.get_int_color(a)
        self.refresh_canvas_image()

    def paint_color_source(self):
        self.clear_pixels()
        for curve in self.curves:
            color_source = curve.color_source
            mask = curve.color_source_mask
            rows, cols = color_source.shape[:2]
            for i in range(rows):
                for j in range(cols):
                    _ = color_source[i, j, 0]
                    if mask[i, j]:
                        self.canvas_pixels[i, j, 0] = utils.get_int_color(color_source[i, j, 0])
                        self.canvas_pixels[i, j, 1] = utils.get_int_color(color_source[i, j, 1])
                        self.canvas_pixels[i, j, 2] = utils.get_int_color(color_source[i, j, 2])
                        self.canvas_pixels[i, j, 3] = 255
                    # self.canvas_pixels[:, :, 0] += color_source[:, :, 0].T
                    # self.canvas_pixels[:, :, 1] += color_source[:, :, 1].T
                    # self.canvas_pixels[:, :, 2] += color_source[:, :, 2].T
                    # self.canvas_pixels[:, :, 3] += 255
        self.refresh_canvas_image()

    def refresh_canvas_image(self):
        self.canvas_image = Image.fromarray(self.canvas_pixels, mode='RGBA')
        self.tkImg = ImageTk.PhotoImage(self.canvas_image)
        self.canvas.itemconfigure(self.sprite, image=self.tkImg)


if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(width=False, height=False)
    app = Application()
    root.mainloop()
