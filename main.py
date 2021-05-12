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


class Mode(Enum):
    CreateMode = auto()
    SelectMode = auto()
    ColorMode = auto()


class Application(tk.Frame):

    def __init__(self):
        super().__init__()

        self.width = 1000  # canvas width
        self.height = 800  # canvas height
        self.center = self.height // 2
        self.white = (255, 255, 255)  # canvas back
        self.button_create_text = 'Select'
        self.button_add_text = 'Create'
        self.button_add_done_text = 'Finish Creation'
        self.button_color_text = 'Pick Color'
        self.circle_radius = 1

        self.curves: [DiffusionCurve] = []
        self.selected_curve: Optional[DiffusionCurve] = None
        self.selected_ctx_pt_idx: int = -1
        self.circle_buf: [(int, int, int)] = []
        self.mode = Mode.SelectMode
        self.curve_selection_threshold = 40 ** 2  # 5px of dist from selection curve
        self.pts_selection_threshold = 10 ** 2  # 5px of dist from selection point

        self.canvas_pixels = 255*np.ones((self.height, self.width, 4), dtype=np.uint8)
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
        self.button_select = tk.Button(control_frame, width=10, text=self.button_create_text,
                                       command=self.on_click_select)
        self.button_select.pack(side=tk.TOP)
        self.button_create = tk.Button(control_frame, width=10, text=self.button_add_text,
                                       command=self.on_click_create)
        self.button_create.pack(side=tk.TOP)
        self.button_color = tk.Button(control_frame, width=10, text=self.button_color_text,
                                      command=self.on_click_color)
        self.button_color.pack(side=tk.TOP)

        self.canvas_image = Image.fromarray(self.canvas_pixels, mode='RGB')
        self.tkImg = ImageTk.PhotoImage(self.canvas_image)
        self.sprite = self.canvas.create_image(self.width / 2, self.height / 2, image=self.tkImg)

    def on_click_select(self):
        self.reset_style()
        self.cleanup_buf()
        self.mode = Mode.SelectMode

    def on_click_create(self):
        self.reset_style()
        self.cleanup_buf()
        if self.mode is not Mode.CreateMode:
            self.button_create['fg'] = '#43454a'
            self.button_create['font'] = '-weight bold'
            self.button_create['text'] = self.button_add_done_text
            self.mode = Mode.CreateMode

    def on_click_color(self):
        self.reset_style()
        self.cleanup_buf()
        self.mode = Mode.ColorMode

    def on_click_canvas(self, event):
        if self.mode is Mode.CreateMode:
            circle_idx = utils.create_circle(self.canvas, (event.x, event.y), self.circle_radius)
            self.circle_buf.append((event.x, event.y, circle_idx))
            if len(self.circle_buf) == 4:
                curve = DiffusionCurve()
                curve.ctx_points = [(x / self.width, y / self.height) for x, y, _ in self.circle_buf]
                curve.rasterize(self.width, self.height)
                self.curves.append(curve)
                self.cleanup_buf()
                self.paint_curves()
        elif self.mode is Mode.SelectMode:
            has_sth_selected = False
            # select curve
            if self.curves:
                dist = [utils.calc_sqr_dist_from_curve(event.x, event.y, c.rasterized_pixels) for c in self.curves]
                min_dist = np.min(dist)
                min_curve_id = np.argmin(dist)
                if min_dist < self.curve_selection_threshold:
                    self.selected_curve = self.curves[min_curve_id]
                    self.selected_ctx_pt_idx = -1
                    has_sth_selected = True
            # draw ctx points
            if self.selected_curve:
                ctx_pts = self.selected_curve.rasterize_ctx_pts(self.width, self.height)
                self.draw_circle_buf(ctx_pts)
            # select ctx points
            if self.circle_buf:
                dist = [utils.calc_sqr_dist_from_pt(event.x, event.y, p[0], p[1]) for p in self.circle_buf]
                min_dist = np.min(dist)
                if min_dist < self.pts_selection_threshold:
                    self.selected_ctx_pt_idx = np.argmin(dist)
                    has_sth_selected = True
            # nothing is selected (too far from ctx pts and curves)
            if not has_sth_selected:
                self.cleanup_buf()
                self.canvas.update()
        elif self.mode is Mode.ColorMode:
            pass

    def on_move_canvas(self, event):
        if self.selected_ctx_pt_idx > 0 and self.circle_buf:
            cid = self.circle_buf[self.selected_ctx_pt_idx][-1]
            self.canvas.moveto(cid, event.x-6*self.circle_radius, event.y-6*self.circle_radius)
            self.canvas.itemconfig(cid, outline='red')

            # update curve
            self.clear_pixels()
            self.selected_curve.ctx_points[self.selected_ctx_pt_idx] = (event.x / self.width, event.y / self.height)
            self.selected_curve.rasterize(self.width, self.height)
            self.paint_curves()

    def on_release_canvas(self, event):
        if self.selected_curve:
            self.draw_circle_buf(self.selected_curve.rasterize_ctx_pts(self.width, self.height))
        self.selected_ctx_pt_idx = -1

    def reset_style(self):
        self.button_create['fg'] = 'black'
        self.button_create['font'] = '-weight normal'
        self.button_create['text'] = self.button_add_text
        self.mode = Mode.SelectMode

    def draw_circle_buf(self, pts: [(int, int)]):
        _ = [self.canvas.delete(idx) for _, _, idx in self.circle_buf]
        circle_idx = [utils.create_circle(self.canvas, (cx, cy), self.circle_radius) for cx, cy in pts]
        self.circle_buf = [(p[0], p[1], circle_idx[i]) for i, p in enumerate(pts)]

    def cleanup_buf(self):
        for _, _, idx in self.circle_buf:
            self.canvas.delete(idx)
        self.circle_buf.clear()
        self.selected_curve = None
        self.selected_ctx_pt_idx = -1

    def clear_pixels(self):
        self.canvas_pixels.fill(0)

    def paint_curves(self):
        for curve in self.curves:
            for pixel in curve.rasterized_pixels:
                px, py, r, g, b, a = pixel.tolist()
                self.canvas_pixels[py, px, 0] = r
                self.canvas_pixels[py, px, 1] = g
                self.canvas_pixels[py, px, 2] = b
                self.canvas_pixels[py, px, 3] = a
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
