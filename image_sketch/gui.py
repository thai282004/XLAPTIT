import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from .io_utils import load_image, save_image, rgb_to_gray, gray_to_rgb
from .pipelines import (
    compute_histograms,
    painting_pipeline,
    pencil_sketch_bw,
    pencil_sketch_color,
)

# Color scheme: Modern light theme
PRIMARY_BG = "#f5f5f5"
SECONDARY_BG = "#ffffff"
ACCENT_COLOR = "#2563eb"
ACCENT_HOVER = "#1d4ed8"
TEXT_PRIMARY = "#1f2937"
TEXT_SECONDARY = "#6b7280"
BORDER_COLOR = "#e5e7eb"


class SectionFrame(tk.Frame):
    """Khung có tiêu đề + content bên trong."""
    def __init__(self, parent, title, *args, **kwargs):
        super().__init__(parent, bg=SECONDARY_BG, bd=1, relief=tk.SOLID, *args, **kwargs)

        header = tk.Frame(self, bg=PRIMARY_BG, height=30)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        lbl = tk.Label(header, text=title, bg=PRIMARY_BG, fg=TEXT_PRIMARY,
                       font=("Segoe UI", 9, "bold"))
        lbl.pack(side=tk.LEFT, padx=8, pady=5)

        self.content = tk.Frame(self, bg=SECONDARY_BG)
        self.content.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)


class RadioGroup(tk.Frame):
    """Group các radio button cho 1 StringVar."""
    def __init__(self, parent, variable, options, command=None, **kwargs):
        super().__init__(parent, bg=SECONDARY_BG, **kwargs)
        self.variable = variable
        for text, val in options:
            rb = tk.Radiobutton(
                self, text=text, variable=self.variable, value=val,
                bg=SECONDARY_BG, fg=TEXT_PRIMARY,
                selectcolor=ACCENT_COLOR, activebackground=SECONDARY_BG,
                font=("Segoe UI", 8), command=command
            )
            rb.pack(anchor="w", pady=1)


class SliderControl(tk.Frame):
    """Label + Scale gọn đẹp."""
    def __init__(self, parent, text, var,
                 from_, to, resolution=1.0, length=120,
                 fg=TEXT_PRIMARY):
        super().__init__(parent, bg=SECONDARY_BG)
        lbl = tk.Label(self, text=text, bg=SECONDARY_BG,
                       fg=fg, font=("Segoe UI", 7))
        lbl.pack(anchor="w", pady=(0, 1))
        scale = tk.Scale(
            self, from_=from_, to=to, resolution=resolution,
            orient=tk.HORIZONTAL, variable=var,
            bg=SECONDARY_BG, fg=TEXT_PRIMARY,
            highlightthickness=0, troughcolor=BORDER_COLOR,
            length=length
        )
        scale.pack(fill=tk.X)
        self.label = lbl
        self.scale = scale


class ImageSketchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image to Sketch & Cartoon Converter (Component UI)")
        self.root.geometry("1500x900")
        self.root.configure(bg=PRIMARY_BG)

        # -------- STATE ảnh full & preview --------
        self.pil_original_full = None
        self.arr_original_full = None
        self.gray_original_full = None

        self.pil_original = None      # preview PIL
        self.arr_original = None      # preview RGB
        self.gray_original = None     # preview gray

        self.result_gray = None       # preview result gray
        self.result_rgb_arr = None    # preview result RGB

        self.tk_original = None
        self.tk_result = None

        self._last_histograms = None

        # -------- Biến điều khiển --------
        self.mode_var = tk.StringVar(value="cartoon")      # cartoon / pencil_bw / pencil_color
        self.method_var = tk.StringVar(value="canny")      # sobel / log / canny
        self.smoothing_method = tk.StringVar(value="gaussian")

        self.sigma_s = tk.DoubleVar(value=2.0)
        self.sigma_r = tk.DoubleVar(value=25.0)

        self.sobel_threshold = tk.IntVar(value=80)
        self.log_sigma = tk.DoubleVar(value=1.0)
        self.log_kernel = tk.IntVar(value=5)

        self.canny_sigma = tk.DoubleVar(value=1.0)
        self.canny_low = tk.DoubleVar(value=0.1)
        self.canny_high = tk.DoubleVar(value=0.3)

        self.quant_levels = tk.IntVar(value=4)
        self.pencil_sigma = tk.DoubleVar(value=10.0)

        # progress + status
        self.progress_var = tk.DoubleVar(value=0)

        self.hist_canvas_original = None
        self.hist_canvas_result = None
        self.canvas_original = None
        self.canvas_result = None

        self._build_layout()

        self.method_var.trace_add("write", lambda *args: self._update_edge_param_visibility())
        self._update_edge_param_visibility()

        self._update_smoothing_sliders()

    # =============== GUI BUILD ===============

    def _build_layout(self):
        # Toolbar
        toolbar = tk.Frame(self.root, bg=SECONDARY_BG,
                           highlightthickness=1, highlightbackground=BORDER_COLOR)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        title = tk.Label(toolbar,
                         text="Image to Sketch & Cartoon Converter",
                         bg=SECONDARY_BG, fg=TEXT_PRIMARY,
                         font=("Segoe UI", 13, "bold"))
        title.pack(side=tk.LEFT, padx=12, pady=8)

        btn_frame = tk.Frame(toolbar, bg=SECONDARY_BG)
        btn_frame.pack(side=tk.RIGHT, padx=10, pady=8)

        tk.Button(btn_frame, text="📂 Open",
                  command=self.on_load_image,
                  bg=SECONDARY_BG, fg=TEXT_PRIMARY,
                  relief=tk.SOLID, bd=1,
                  font=("Segoe UI", 9),
                  cursor="hand2").pack(side=tk.LEFT, padx=3)

        tk.Button(btn_frame, text="💾 Save",
                  command=self.on_save_image,
                  bg=ACCENT_COLOR, fg="white",
                  activebackground=ACCENT_HOVER,
                  relief=tk.FLAT, bd=0,
                  font=("Segoe UI", 9, "bold"),
                  cursor="hand2").pack(side=tk.LEFT, padx=3)

        # Main container
        main = tk.Frame(self.root, bg=PRIMARY_BG)
        main.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        sidebar = tk.Frame(main, bg=PRIMARY_BG, width=340)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Scrollable area container (keeps apply button fixed below)
        scroll_container = tk.Frame(sidebar, bg=PRIMARY_BG)
        scroll_container.pack(fill=tk.BOTH, expand=True)

        canvas_scroll = tk.Canvas(scroll_container, bg=PRIMARY_BG,
                                  highlightthickness=0)
        scroll_frame = tk.Frame(canvas_scroll, bg=PRIMARY_BG)
        vbar = tk.Scrollbar(scroll_container, orient=tk.VERTICAL,
                            command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=vbar.set)

        # Create a window on the canvas and keep its id so we can
        # update the inner frame width when the canvas resizes.
        inner_id = canvas_scroll.create_window((0, 0), window=scroll_frame, anchor="nw")

        # When the inner frame changes size, update the scrollregion.
        def _on_frame_config(e):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        scroll_frame.bind("<Configure>", _on_frame_config)

        # When the canvas itself changes size (e.g. window resize),
        # update the inner window width so the content wraps properly
        # and the scrollbar doesn't disappear/overlap.
        def _on_canvas_config(e):
            canvas_scroll.itemconfigure(inner_id, width=e.width)
        canvas_scroll.bind("<Configure>", _on_canvas_config)

        # Mouse-wheel support (Windows/Mac/Linux). Use bind_all for
        # simplicity so wheel works when pointer is over the sidebar.
        def _on_mousewheel(e):
            # Linux: Button-4 / Button-5 events use e.num
            if hasattr(e, 'num') and e.num in (4, 5):
                delta = -1 if e.num == 4 else 1
                canvas_scroll.yview_scroll(delta, "units")
            else:
                # Windows / macOS: e.delta is multiple of 120
                try:
                    step = int(-e.delta / 120)
                except Exception:
                    step = 0
                if step != 0:
                    canvas_scroll.yview_scroll(step, "units")

        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)
        canvas_scroll.bind_all("<Button-4>", _on_mousewheel)
        canvas_scroll.bind_all("<Button-5>", _on_mousewheel)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Apply button container fixed at bottom so it's always visible
        apply_frame = tk.Frame(sidebar, bg=PRIMARY_BG)
        apply_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

        # Content area
        content = tk.Frame(main, bg=PRIMARY_BG)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ----- Sidebar sections -----
        tk.Label(scroll_frame, text="Controls", bg=PRIMARY_BG,
                 fg=TEXT_PRIMARY, font=("Segoe UI", 11, "bold")).pack(
                     anchor="w", padx=10, pady=(10, 4))

        # Output style
        sec_mode = SectionFrame(scroll_frame, "Output Style")
        sec_mode.pack(fill=tk.X, padx=10, pady=4)
        RadioGroup(sec_mode.content, self.mode_var, [
            ("Cartoon (quantize + edges)", "cartoon"),
            ("Pencil Sketch B&W", "pencil_bw"),
            ("Pencil Sketch Color", "pencil_color"),
        ]).pack(anchor="w")

        # Edge
        sec_edge = SectionFrame(scroll_frame, "Edge Detection")
        sec_edge.pack(fill=tk.X, padx=10, pady=4)
        RadioGroup(sec_edge.content, self.method_var, [
            ("Sobel", "sobel"),
            ("LoG (fast)", "log"),
            ("Canny", "canny"),
        ], command=self._update_edge_param_visibility).pack(anchor="w")

        self.sobel_param_frame = tk.Frame(sec_edge.content, bg=SECONDARY_BG)
        SliderControl(self.sobel_param_frame, "Threshold",
                      self.sobel_threshold, 0, 255, 1).pack(fill=tk.X)

        self.log_param_frame = tk.Frame(sec_edge.content, bg=SECONDARY_BG)
        SliderControl(self.log_param_frame, "Sigma (Gaussian)",
                      self.log_sigma, 0.5, 5.0, 0.1).pack(fill=tk.X)
        SliderControl(self.log_param_frame, "Kernel size (ít ảnh hưởng)",
                      self.log_kernel, 3, 15, 2, fg=TEXT_SECONDARY).pack(fill=tk.X)

        self.canny_param_frame = tk.Frame(sec_edge.content, bg=SECONDARY_BG)
        SliderControl(self.canny_param_frame, "Gaussian Sigma",
                      self.canny_sigma, 0.5, 5.0, 0.1).pack(fill=tk.X)
        SliderControl(self.canny_param_frame, "Low threshold ratio",
                      self.canny_low, 0.01, 0.5, 0.01).pack(fill=tk.X)
        SliderControl(self.canny_param_frame, "High threshold ratio",
                      self.canny_high, 0.1, 1.0, 0.01).pack(fill=tk.X)

        # Smoothing
        sec_smooth = SectionFrame(scroll_frame, "Smoothing")
        sec_smooth.pack(fill=tk.X, padx=10, pady=4)
        RadioGroup(sec_smooth.content, self.smoothing_method, [
            ("Gaussian (fast)", "gaussian"),
            ("Bilateral (edge-preserving)", "bilateral"),
        ], command=self._update_smoothing_sliders).pack(anchor="w")

        self.sigma_s_ctrl = SliderControl(sec_smooth.content,
                                          "Sigma (spatial / Gaussian)",
                                          self.sigma_s, 1.0, 5.0, 0.5)
        self.sigma_s_ctrl.pack(fill=tk.X, pady=(4, 0))

        self.sigma_r_ctrl = SliderControl(sec_smooth.content,
                                          "Sigma (range, Bilateral)",
                                          self.sigma_r, 5.0, 80.0, 5.0)
        self.sigma_r_ctrl.pack(fill=tk.X, pady=(2, 0))

        # Cartoon
        sec_cartoon = SectionFrame(scroll_frame, "Cartoon Effect")
        sec_cartoon.pack(fill=tk.X, padx=10, pady=4)
        SliderControl(sec_cartoon.content,
                      "Color Levels (3-12)",
                      self.quant_levels, 3, 12, 1).pack(fill=tk.X)

        # Pencil
        sec_pencil = SectionFrame(scroll_frame, "Pencil Sketch")
        sec_pencil.pack(fill=tk.X, padx=10, pady=4)
        SliderControl(sec_pencil.content,
                      "Blur Sigma",
                      self.pencil_sigma, 3.0, 20.0, 1.0).pack(fill=tk.X)

        tk.Button(apply_frame, text="▶ Apply Processing",
              command=self.on_apply,
              bg=ACCENT_COLOR, fg="white",
              activebackground=ACCENT_HOVER,
              relief=tk.FLAT, bd=0,
              font=("Segoe UI", 10, "bold"),
              cursor="hand2").pack(fill=tk.X, padx=6, pady=4)

        # ----- Content: images + hist -----
        img_panel = tk.Frame(content, bg=PRIMARY_BG)
        img_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = tk.Frame(img_panel, bg=PRIMARY_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Label(left, text="Original (Preview)",
                 bg=PRIMARY_BG, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 3))
        self.canvas_original = tk.Label(left, text="No image",
                                        bg=SECONDARY_BG, fg=TEXT_SECONDARY,
                                        relief=tk.FLAT, bd=1,
                                        highlightthickness=1,
                                        highlightbackground=BORDER_COLOR)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(img_panel, bg=PRIMARY_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        tk.Label(right, text="Processed (Preview)",
                 bg=PRIMARY_BG, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 3))
        self.canvas_result = tk.Label(right, text="Result will appear here",
                                      bg=SECONDARY_BG, fg=TEXT_SECONDARY,
                                      relief=tk.FLAT, bd=1,
                                      highlightthickness=1,
                                      highlightbackground=BORDER_COLOR)
        self.canvas_result.pack(fill=tk.BOTH, expand=True)

        bottom = tk.Frame(content, bg=PRIMARY_BG)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))

        tk.Label(bottom, text="Histograms (Preview Gray)",
                 bg=PRIMARY_BG, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")

        hist_row = tk.Frame(bottom, bg=PRIMARY_BG)
        hist_row.pack(fill=tk.X, pady=2)

        self.hist_canvas_original = tk.Canvas(hist_row, height=90,
                                              bg=SECONDARY_BG,
                                              highlightthickness=1,
                                              highlightbackground=BORDER_COLOR)
        self.hist_canvas_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        self.hist_canvas_result = tk.Canvas(hist_row, height=90,
                                            bg=SECONDARY_BG,
                                            highlightthickness=1,
                                            highlightbackground=BORDER_COLOR)
        self.hist_canvas_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))

        status = tk.Frame(bottom, bg=SECONDARY_BG,
                          highlightthickness=1,
                          highlightbackground=BORDER_COLOR)
        status.pack(fill=tk.X, pady=(6, 0))

        self.status_label = tk.Label(status, text="Ready",
                                     bg=SECONDARY_BG, fg=TEXT_PRIMARY,
                                     font=("Segoe UI", 8))
        self.status_label.pack(side=tk.LEFT, padx=8, pady=4)

        self.progress_canvas = tk.Canvas(status, height=14,
                                         bg=BORDER_COLOR, highlightthickness=0)
        self.progress_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=4)

        self.progress_text = tk.Label(status, text="0%",
                                      bg=SECONDARY_BG, fg=TEXT_PRIMARY,
                                      font=("Segoe UI", 7))
        self.progress_text.pack(side=tk.LEFT, padx=8, pady=4)

    # =============== UI UPDATE ===============

    def _update_edge_param_visibility(self):
        for f in (self.sobel_param_frame,
                  self.log_param_frame,
                  self.canny_param_frame):
            f.pack_forget()

        method = self.method_var.get()
        if method == "sobel":
            self.sobel_param_frame.pack(fill=tk.X, pady=(4, 0))
        elif method == "log":
            self.log_param_frame.pack(fill=tk.X, pady=(4, 0))
        elif method == "canny":
            self.canny_param_frame.pack(fill=tk.X, pady=(4, 0))

    def _update_smoothing_sliders(self):
        method = self.smoothing_method.get()

        if method == "gaussian":
            for child in self.sigma_s_ctrl.winfo_children():
                try:
                    child.configure(state=tk.NORMAL)
                except Exception:
                    pass
            for child in self.sigma_r_ctrl.winfo_children():
                try:
                    child.configure(state=tk.DISABLED)
                except Exception:
                    pass
        else:  # bilateral
            for child in self.sigma_s_ctrl.winfo_children():
                try:
                    child.configure(state=tk.NORMAL)
                except Exception:
                    pass
            for child in self.sigma_r_ctrl.winfo_children():
                try:
                    child.configure(state=tk.NORMAL)
                except Exception:
                    pass

    def _set_progress(self, value, msg=None):
        value = max(0, min(100, value))
        self.progress_var.set(value)
        self.progress_text.config(text=f"{int(value)}%")
        self.progress_canvas.delete("all")
        w = self.progress_canvas.winfo_width() or 1
        h = self.progress_canvas.winfo_height() or 1
        fw = w * (value / 100.0)
        self.progress_canvas.create_rectangle(0, 0, fw, h,
                                              fill=ACCENT_COLOR,
                                              outline="")
        if msg is not None:
            self.status_label.config(text=msg)
        self.root.update_idletasks()

    # =============== IMAGE HANDLERS ===============

    def on_load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")])
        if not path:
            return
        try:
            self.pil_original_full, self.arr_original_full = load_image(path)
            self.gray_original_full = rgb_to_gray(self.arr_original_full)

            # preview
            pil_prev = self.pil_original_full.copy()
            pil_prev.thumbnail((1200, 1200))
            self.pil_original = pil_prev
            self.arr_original = np.array(pil_prev).astype(np.float32)
            self.gray_original = rgb_to_gray(self.arr_original)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.result_gray = None
        self.result_rgb_arr = None
        self._last_histograms = None
        self._update_image_views()
        self._clear_histograms()
        self._set_progress(0, "Ready")

    def on_save_image(self):
        if self.result_rgb_arr is None:
            messagebox.showwarning("Warning", "No result to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")])
        if not path:
            return
        save_image(path, self.result_rgb_arr)

    # =============== PROCESSING (THREAD) ===============

    def on_apply(self):
        if self.gray_original is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        self.root.config(cursor="wait")
        self._set_progress(5, "Processing...")
        t = threading.Thread(target=self._worker_apply, daemon=True)
        t.start()

    def _worker_apply(self):
        try:
            mode = self.mode_var.get()
            method = self.method_var.get()
            smoothing = self.smoothing_method.get()

            gray = self.gray_original.copy()
            rgb = self.arr_original.copy()

            if mode == "cartoon":
                edge_params = {}
                if method == "sobel":
                    edge_params["threshold"] = self.sobel_threshold.get()
                elif method == "log":
                    edge_params["sigma"] = float(self.log_sigma.get())
                    edge_params["kernel_size"] = int(self.log_kernel.get())
                elif method == "canny":
                    edge_params["sigma"] = float(self.canny_sigma.get())
                    edge_params["low_ratio"] = float(self.canny_low.get())
                    edge_params["high_ratio"] = float(self.canny_high.get())

                smoothing_params = {
                    "sigma_s": float(self.sigma_s.get()),
                    "sigma_r": float(self.sigma_r.get())
                }
                smoothing_sigma = float(self.sigma_s.get())
                quant = int(self.quant_levels.get())

                res, edges = painting_pipeline(
                    gray,
                    method=method,
                    smoothing_method=smoothing,
                    smoothing_sigma=smoothing_sigma,
                    smoothing_params=smoothing_params,
                    edge_params=edge_params,
                    quant_levels=quant
                )
                result_gray = res
                result_rgb = gray_to_rgb(res)

            elif mode == "pencil_bw":
                sigma_p = float(self.pencil_sigma.get())
                res = pencil_sketch_bw(gray, sigma=sigma_p)
                result_gray = res
                result_rgb = gray_to_rgb(res)

            elif mode == "pencil_color":
                sigma_p = float(self.pencil_sigma.get())
                sketch_gray = pencil_sketch_bw(gray, sigma=sigma_p)
                color = pencil_sketch_color(rgb, sketch_gray)
                result_gray = sketch_gray
                result_rgb = color

            else:
                raise ValueError(f"Unknown mode: {mode}")

            h_orig, h_proc = compute_histograms(gray, result_gray)

            self.result_gray = result_gray
            self.result_rgb_arr = result_rgb
            self._last_histograms = (h_orig, h_proc)

            self.root.after(0, self._apply_done)
        except Exception as e:
            self.root.after(0, lambda: self._apply_failed(e))

    def _apply_done(self):
        self._update_image_views()
        if self._last_histograms is not None:
            h_orig, h_proc = self._last_histograms
            self._draw_histogram(self.hist_canvas_original, h_orig, "Original")
            self._draw_histogram(self.hist_canvas_result, h_proc, "Processed")
        self._set_progress(100, "Done!")
        self.root.config(cursor="")
    
    def _apply_failed(self, e):
        messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
        self._set_progress(0, "Error")
        self.root.config(cursor="")

    # =============== VIEW / HIST ===============

    def _update_image_views(self):
        if self.pil_original is not None:
            disp = self._resize_for_display(self.pil_original)
            self.tk_original = ImageTk.PhotoImage(disp)
            self.canvas_original.config(image=self.tk_original, text="")
            self.canvas_original.image = self.tk_original

        if self.result_rgb_arr is not None:
            pil_res = Image.fromarray(self.result_rgb_arr.astype(np.uint8))
            disp = self._resize_for_display(pil_res)
            self.tk_result = ImageTk.PhotoImage(disp)
            self.canvas_result.config(image=self.tk_result, text="")
            self.canvas_result.image = self.tk_result
        else:
            self.canvas_result.config(text="Result will appear here", image=None)

    def _resize_for_display(self, pil_img, max_size=500):
        w, h = pil_img.size
        scale = min(max_size / w, max_size / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        return pil_img.resize((nw, nh))

    def _clear_histograms(self):
        self.hist_canvas_original.delete("all")
        self.hist_canvas_result.delete("all")

    def _draw_histogram(self, canvas, hist, title):
        canvas.delete("all")
        w = canvas.winfo_width() or 300
        h = canvas.winfo_height() or 90
        max_val = max(hist) if max(hist) > 0 else 1
        bar_w = (w - 8) / 256.0

        canvas.create_text(4, 2, text=title, anchor="nw",
                           font=("Segoe UI", 7, "bold"),
                           fill=TEXT_PRIMARY)

        for i, v in enumerate(hist):
            x0 = 4 + i * bar_w
            x1 = 4 + (i + 1) * bar_w
            bar_h = (v / max_val) * (h - 18)
            y0 = h - 2 - bar_h
            y1 = h - 2
            canvas.create_rectangle(x0, y0, x1, y1,
                                    fill=ACCENT_COLOR, outline="")
