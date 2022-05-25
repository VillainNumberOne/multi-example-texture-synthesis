import tkinter as tk
import tkinter.ttk as ttk
from tkinter import Image, filedialog as fd
from PIL import Image
from tkinter import messagebox

from src.texture_synthesis.function import image_to_tensor, square_crop
from async_opeations import ImageLoader
from utils import process_str_value


class OpenImageDialog(tk.Toplevel):
    def __init__(self, window=None, mode='single'):
        super().__init__()
        self.window = window

        self.title("Open Image")
        self.resizable(0, 0)
        self.frame = tk.Frame(self, relief='flat', borderwidth=5)
        self.frame.grid(row=0, column=0)

        self.show_interface()
        self.result = None
        self.paths = None
        self.window = window

    def show_interface(self):
        self.show()

    def show(self):

        self.path_label = tk.Label(self.frame, text="Path:")
        self.path_label.grid(row=0, column=0, sticky='W')
        self.path_entry = tk.Entry(self.frame, width=30)
        self.path_entry.grid(row=0, column=1, padx=5)
        self.path_btn = tk.Button(self.frame, text='Search', command=self.open_file)
        self.path_btn.grid(row=0, column=2)

        self.resize_w_label = tk.Label(self.frame, text="Resize width:")
        self.resize_w_label.grid(row=1, column=0, sticky='W')
        self.resize_w_entry = tk.Entry(self.frame, width=5)
        self.resize_w_entry.grid(row=1, column=1, sticky='W', padx=5)

        self.resize_h_label = tk.Label(self.frame, text="Resize height:")
        self.resize_h_label.grid(row=2, column=0, sticky='W')
        self.resize_h_entry = tk.Entry(self.frame, width=5)
        self.resize_h_entry.grid(row=2, column=1, sticky='W', padx=5)

        self.crop_checkbox_var = tk.IntVar()
        self.crop_checkbox = tk.Checkbutton(self.frame, text='Square Crop', variable=self.crop_checkbox_var)
        self.crop_checkbox.grid(row=3, column=0, sticky='W', padx=5)

        self.scale_label = tk.Label(self.frame, text="Scale:")
        self.scale_label.grid(row=4, column=0, sticky='W')
        self.scale_entry = tk.Entry(self.frame, width=5)
        self.scale_entry.grid(row=4, column=1, sticky='W', padx=5)

        self.status_label = tk.Label(self.frame, text='')
        self.status_label.grid(row=5, column=0, sticky='W')
        self.cancel_btn = tk.Button(self.frame, text='Cancel', command=self.destroy)
        self.cancel_btn.grid(row=5, column=1, sticky='E')
        self.open_btn = tk.Button(self.frame, text='Open', command=self.load_image)
        self.open_btn.grid(row=5, column=2, sticky='E')

    def disable_all(self):
        for child in self.frame.winfo_children():
            child.configure(state='disable')

    def enable_all(self):
        for child in self.frame.winfo_children():
            child.configure(state='enable')

    def open_file(self, mode='many'):
        filetypes = [('image files', ('.png', '.jpg'))]

        if mode == 'single':
            paths = [fd.askopenfilename(
                title='Open image',
                initialdir='./',
                filetypes=filetypes
            )]
        elif mode == 'many':
            paths = fd.askopenfilenames(
                parent=self.frame,
                title='Open images',
                initialdir='./',
                filetypes=filetypes
            )

        self.paths = paths
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(tk.END, ','.join(paths))

    def get_result(self):
        self.deiconify()
        self.wait_window()
        return self.result

    def load_image(self):
        self.disable_all()
        crop = self.crop_checkbox_var.get()
        resize_w = process_str_value(self.resize_w_entry.get())
        resize_h = process_str_value(self.resize_h_entry.get())
        scale = process_str_value(self.scale_entry.get())
        if scale is None:
            scale = 0

        pil_images = []
        try:
            for path in self.paths:
                pil_image = Image.open(path).convert("RGB")

                if bool(crop):
                    pil_image = square_crop(pil_image)
                w, h = pil_image.size
                if resize_w is None:
                    resize_w = w
                if resize_h is None:
                    resize_h = h

                pil_image = pil_image.resize((resize_w, resize_h))
                pil_images.append(pil_image)

        except Exception:
            messagebox.showerror("Exception", "Unable to load image")
            self.destroy()

        self.status_label.config(text='Loading...')
        thread = ImageLoader(
            self.window.FE, 
            pil_images, 
            scale, 
            self.window.device
        )
        thread.start()
        self.monitor_image_loading(thread)

    def monitor_image_loading(self, thread):
        if thread.is_alive():
            self.frame.after(100, lambda: self.monitor_image_loading(thread))
        else:
            self.status_label.config(text='Done!')
            self.result = thread.image_tensors
            self.destroy()
