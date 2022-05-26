import tkinter as tk
from PIL import ImageTk, Image
import tkinter.ttk as ttk

from async_opeations import Generator
from utils import process_str_value

class GenerateDialog(tk.Toplevel):
    def __init__(self, window=None):
        super().__init__()
        self.window = window

        self.title("Generate")
        self.resizable(0, 0)
        self.frame = tk.Frame(self, relief='flat', borderwidth=5)
        self.frame.grid(row=0, column=0)

        self.show_interface()
        self.result = None
        self.paths = None
        self.window = window

        self.max_w, self.max_h = 1025, 1024
        self.default_w, self.default_h = 256, 256
        self.default_n_iter = 20

        self.generated_texture = None

    def show_interface(self):
        self.settings_panel_frame = self.settings_panel(self.frame)
        self.settings_panel_frame.grid(row=0, column=0, sticky='NWE')

        self.buttons_frame = tk.Frame(self.frame, relief='flat', borderwidth=0, pady=5)

        self.generate_btn = tk.Button(self.buttons_frame, text="Generate", command=self.generate)
        self.generate_btn.grid(row=0, column=0, padx=[5, 0], sticky='WE')

        self.save_btn = tk.Button(self.buttons_frame, text="Save", state='disable', command=self.save_image)
        self.save_btn.grid(row=1, column=0, padx=[5, 0], sticky='WE')

        self.buttons_frame.grid(row=1, column=0, sticky='SWE')

        
        self.result_panel_frame = self.result_panel(self.frame)
        self.result_panel_frame.grid(row=0, column=1, rowspan=2, padx=[10, 0])

        self.pb_label = tk.Label(self.frame, text='', font=("Courier", 8))
        self.pb_label.grid(row=2, column=0, columnspan=2, sticky='W')
        self.pb = ttk.Progressbar(
            self.frame,
            orient='horizontal',
            mode='determinate',
        )
        self.pb.grid(row=3, column=0, columnspan=2, pady=[0, 5], sticky='EW')

    def result_panel(self, parent):
        result_panel_frame = tk.LabelFrame(parent, text='Output', padx=5, pady=5)
        pil_image = Image.new("RGB", (512, 512), (255, 255, 255))
        self.img = ImageTk.PhotoImage(pil_image)
        
        self.image_label = tk.Label(result_panel_frame, image=self.img)
        self.image_label.grid(row=0, column=0, padx=5, pady=5)
        return result_panel_frame

    def settings_panel(self, parent):
        settings_panel_frame = tk.LabelFrame(parent, text='Settings', padx=5, pady=5)

        self.w_label = tk.Label(settings_panel_frame, text="Width:")
        self.w_label.grid(row=1, column=0, sticky='W')
        self.w_entry = tk.Entry(settings_panel_frame, width=5)
        self.w_entry.grid(row=1, column=1, sticky='W', padx=5)

        self.h_label = tk.Label(settings_panel_frame, text="Height:")
        self.h_label.grid(row=2, column=0, sticky='W')
        self.h_entry = tk.Entry(settings_panel_frame, width=5)
        self.h_entry.grid(row=2, column=1, sticky='W', padx=5)

        self.n_iter_label = tk.Label(settings_panel_frame, text="Iterations:")
        self.n_iter_label.grid(row=4, column=0, sticky='W')
        self.n_iter_entry = tk.Entry(settings_panel_frame, width=5)
        self.n_iter_entry.grid(row=4, column=1, sticky='W', padx=5)

        return settings_panel_frame

    def update_image(self, generated_texture):
        self.generated_texture = generated_texture
        self.img = ImageTk.PhotoImage(generated_texture)
        self.image_label.config(image=self.img)

    def generate(self):

        w = process_str_value(
            self.h_entry.get(), 
            default=self.default_w, 
            target=int, 
            min_value=64, 
            max_value=self.max_w
        )
        h = process_str_value(
            self.h_entry.get(), 
            default=self.default_h, 
            target=int, 
            min_value=64, 
            max_value=self.max_h
        )
        n_iter = process_str_value(self.n_iter_entry.get(), default=self.default_n_iter, target=int, min_value=1)

        self.pb_label.config(text='Generating...')
        self.generate_btn.config(state='disable')
        self.save_btn.config(state='disable')
        self.pb.start()
        generator_thread = Generator(self.window.SG, self.window.out, [w, h], n_iter)
        generator_thread.start()

        self.monitor_generation(generator_thread)

    def monitor_generation(self, generator_thread):
        if generator_thread.is_alive():
            self.after(100, lambda: self.monitor_generation(generator_thread))
        else:
            self.pb.stop()
            self.generate_btn.config(state='active')
            self.save_btn.config(state='active')
            self.pb_label.config(text='Done!')
            self.update_image(generator_thread.generated_texture)

    def save_image(self):
        try:
            f = tk.filedialog.asksaveasfile(
                title='Save image',
                mode='a', 
                defaultextension='.png', 
                initialdir='./'
            )
            if f is None:
                return
            self.generated_texture.save(f.name)
            f.close()
        except Exception as e:
            print(e)
            tk.messagebox.showerror("Exception", "Unable to save image")
