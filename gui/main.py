import tkinter as tk
import tkinter.ttk as ttk
from PIL import ImageTk, Image
import torch

from mexts.feature_extractor import FeatureExtractor
from mexts.gen import TextureGen
from mexts.adain_autoencoder import AdaINAutoencoder
from mexts.style_features_manipulation import style_attribute_extraction_means, style_attribute_extraction_svm

from async_opeations import Generator
from generate import GenerateDialog
from open_files import OpenImageDialog
from utils import process_str_value, process_tensor_lists, process_tensors
from warnings_errors import *


class Window():
    def __init__(self):

        self.font = ("Courier", 12)
        self.window = tk.Tk()
        self.window.title("MExTS-gui")
        self.window.resizable(0, 0)
        self.preview_size = [256, 256]

        self.buttons_to_disable = []
        self.open_image_buttons = []
        self.generate_dialog_window = None

        self._init_models()
        self.show_interface()
        self.out = None

        self.interpolation_slots = [None, None]
        self.linear_slots = [None, None]
        self.extraction_slots = [None, None]

    def _init_models(self):
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')
        self.FE = FeatureExtractor().to(self.device)
        self.AA = AdaINAutoencoder().to(self.device)
        self.G = TextureGen(self.FE, self.AA, self.device)

    def disable_buttons(self):
        for btn in self.buttons_to_disable:
            btn.configure(state='disable')

        if self.generate_dialog_window is not None:
            self.generate_dialog_window.generate_btn.configure(state='disable')

    def activate_buttons(self):
        for btn in self.buttons_to_disable:
            btn.configure(state='active')

        if self.generate_dialog_window is not None:
            self.generate_dialog_window.generate_btn.configure(state='active')

    def disable_open_image(self):
        for btn in self.open_image_buttons:
            btn.configure(state='disable')

    def enable_open_image(self):
        for btn in self.open_image_buttons:
            btn.configure(state='active')

    def show_interface(self):
        self.output_frame = tk.Frame(self.window, relief='flat', borderwidth=5)

        self.output_panel_frame = self.output_panel(self.output_frame)
        self.output_panel_frame.grid(row=0, column=1)

        self.output_frame.grid(row=0, column=1, sticky='N')

        # Tools
        self.tools_frame = tk.Frame(self.window, relief='flat', borderwidth=5)

        self.simple_generation_tool_frame = self.simple_generation_tool(self.tools_frame)
        self.simple_generation_tool_frame.grid(row=0, column=0, sticky='NWE')

        self.interpolation_tool_frame = self.interpolation_tool(self.tools_frame)
        self.interpolation_tool_frame.grid(row=1, column=0, sticky='NWE')

        self.linear_operation_tool_frame = self.linear_operation_tool(self.tools_frame)
        self.linear_operation_tool_frame.grid(row=2, column=0, sticky='NWE')

        self.extraction_tool_frame = self.extraction_tool(self.tools_frame)
        self.extraction_tool_frame.grid(row=3, column=0, sticky='NWE')

        self.tools_frame.grid(row=0, column=0, sticky='N')

        # Progress bar
        self.pb_label = tk.Label(self.window, text='', font=("Courier", 8))
        self.pb_label.grid(row=3, column=0, columnspan=2, sticky='W')
        self.pb = ttk.Progressbar(
            self.window,
            orient='horizontal',
            mode='determinate',
        )
        self.pb.grid(row=4, column=0, columnspan=2, padx=5, pady=[0, 5], sticky='EW')
        

    def output_panel(self, parent):
        output_panel_frame = tk.LabelFrame(parent, text='Results', padx=5, pady=5)
        
        pil_image = Image.new("RGB", self.preview_size, (255, 255, 255))
        self.preview_img = ImageTk.PhotoImage(pil_image)
        
        self.image_label = tk.Label(output_panel_frame, image = self.preview_img)
        self.image_label.grid(row=0, column=0, padx=5, pady=5)

        self.generate_preview_btn = tk.Button(output_panel_frame, text="Generate Preview", command=self.generate_preview, state='disable')
        self.generate_preview_btn.grid(row=1, column=0, sticky='EW')

        tk.Label(output_panel_frame, height=1).grid(row=2, column=0)

        self.save_tensor_btn = tk.Button(output_panel_frame, text="Save Vector", command=self.save_tensor, state='disable')
        self.save_tensor_btn.grid(row=3, column=0, sticky='EW')

        self.generate_btn = tk.Button(output_panel_frame, text="Generate", command=self.generate, state='active')
        self.generate_btn.grid(row=4, column=0, sticky='EW')

        self.buttons_to_disable += [self.generate_preview_btn]

        return output_panel_frame

    def update_out(self, style_tensor):
        self.out = style_tensor
        self.generate_preview_btn.configure(state='active')
        self.save_tensor_btn.configure(state='active')

    def interpolation_tool(self, parent):
        def action(style_tensor, slot):
            assert slot in [0, 1]
            if style_tensor is not None:
                self.interpolation_slots[slot] = style_tensor
                statuses[slot].config(text='Loaded')

                if None not in self.interpolation_slots:
                    compute_btn.config(state='active')

        action1 = lambda style_tensor: action(style_tensor, 0)
        action2 = lambda style_tensor: action(style_tensor, 1)

        def compute(alpha):
            assert None not in self.interpolation_slots
            alpha = process_str_value(alpha, 0, float)
            if not 0 <= alpha <= 1:
                invalid_alpha()
                return
            t1, t2 = process_tensors(self.linear_slots[0], self.linear_slots[1])

            result_tensor = t1 * (1 - alpha) + t2 * alpha
            self.update_out(result_tensor)

        
        tool_frame = tk.LabelFrame(parent, text='Interpolation', padx=5, pady=5)
        statuses = [None, None]

        texture1_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture1_label = tk.Label(texture1_frame, text='Texture 1:')
        statuses[0] = tk.Label(texture1_frame, text='Empty')
        texture1_label.grid(row=0, column=0, sticky='W')
        statuses[0].grid(row=0, column=1, sticky='W')

        open_image_btn1 = tk.Button(texture1_frame, text="Image", command=lambda: self.open(action1, 'image'))
        self.open_image_buttons += [open_image_btn1]
        open_tensor_btn1 = tk.Button(texture1_frame, text="Vector", command=lambda: self.open(action1, 'tensor'))
        open_image_btn1.grid(row=0, column=2)
        open_tensor_btn1.grid(row=0, column=3)

        texture2_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture2_label = tk.Label(texture2_frame, text='Texture 2:')
        statuses[1] = tk.Label(texture2_frame, text='Empty')
        texture2_label.grid(row=1, column=0, sticky='W')
        statuses[1].grid(row=1, column=1, sticky='W')

        open_image_btn2 = tk.Button(texture2_frame, text="Image", command=lambda: self.open(action2, 'image'))
        self.open_image_buttons += [open_image_btn2]
        open_tensor_btn2 = tk.Button(texture2_frame, text="Vector", command=lambda: self.open(action2, 'tensor'))
        open_image_btn2.grid(row=1, column=2)
        open_tensor_btn2.grid(row=1, column=3)

        alpha_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        tk.Label(alpha_frame, text='alpha:').grid(row=0, column=0)
        alpha_entry = tk.Entry(alpha_frame)
        alpha_entry.insert(tk.END, 0)
        alpha_entry.grid(row=0, column=1)

        compute_btn = tk.Button(tool_frame, text="Compute", state='disabled', command=lambda: compute(alpha_entry.get()))

        texture1_frame.grid(row=0, column=0)
        texture2_frame.grid(row=1, column=0)
        alpha_frame.grid(row=2, column=0, sticky='EW')
        compute_btn.grid(row=3, column=0, sticky='W')

        return tool_frame

    def extraction_tool(self, parent):
        def action(style_tensor, slot):
            assert slot in [0, 1]
            if style_tensor is not None:
                self.extraction_slots[slot] = style_tensor
                statuses[slot].config(text='Loaded')

                if None not in self.extraction_slots:
                    compute_btn.config(state='active')

        action1 = lambda style_tensor: action(style_tensor, 0)
        action2 = lambda style_tensor: action(style_tensor, 1)

        def compute():
            assert None not in self.extraction_slots
            s1, s2 = process_tensor_lists(self.extraction_slots[0], self.extraction_slots[1])
            if self.method_var.get() == 0:
                result_tensor, _ = style_attribute_extraction_means(s1, s2)
            elif self.method_var.get() == 1:
                result_tensor, _ = style_attribute_extraction_svm(s1, s2)
            self.update_out(result_tensor)

        
        tool_frame = tk.LabelFrame(parent, text='Extraction', padx=5, pady=5)
        statuses = [None, None]

        texture1_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture1_label = tk.Label(texture1_frame, text='Texture Set 1:')
        statuses[0] = tk.Label(texture1_frame, text='Empty')
        texture1_label.grid(row=0, column=0, sticky='W')
        statuses[0].grid(row=0, column=1, sticky='W')

        open_image_set_btn1 = tk.Button(texture1_frame, text="Images", command=lambda: self.open(action1, 'many'))
        self.open_image_buttons += [open_image_set_btn1]
        open_image_set_btn1.grid(row=0, column=2)

        texture2_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture2_label = tk.Label(texture2_frame, text='Image Set 2:')
        statuses[1] = tk.Label(texture2_frame, text='Empty')
        texture2_label.grid(row=1, column=0, sticky='W')
        statuses[1].grid(row=1, column=1, sticky='W')

        open_image_set_btn2 = tk.Button(texture2_frame, text="Images", command=lambda: self.open(action2, 'many'))
        self.open_image_buttons += [open_image_set_btn2]
        open_image_set_btn2.grid(row=1, column=2)

        self.method_var = tk.IntVar()
        self.method_var.set(0)
        R1 = tk.Radiobutton(tool_frame, text="Means", variable=self.method_var, value=0)
        R2 = tk.Radiobutton(tool_frame, text="SVM", variable=self.method_var, value=1)

        compute_btn = tk.Button(tool_frame, text="Compute", state='disabled', command=compute)

        texture1_frame.grid(row=0, column=0)
        texture2_frame.grid(row=1, column=0)
        R1.grid(row=2, column=0, sticky='W')
        R2.grid(row=3, column=0, sticky='W')
        compute_btn.grid(row=4, column=0, sticky='W')

        return tool_frame

    def linear_operation_tool(self, parent):
        def action(style_tensor, slot):
            assert slot in [0, 1]
            if style_tensor is not None:
                self.linear_slots[slot] = style_tensor
                statuses[slot].config(text='Loaded')

                if None not in self.linear_slots:
                    compute_btn.config(state='active')

        action1 = lambda style_tensor: action(style_tensor, 0)
        action2 = lambda style_tensor: action(style_tensor, 1)

        def compute(alpha):
            assert None not in self.linear_slots
            alpha = process_str_value(alpha, 0, float)
            t1, t2 = process_tensors(self.linear_slots[0], self.linear_slots[1])
            result_tensor = t1 + t2 * alpha
            self.update_out(result_tensor)

        
        tool_frame = tk.LabelFrame(parent, text='Linear Operations', padx=5, pady=5)
        statuses = [None, None]

        texture1_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture1_label = tk.Label(texture1_frame, text='Texture 1:')
        statuses[0] = tk.Label(texture1_frame, text='Empty')
        texture1_label.grid(row=0, column=0, sticky='W')
        statuses[0].grid(row=0, column=1, sticky='W')

        open_image_btn1 = tk.Button(texture1_frame, text="Image", command=lambda: self.open(action1, 'image'))
        self.open_image_buttons += [open_image_btn1]
        open_tensor_btn1 = tk.Button(texture1_frame, text="Vector", command=lambda: self.open(action1, 'tensor'))
        open_image_btn1.grid(row=0, column=2)
        open_tensor_btn1.grid(row=0, column=3)

        texture2_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture2_label = tk.Label(texture2_frame, text='Texture 2:')
        statuses[1] = tk.Label(texture2_frame, text='Empty')
        texture2_label.grid(row=1, column=0, sticky='W')
        statuses[1].grid(row=1, column=1, sticky='W')

        open_image_btn2 = tk.Button(texture2_frame, text="Image", command=lambda: self.open(action2, 'image'))
        self.open_image_buttons += [open_image_btn2]
        open_tensor_btn2 = tk.Button(texture2_frame, text="Vector", command=lambda: self.open(action2, 'tensor'))
        open_image_btn2.grid(row=1, column=2)
        open_tensor_btn2.grid(row=1, column=3)

        alpha_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        tk.Label(alpha_frame, text='alpha:').grid(row=0, column=0)
        alpha_entry = tk.Entry(alpha_frame)
        alpha_entry.insert(tk.END, 0)
        alpha_entry.grid(row=0, column=1)

        compute_btn = tk.Button(tool_frame, text="Compute", state='disabled', command=lambda: compute(alpha_entry.get()))

        texture1_frame.grid(row=0, column=0)
        texture2_frame.grid(row=1, column=0)
        alpha_frame.grid(row=2, column=0, sticky='EW')
        compute_btn.grid(row=3, column=0, columnspan=4, sticky='W')

        return tool_frame

    def simple_generation_tool(self, parent):
        def action(style_tensor):
            if style_tensor is not None:
                texture_status.config(text='Loaded')
                self.update_out(style_tensor)

        tool_frame = tk.LabelFrame(parent, text='Simple Synthesis', padx=5, pady=5)
        texture_label = tk.Label(tool_frame, text='Texture:')
        texture_status = tk.Label(tool_frame, text='Empty')
        open_image_btn = tk.Button(tool_frame, text="Image", command=lambda: self.open(action, 'image'))
        self.open_image_buttons += [open_image_btn]
        open_tensor_btn = tk.Button(tool_frame, text="Vector", command=lambda: self.open(action, 'tensor'))

        texture_label.grid(row=0, column=0)
        texture_status.grid(row=0, column=1)
        open_image_btn.grid(row=0, column=2)
        open_tensor_btn.grid(row=0, column=3)

        return tool_frame

    def generate(self):
        self.generate_btn.config(state='disabled')
        self.generate_dialog_window = GenerateDialog(self)
        self.window.wait_window(self.generate_dialog_window)
        self.generate_dialog_window = None
        try:
            self.generate_btn.config(state='active')
        except Exception:
            pass

    def save_tensor(self):
        try:
            f = tk.filedialog.asksaveasfile(
                title='Save tensor',
                mode='a', 
                defaultextension='.pth', 
                initialdir='./'
            )
            if f is None:
                return
            torch.save(self.out, f.name)
            f.close()
        except Exception:
            unable_to_save_tensor()

    def open(self, action, mode='image'):
        if mode == 'image':
            style_tensor = self.open_image()
        elif mode == 'tensor':
            style_tensor = self.open_tensor()
        elif mode == 'many':
            style_tensor = self.open_images()
        action(style_tensor)

    def open_image(self):
        # self.disable_open_image()
        dialog_window = OpenImageDialog(self)
        style_tensor = dialog_window.get_result()
        if style_tensor is not None:
            style_tensor = style_tensor[0]
        # self.window.wait_window(dialog_window)
        # self.enable_open_image()
        return style_tensor

    def open_images(self):
        # self.disable_open_image()
        dialog_window = OpenImageDialog(self)
        style_tensors = dialog_window.get_result()
        # self.window.wait_window(dialog_window)
        # self.enable_open_image()
        return style_tensors
    
    def open_tensor(self):
        filetypes = [('torch tensors', ('.pth'))]
        path = tk.filedialog.askopenfilename(
            title='Open tensor',
            initialdir='./',
            filetypes=filetypes)
        if path == '':
            return

        try:
            style_tensor = torch.load(path)
        except Exception:
            unable_to_open_tensor()

        return style_tensor

    def update_preview(self, generated_texture):
        self.preview_img = ImageTk.PhotoImage(generated_texture)
        self.image_label.config(image=self.preview_img)

    def generate_preview(self):
        self.pb_label.config(text='Generating...')
        self.pb.start()
        self.disable_buttons()
        generator_thread = Generator(self.G, self.out, self.preview_size, 0, 1)
        generator_thread.start()

        self.monitor_preview_generation(generator_thread)

    def monitor_preview_generation(self, generator_thread):
        if generator_thread.is_alive():
            self.window.after(100, lambda: self.monitor_preview_generation(generator_thread))
        else:
            self.pb.stop()
            if generator_thread.cuda_oom:
                self.pb_label.config(text='CUDA out of memory!')
                cuda_oom_warning()
            else:
                self.pb_label.config(text='Done!')
            self.update_preview(generator_thread.generated_texture)
            self.activate_buttons()

    def mainloop(self):
        self.window.mainloop()


def main():
    window = Window()
    window.mainloop()


if __name__ == "__main__":
    main()