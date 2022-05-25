import tkinter as tk
import tkinter.ttk as ttk
from PIL import ImageTk, Image
import torch

from src.texture_synthesis.feature_extractor import FeatureExtractor
from src.texture_synthesis.gen import SlowGen, FastGen2
from src.texture_synthesis.adain_autoencoder import AdaINAutoencoder
from src.texture_synthesis.style_features_manipulation import style_attribute_extraction_means, style_attribute_extraction_svm

from async_opeations import PreviewGenerator
from generate import GenerateDialog
from open_files import OpenImageDialog
from utils import process_str_value


class Window():
    def __init__(self):

        self.font = ("Courier", 12)
        self.window = tk.Tk()
        self.window.title("Texture Generator")
        # self.window.configure(bg='white')
        # self.window.geometry("600x900")
        self.window.resizable(0, 0)

        self.objects_dict = {}

        self._init_models()
        self.show_interface()
        self.out = None

        self.interpolation_slots = [None, None]
        self.linear_slots = [None, None]
        self.extraction_slots = [None, None]

    def _init_models(self):
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')
        self.FE = FeatureExtractor().to(self.device)
        self.SG = SlowGen(self.FE, self.device)
        self.AA = AdaINAutoencoder().to(self.device)
        self.FG = FastGen2(self.FE, self.AA, self.device)

    def show_interface(self):
        # Output
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
        output_panel_frame = tk.LabelFrame(parent, text='Output', padx=5, pady=5)
        
        pil_image = Image.new("RGB", (256, 256), (255, 255, 255))
        self.preview_img = ImageTk.PhotoImage(pil_image)
        
        self.image_label = tk.Label(output_panel_frame, image = self.preview_img)
        self.image_label.grid(row=0, column=0, padx=5, pady=5)

        self.generate_preview_btn = tk.Button(output_panel_frame, text="Generate Preview", command=self.generate_preview, state='disable')
        self.generate_preview_btn.grid(row=1, column=0, sticky='EW')

        tk.Label(output_panel_frame, height=1).grid(row=2, column=0)

        self.save_tensor_btn = tk.Button(output_panel_frame, text="Save Tensor", command=self.save_tensor, state='disable')
        self.save_tensor_btn.grid(row=3, column=0, sticky='EW')

        self.generate_btn = tk.Button(output_panel_frame, text="Generate", command=self.generate, state='active')
        self.generate_btn.grid(row=4, column=0, sticky='EW')

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

                if not None in self.interpolation_slots:
                    compute_btn.config(state='active')
            else:
                print("why")

        action1 = lambda style_tensor: action(style_tensor, 0)
        action2 = lambda style_tensor: action(style_tensor, 1)

        def compute(alpha):
            assert not None in self.interpolation_slots
            alpha = process_str_value(alpha, 0, float)
            if not 0 <= alpha <= 1:
                tk.messagebox.showerror("Exception", "alpha must be in [0, 1]")
                return

            result_tensor = self.interpolation_slots[0] * (1 - alpha) + self.interpolation_slots[1] * alpha
            self.update_out(result_tensor)

        
        tool_frame = tk.LabelFrame(parent, text='Interpolation', padx=5, pady=5)
        statuses = [None, None]

        texture1_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture1_label = tk.Label(texture1_frame, text='Texture 1:')
        statuses[0] = tk.Label(texture1_frame, text='Empty')
        texture1_label.grid(row=0, column=0, sticky='W')
        statuses[0].grid(row=0, column=1, sticky='W')

        open_image_btn1 = tk.Button(texture1_frame, text="Open Image", command=lambda: self.open(action1, 'image'))
        open_tensor_btn1 = tk.Button(texture1_frame, text="Open Tensor", command=lambda: self.open(action1, 'tensor'))
        open_image_btn1.grid(row=0, column=2)
        open_tensor_btn1.grid(row=0, column=3)

        texture2_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture2_label = tk.Label(texture2_frame, text='Texture 2:')
        statuses[1] = tk.Label(texture2_frame, text='Empty')
        texture2_label.grid(row=1, column=0, sticky='W')
        statuses[1].grid(row=1, column=1, sticky='W')

        open_image_btn2 = tk.Button(texture2_frame, text="Open Image", command=lambda: self.open(action2, 'image'))
        open_tensor_btn2 = tk.Button(texture2_frame, text="Open Tensor", command=lambda: self.open(action2, 'tensor'))
        open_image_btn2.grid(row=1, column=2)
        open_tensor_btn2.grid(row=1, column=3)

        alpha_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        tk.Label(alpha_frame, text='alpha:').grid(row=0, column=0)
        alpha_entry = tk.Entry(alpha_frame)
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

                if not None in self.extraction_slots:
                    compute_btn.config(state='active')
            else:
                print("why")

        action1 = lambda style_tensor: action(style_tensor, 0)
        action2 = lambda style_tensor: action(style_tensor, 1)

        def compute():
            assert not None in self.extraction_slots
            result_tensor, _ = style_attribute_extraction_svm(self.extraction_slots[0], self.extraction_slots[1])
            self.update_out(result_tensor)

        
        tool_frame = tk.LabelFrame(parent, text='Extraction', padx=5, pady=5)
        statuses = [None, None]

        texture1_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture1_label = tk.Label(texture1_frame, text='Texture 1:')
        statuses[0] = tk.Label(texture1_frame, text='Empty')
        texture1_label.grid(row=0, column=0, sticky='W')
        statuses[0].grid(row=0, column=1, sticky='W')

        open_image_set_btn1 = tk.Button(texture1_frame, text="Open Images", command=lambda: self.open(action1, 'many'))
        open_image_set_btn1.grid(row=0, column=2)

        texture2_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture2_label = tk.Label(texture2_frame, text='Texture 2:')
        statuses[1] = tk.Label(texture2_frame, text='Empty')
        texture2_label.grid(row=1, column=0, sticky='W')
        statuses[1].grid(row=1, column=1, sticky='W')

        open_image_set_btn2 = tk.Button(texture2_frame, text="Open Images", command=lambda: self.open(action2, 'many'))
        open_image_set_btn2.grid(row=1, column=2)

        compute_btn = tk.Button(tool_frame, text="Compute", state='disabled', command=compute)

        texture1_frame.grid(row=0, column=0)
        texture2_frame.grid(row=1, column=0)
        compute_btn.grid(row=2, column=0, sticky='W')

        return tool_frame

    def linear_operation_tool(self, parent):
        def action(style_tensor, slot):
            assert slot in [0, 1]
            if style_tensor is not None:
                self.linear_slots[slot] = style_tensor
                statuses[slot].config(text='Loaded')

                if not None in self.linear_slots:
                    compute_btn.config(state='active')
            else:
                print("why")

        action1 = lambda style_tensor: action(style_tensor, 0)
        action2 = lambda style_tensor: action(style_tensor, 1)

        def compute(alpha):
            assert not None in self.linear_slots
            alpha = process_str_value(alpha, 0, float)

            result_tensor = self.linear_slots[0] + self.linear_slots[1] * alpha
            self.update_out(result_tensor)

        
        tool_frame = tk.LabelFrame(parent, text='Linear Operations', padx=5, pady=5)
        statuses = [None, None]

        texture1_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture1_label = tk.Label(texture1_frame, text='Texture 1:')
        statuses[0] = tk.Label(texture1_frame, text='Empty')
        texture1_label.grid(row=0, column=0, sticky='W')
        statuses[0].grid(row=0, column=1, sticky='W')

        open_image_btn1 = tk.Button(texture1_frame, text="Open Image", command=lambda: self.open(action1, 'image'))
        open_tensor_btn1 = tk.Button(texture1_frame, text="Open Tensor", command=lambda: self.open(action1, 'tensor'))
        open_image_btn1.grid(row=0, column=2)
        open_tensor_btn1.grid(row=0, column=3)

        texture2_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        texture2_label = tk.Label(texture2_frame, text='Texture 2:')
        statuses[1] = tk.Label(texture2_frame, text='Empty')
        texture2_label.grid(row=1, column=0, sticky='W')
        statuses[1].grid(row=1, column=1, sticky='W')

        open_image_btn2 = tk.Button(texture2_frame, text="Open Image", command=lambda: self.open(action2, 'image'))
        open_tensor_btn2 = tk.Button(texture2_frame, text="Open Tensor", command=lambda: self.open(action2, 'tensor'))
        open_image_btn2.grid(row=1, column=2)
        open_tensor_btn2.grid(row=1, column=3)

        alpha_frame = tk.Frame(tool_frame, relief='flat', borderwidth=0, pady=5)
        tk.Label(alpha_frame, text='alpha:').grid(row=0, column=0)
        alpha_entry = tk.Entry(alpha_frame)
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
            else:
                print("why")

        tool_frame = tk.LabelFrame(parent, text='Simple Generation', padx=5, pady=5)
        texture_label = tk.Label(tool_frame, text='Texture:')
        texture_status = tk.Label(tool_frame, text='Empty')
        open_image_btn = tk.Button(tool_frame, text="Open Image", command=lambda: self.open(action, 'image'))
        open_tensor_btn = tk.Button(tool_frame, text="Open Tensor", command=lambda: self.open(action, 'tensor'))

        texture_label.grid(row=0, column=0)
        texture_status.grid(row=0, column=1)
        open_image_btn.grid(row=0, column=2)
        open_tensor_btn.grid(row=0, column=3)

        return tool_frame

    def generate(self):
        dialog_window = GenerateDialog(self)

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
            tk.messagebox.showerror("Exception", "Unable to save tensor")

    def open(self, action, mode='image'):
        if mode == 'image':
            style_tensor = self.open_image()
        elif mode == 'tensor':
            style_tensor = self.open_tensor()
        elif mode == 'many':
            style_tensor = self.open_images()
        action(style_tensor)

    def open_image(self):
        dialog_window = OpenImageDialog(self)
        style_tensor = dialog_window.get_result()
        if style_tensor is not None:
            style_tensor = style_tensor[0]
        return style_tensor

    def open_images(self):
        dialog_window = OpenImageDialog(self)
        style_tensors = dialog_window.get_result()
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
            tk.messagebox.showerror("Exception", "Unable to open tensor")

        return style_tensor

    def update_preview(self, generated_texture):
        self.preview_img = ImageTk.PhotoImage(generated_texture)
        self.image_label.config(image=self.preview_img)

    def generate_preview(self):
        self.pb_label.config(text='Generating...')
        self.pb.start()
        generator_thread = PreviewGenerator(self.FG, self.out)
        generator_thread.start()

        self.monitor_preview_generation(generator_thread)

    def monitor_preview_generation(self, generator_thread):
        if generator_thread.is_alive():
            self.window.after(100, lambda: self.monitor_preview_generation(generator_thread))
        else:
            self.pb.stop()
            self.pb_label.config(text='Done!')
            self.update_preview(generator_thread.generated_texture)

    def mainloop(self):
        self.window.mainloop()


def main():
    window = Window()
    window.mainloop()


if __name__ == "__main__":
    main()