import tkinter as tk

def cuda_oom_warning():
    tk.messagebox.showwarning(
        title='Warning', 
        message='Out of memory. Results may differ from expected. Try to reduce image size.'
    )

def unable_to_save_image():
    tk.messagebox.showerror(
        title="Error", 
        message="Unable to save image"
    )

def unable_to_save_tensor():
    tk.messagebox.showerror(
        title="Error", 
        message="Unable to save vector"
    )

def unable_to_open_tensor():
    tk.messagebox.showerror(
        title="Error", 
        message="Unable to open tensor"
    )

def invalid_alpha():
    tk.messagebox.showerror(
        title="Error", 
        message="Alpha must be within interval: [0, 1]"
    )

def incompatible_tensors():
    tk.messagebox.showwarning(
        title='Warning', 
        message='Different scales of images. Results may differ from expected.'
    )