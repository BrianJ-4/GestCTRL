import tkinter as tk
from tkinter import ttk
import sv_ttk

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GestCTRL")
        self.set_geometry(900, 600)
        self.root.resizable(False, False)
        sv_ttk.set_theme("dark")

    def set_geometry(self, width, height):
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()