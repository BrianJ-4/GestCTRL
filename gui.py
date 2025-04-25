import cv2
import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GestCTRL")
        self.set_geometry(self.root, 900, 600)
        self.root.resizable(False, False)
        self.root.config(bg="#3b3b3b")
        self.logo = ImageTk.PhotoImage(Image.open('assets/icons/logo_nobg1.png').resize((300, 100)))
        self.start_ui()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam unable to open")
        self.camera_preview()

    def set_geometry(self, parent, width, height):
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        parent.geometry(f"{width}x{height}+{x}+{y}")

    def start_ui(self):
        #Pose List and Thumbnail
        self.pose_frame = tk.Frame(self.root, bg="#636363", width=300, height=300)
        self.pose_frame.pack(side="left", fill="both", padx = 5, pady = 5)
        self.pose_frame.pack_propagate(False)
        self.logo_label = tk.Label(self.pose_frame, image=self.logo, bg="#636363")
        self.logo_label.pack()
        cols = ("pose", "action")
        self.pose_tree = ttk.Treeview(self.pose_frame, columns= cols, show="headings")
        self.pose_tree.column("pose", anchor="center", width=100)
        self.pose_tree.heading("pose", text="Pose")
        self.pose_tree.column("action", anchor="center", width=100)
        self.pose_tree.heading("action", text="Action")
        self.pose_tree.pack(fill="both", expand=True)

        #Preview Frame (Here as an example)
        self.preview_frame = ttk.Frame(self.root, width=650, height=300)
        self.preview_frame.pack(side="right", fill="both", padx = 10, pady = 10)
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_frame)
        self.preview_label.pack(fill="both", expand=True)

        #Add Pose Button
        self.pose_button = ttk.Button(self.pose_frame, text="Add Pose")
        self.pose_button.pack(pady=10)

        #Train Button
        self.pose_button = ttk.Button(self.pose_frame, text="Train")
        self.pose_button.pack(pady=10)

        #Settings Button
        self.pose_button = ttk.Button(self.pose_frame, text="Settings", command=self.settings_ui)
        self.pose_button.pack(pady=10)

    def settings_ui(self):
        self.settings_window = Toplevel(self.root)
        self.settings_window.title("Settings")
        self.set_geometry(self.settings_window, 900, 600)
        self.settings_window.resizable(False, False)
        self.settings_window.config(bg="#3b3b3b")

    def camera_preview(self):
            success, image = self.cap.read()
            if not success:
                return
            rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            width = self.preview_frame.winfo_width()
            height = self.preview_frame.winfo_height()
            results = rgb.resize((width, height))
            self.live_camera = ImageTk.PhotoImage(results)
            self.preview_label.config(image=self.live_camera)
            self.root.after(15, self.camera_preview)
        
    def close(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
    app.close()