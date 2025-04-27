import cv2
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image
import sv_ttk
import threading

from gesture_manager import GestureManager
from model_trainer import train_model
from pose_action_manager import PoseActionManager
from pose_recorder import GestureRecorder

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GestCTRL")
        self.set_geometry(self.root, 900, 600)
        self.root.resizable(False, False)
        self.root.config(bg="#3b3b3b")
        self.gesture_manager = GestureManager()
        self.pose_action_manager = PoseActionManager()
        self.pose_recorder = GestureRecorder
        sv_ttk.set_theme("dark")
        self.start_ui()

    def set_geometry(self, parent, width, height):
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        parent.geometry(f"{width}x{height}+{x}+{y}")

    def start_ui(self):
        #Pose List and Thumbnail
        self.pose_frame = tk.Frame(self.root, bg="#636363", width=300, height=300)
        self.pose_frame.pack(side="right", fill="both", padx = 5, pady = 5)
        self.pose_frame.pack_propagate(False)
        cols = ("pose", "action")
        self.pose_tree_frame = tk.Frame(self.pose_frame, bg="#636363")
        self.pose_tree_frame.pack(fill="both", expand="True")
        self.pose_tree = ttk.Treeview(self.pose_tree_frame, columns= cols, show="headings")
        self.pose_tree.column("pose", anchor="center", width=100)
        self.pose_tree.heading("pose", text="Pose")
        self.pose_tree.column("action", anchor="center", width=100)
        self.pose_tree.heading("action", text="Action")
        self.pose_tree.pack(side="left", fill="both", expand=True)
        self.button_row_frame = tk.Frame(self.pose_frame, bg="#636363")
        self.button_row_frame.pack(fill="both", pady=10)
        self.pose_tree_scrollbar = ttk.Scrollbar(self.pose_tree_frame, orient="vertical", command=self.pose_tree.yview)
        self.pose_tree.configure(yscrollcommand = self.pose_tree_scrollbar.set)
        self.pose_tree_scrollbar.pack(side="right", fill="y")

        #Preview Frame (Here as an example)
        self.preview_frame = ttk.Frame(self.root, width=650, height=450)
        self.preview_frame.pack(side="left", anchor="n", fill="x", padx = 10, pady = 10)
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_frame)
        self.preview_label.pack(side="top", anchor="n", fill="x", expand=True)

        #Add Pose Button
        self.add_pose_button = ttk.Button(self.button_row_frame, text="Add Pose", command=self.add_pose_ui)
        self.add_pose_button.pack(side="left", padx=(50, 10))

        #Train Button
        self.train_button = ttk.Button(self.button_row_frame, text="Train", style="Accent.TButton", command = self.train_model_clicked )
        self.train_button.pack(side="right", padx=(10, 60))

        #Settings Button
        self.settings_button = ttk.Button(self.pose_frame, text="Settings", command=self.settings_ui)
        self.settings_button.pack(pady=10)

        self.updateList()

    def settings_ui(self):
            self.settings_window = Toplevel(self.root)
            self.settings_window.title("Settings")
            self.set_geometry(self.settings_window, 900, 600)
            self.settings_window.resizable(False, False)
            self.settings_window.config(bg="#3b3b3b")

    def train_model_clicked(self):
        def train_thread():
            self.train_button.config(state = "disabled", text = "Training...")
            print("Training started")
            train_model()
            self.train_button.config(state = "normal", text = "Train")
            print("Training complete")
        threading.Thread(target = train_thread, daemon = True).start()

    # Update the tree view with pose and action mappings
    def updateList(self):
        self.pose_tree.delete(*self.pose_tree.get_children())
        mappings = self.pose_action_manager.get_mappings()
        for pose, action in mappings.items():
            self.pose_tree.insert("", "end", values = (pose, action))

    def add_pose_ui(self):
        self.add_pose_window = Toplevel(self.root)
        self.add_pose_window.title("Add Pose")
        self.set_geometry(self.add_pose_window, 900, 600)
        self.add_pose_window.resizable(False, False)
        self.add_pose_window.config(bg="#3b3b3b")

        #Preview Frame
        self.add_pose_left_frame = ttk.Frame(self.add_pose_window, width=650, height=500)
        self.add_pose_left_frame.pack(side="left", anchor="n", fill="x", pady = 10, padx=10)
        self.add_pose_left_frame.pack_propagate(False)
        self.add_pose_preview_label = tk.Label(self.add_pose_left_frame)
        self.add_pose_preview_label.pack(fill="both", expand=True)

        #Pose List and Add Name
        self.add_pose_right_frame = tk.Frame(self.add_pose_window, bg="#636363", width=300, height=300)
        self.add_pose_right_frame.pack(side="right", fill="both", padx = 5, pady = 5)
        self.add_pose_right_frame.pack_propagate(False)
        col = ("pose")
        self.add_pose_tree = ttk.Treeview(self.add_pose_right_frame, columns= col, show="headings")
        self.add_pose_tree.column("pose", anchor="center", width=100)
        self.add_pose_tree.heading("pose", text="Pose")
        self.add_pose_tree.pack(anchor="n", fill="both", expand=True)
        self.updateListPoses()

        #Pose Name Entry
        self.add_pose_name = StringVar()
        self.add_pose_entry = tk.Entry(self.add_pose_right_frame, textvariable=self.add_pose_name, justify="center", font=("Arial, 14"), fg="#A9A9AC")
        self.add_pose_entry.insert(0, "Insert Pose Name")
        self.add_pose_entry.pack(pady=10)
        self.add_pose_entry.bind("<FocusIn>", self.entryFieldPlaceholder)

        #Add Pose Button
        self.add_pose_record_button = ttk.Button(self.add_pose_right_frame, text="Record Pose", command=self.record_button_clicked)
        self.add_pose_record_button.pack(pady=(5, 10))

    def updateListPoses(self):
        self.add_pose_tree.delete(*self.add_pose_tree.get_children())
        poses = self.gesture_manager.get_all_poses()
        for pose in poses:
            self.add_pose_tree.insert("", "end", values = pose)
    
    def entryFieldPlaceholder(self, event):
        if self.add_pose_entry.get() == "Insert Pose Name":
            self.add_pose_entry.delete(0, "end")
            self.add_pose_entry.config(fg="#FFFFFF")

    def record_button_clicked(self):
        recorded_pose = self.add_pose_name.get().strip()
        if not recorded_pose or recorded_pose == "Insert Pose Name":
            messagebox.showwarning("showwarning", "Please Enter a Name", parent=self.add_pose_window)
            return
        self.gesture_manager.add_pose(recorded_pose)
        self.updateListPoses()     

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()