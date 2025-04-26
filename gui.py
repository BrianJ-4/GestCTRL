import cv2
import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import sv_ttk
import threading

from model_trainer import train_model
from pose_action_manager import PoseActionManager
from action_controller import ActionController
from gesture_manager import GestureManager

class GestureApp:
    def __init__(self, root):
        # Window setup
        self.root = root
        self.root.title("GestCTRL")
        self.set_geometry(self.root, 900, 600)
        self.root.resizable(False, False)
        self.root.config(bg="#3b3b3b")

        # Create objects to interact with backend data
        self.pose_action_manager = PoseActionManager()
        self.action_controller = ActionController()
        self.gesture_manager = GestureManager()

        # Set theme
        sv_ttk.set_theme("dark")

        # Load icons
        self.save_icon = tk.PhotoImage(file = './assets/icons/save.png')
        self.delete_icon = tk.PhotoImage(file = './assets/icons/delete.png')

        # Used to track when the model should be retrained based on user adding/deleting poses
        self.changed = False

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
        self.pose_frame.pack(side="left", fill="both", padx = 5, pady = 5)
        self.pose_frame.pack_propagate(False)
        cols = ("pose", "action")
        self.pose_tree = ttk.Treeview(self.pose_frame, columns= cols, show="headings")
        self.pose_tree.column("pose", anchor="center", width=100)
        self.pose_tree.heading("pose", text="Pose")
        self.pose_tree.column("action", anchor="center", width=100)
        self.pose_tree.heading("action", text="Action")
        self.pose_tree.bind("<Double-1>", self.open_pose_menu)
        self.pose_tree.pack(fill="both", expand=True)

        #Preview Frame (Here as an example)
        self.preview_frame = ttk.Frame(self.root, width=650, height=300)
        self.preview_frame.pack(side="right", fill="both", padx = 10, pady = 10)
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_frame)
        self.preview_label.pack(fill="both", expand=True)

        #Add Pose Button
        self.add_pose_button = ttk.Button(self.pose_frame, text="Add Pose")
        self.add_pose_button.pack(pady=10)

        #Train Button
        self.train_button = ttk.Button(self.pose_frame, text="Train", style="Accent.TButton", state = "disabled", command = self.train_model_clicked )
        self.train_button.pack(pady=10)

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
            self.changed = False
            self.update_train_button()
            print("Training complete")
        threading.Thread(target = train_thread, daemon = True).start()

    def update_train_button(self):
        if self.changed:
            self.train_button.config(state = "enabled", text = "Train")
        else:
            self.train_button.config(state = "disabled", text = "Train")

    # Update the tree view with pose and action mappings
    def updateList(self):
        self.pose_tree.delete(*self.pose_tree.get_children())
        mappings = self.pose_action_manager.get_mappings()
        for pose, action in mappings.items():
            self.pose_tree.insert("", "end", values = (pose, action))

    def open_pose_menu(self, event):
        selected_item = self.pose_tree.focus()
        pose, action = self.pose_tree.item(selected_item, "values")

        menu = Toplevel(self.root)
        menu.title(f"Edit Pose: {pose}")
        self.set_geometry(menu, 400, 250)
        menu.resizable(False, False)
        menu.config(bg = "#3b3b3b")

        # Pose name label
        pose_label = ttk.Label(menu, text = pose, font = ("Arial", 14), background = "#3b3b3b")
        pose_label.pack(pady = 20)

        # Dropdown for actions
        actions = self.action_controller.get_actions() + ["Right Click", "Left Click", "Mouse Mode", "Neutral"]
        action = tk.StringVar(value = action)
        action_dropdown = ttk.Combobox(menu, textvariable = action, values = actions, state = "readonly")
        action_dropdown.pack(pady = 10)

        # Save Button
        save_button = ttk.Button(
            menu,
            image = self.save_icon,
            text = "Save Mapping",
            compound = tk.LEFT,
            command = lambda: self.save_action(pose, action.get(), menu),
            style = "Accent.TButton"
        )
        save_button.pack(pady = 10)

        # Delete Button
        delete_button = ttk.Button(
            menu,
            image = self.delete_icon,
            text = "Delete Pose",
            compound = tk.LEFT,
            command=lambda: self.delete_pose(pose, menu)
        )
        delete_button.pack(pady = 10)

    def save_action(self, pose, action, window):
        self.pose_action_manager.set_pose_action(pose, action)
        self.updateList()
        window.destroy()

    def delete_pose(self, pose, window):
        self.gesture_manager.delete_pose(pose)
        self.updateList()
        self.changed = True
        self.update_train_button()
        window.destroy()