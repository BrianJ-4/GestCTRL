import cv2
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image
import sv_ttk
import threading
import mediapipe as mp

from gesture_manager import GestureManager
from model_trainer import train_model
from pose_action_manager import PoseActionManager
from pose_recorder import GestureRecorder
from action_controller import ActionController

class GestureApp:
    def __init__(self, root, gesture_controller, camera_manager):
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
        self.gesture_controller = gesture_controller
        self.camera_manager = camera_manager

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
        self.pose_tree.bind("<Double-1>", self.open_pose_menu)

        # Preview Frame
        self.preview_frame = ttk.Frame(self.root, width=650, height=450)
        self.preview_frame.pack_propagate(False)
        self.preview_frame.pack(side="left", anchor="n", padx=10, pady=10)
        self.preview_label = tk.Label(self.preview_frame, bg="black")
        self.preview_label.pack(fill="both", expand=True)

        #Add Pose Button
        self.add_pose_button = ttk.Button(self.button_row_frame, text="Add Pose", command=self.add_pose_ui)
        self.add_pose_button.pack(side="left", padx=(50, 10))

        #Train Button
        self.train_button = ttk.Button(self.button_row_frame, text="Train", style="Accent.TButton", state = "disabled", command = self.train_model_clicked )
        self.train_button.pack(side="right", padx=(10, 60))

        #Settings Button
        self.settings_button = ttk.Button(self.pose_frame, text="Settings", command=self.settings_ui)
        self.settings_button.pack(pady=10)

        self.updateList()
        self.root.after_idle(self.start_camera_loop)

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
            self.gesture_controller.reload_model() # Reload to use the newly trained model
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
        new_pose = self.add_pose_name.get().strip()
        if not new_pose or new_pose == "Insert Pose Name":
            messagebox.showwarning("Warning", "Please Enter a Name", parent=self.add_pose_window)
            return
        
        self.add_pose_record_button.config(state="disabled", text="Recording...")
        self.pose_recorder = GestureRecorder()
        self.pose_recorder.start_recording(new_pose)
        self.recording_thread = threading.Thread(target=self.pose_recorder.run, daemon=True)
        self.recording_thread.start()

        self.add_pose_window.bind("<Return>", self.on_enter_pressed)
        self.add_pose_window.bind("<Escape>", self.on_escape_pressed)
        
    def on_enter_pressed(self, event=None):
        if self.pose_recorder and self.pose_recorder.running:
            self.pose_recorder.record_frame()
            print("Frame saved.")

    def on_escape_pressed(self, event=None):
        if self.pose_recorder and self.pose_recorder.running:
            self.pose_recorder.stop_recording()
            self.pose_recorder.stop()
            self.add_pose_window.after(500, lambda: self.recording_finished(self.pose_recorder))
            self.add_pose_window.unbind("<Return>")
            self.add_pose_window.unbind("<Escape>")
        
    def recording_finished(self):
        self.pose_recorder.stop()

        #Changing Button Back
        self.add_pose_record_button.config(state="normal", text="Record Pose")    
        self.changed = True
        self.update_train_button()
        self.updateListPoses()
        self.updateList()

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

    def start_camera_loop(self):
        self.update_camera_preview()

    def update_camera_preview(self):
        frame = self.camera_manager.get_frame()
        if frame is not None:
            frame = frame.copy()
            frame = cv2.flip(frame, 1)

            hands = self.gesture_controller.hands
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            width = self.preview_label.winfo_width()
            height = self.preview_label.winfo_height()
            if width > 10 and height > 10:
                img_resized = img.resize((width, height))
                self.live_image = ImageTk.PhotoImage(img_resized)
                self.preview_label.config(image = self.live_image)

                if hasattr(self, 'add_pose_preview_label') and self.add_pose_preview_label.winfo_exists():
                    width = self.add_pose_preview_label.winfo_width()
                    height = self.add_pose_preview_label.winfo_height()
                    img_resized_add = img.resize((width, height))
                    self.live_image_add = ImageTk.PhotoImage(img_resized_add)
                    self.add_pose_preview_label.config(image=self.live_image_add)

        self.root.after(15, self.update_camera_preview)