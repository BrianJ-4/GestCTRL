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
        self.add_icon = tk.PhotoImage(file='./assets/icons/add.png')
        self.train_icon = tk.PhotoImage(file='./assets/icons/train.png')
        self.settings_icon = tk.PhotoImage(file='./assets/icons/settings.png')
        self.setup_ui()

    def set_geometry(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def setup_ui(self):
        # Webcam Preview --------------------------------------------------
        self.preview_frame = ttk.Frame(self.root)
        self.preview_frame.place(x = 20, y = 20, width = 580, height = 360)
        self.preview_label = tk.Label(
            self.preview_frame,
            bg = "black",
            text = "Waiting for video frame...",
            fg = "white",
            font = ("Arial", 12)
        )
        self.preview_label.pack(expand = True, fill = "both")

        # Poses List -----------------------------------------------------
        self.pose_frame = ttk.Frame(self.root)
        self.pose_frame.place(x = 620, y = 15, width = 260, height = 300)

        self.pose_tree = ttk.Treeview(self.pose_frame, columns = ("Action",), show = "headings", height = 15)
        self.pose_tree.heading("#1", text = "Action")
        self.pose_tree["columns"] = ("Pose", "Action")
        self.pose_tree.column("Pose", width = 100, anchor = "w")
        self.pose_tree.column("Action", width = 140, anchor = "w")
        self.pose_tree.heading("Pose", text = "Pose")
        self.pose_tree.heading("Action", text = "Action")
        self.pose_tree.pack(fill = "both", expand = True, padx = 5, pady = 5)

        scrollbar = ttk.Scrollbar(self.pose_frame, orient = "vertical", command = self.pose_tree.yview)
        self.pose_tree.configure(yscrollcommand = scrollbar.set)
        scrollbar.place(relx = 1, rely = 0, relheight = 1, anchor = "ne")

        # Sample data (for testing)
        sample_data = [
            ("Open", "Mouse Mode"),
            ("Closed", "Neutral"),
            ("Peace", "Mouse Mode"),
            ("Thumbs Up", "Left Click"),
            ("RockNRoll", "Refresh"),
        ]
        for pose, action in sample_data:
            self.pose_tree.insert("", "end", values = (pose, action))

        # Add Pose Button ------------------------------------------------
        self.add_pose_button = ttk.Button(
            self.root,
            image = self.add_icon,
            text = "Add Pose",
            compound = tk.LEFT
        )
        self.add_pose_button.place(x = 625, y = 340, width = 120, height = 40)

        # Train Model Button ---------------------------------------------
        self.train_button = ttk.Button(
            self.root,
            image = self.train_icon,
            text = "Train Model",
            compound = tk.LEFT,
            style = "Accent.TButton"
        )
        self.train_button.place(x = 750, y = 340, width = 120, height = 40)

        # Settings Button ------------------------------------------------
        self.settings_button = ttk.Button(
            self.root,
            image = self.settings_icon,
        )
        self.settings_button.place(x = 840, y = 550)

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()