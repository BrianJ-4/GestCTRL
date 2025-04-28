import pyautogui
import time

class ActionController:
    def __init__(self):
        self.actions = {
            "Copy": self.copy,
            "Paste": self.paste,
            "Refresh": self.refresh,
            "Zoom In": self.zoom_in,
            "Zoom Out": self.zoom_out,
        }

    def perform_action(self, action_name):
        action = self.actions.get(action_name)
        action()

    def copy(self):
        pyautogui.hotkey("ctrl", "c")

    def paste(self):
        pyautogui.hotkey("ctrl", "v")

    def refresh(self):
        pyautogui.press("f5")

    def zoom_in(self):
        pyautogui.hotkey("ctrl", "+")

    def zoom_out(self):
        pyautogui.hotkey("ctrl", "-")

    def get_actions(self):
        return list(self.actions.keys())