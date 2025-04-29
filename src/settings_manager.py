import sys
import os
from utils import resource_path

class SettingsManager:
    def __init__(self):
        self.webcam_index = self.get_webcam_index_setting()
        self.display_help = self.get_display_help_setting()

    def get_webcam_index_setting(self):
        with open(resource_path("data/settings.txt"), "r") as file:
            index = int(file.readline().strip())
            return index
        
    def get_display_help_setting(self):
        with open(resource_path("data/settings.txt"), "r") as file:
            value = file.readlines()[1].strip()
            return value.lower() == "true"
            
    def set_webcam_index_setting(self, index):
        with open(resource_path("data/settings.txt"), "r") as file:
            lines = file.readlines()
            lines[0] = f"{index}\n"
        with open(resource_path("data/settings.txt"), "w") as file:
            file.writelines(lines)
        self.webcam_index = index

    def set_display_help_setting(self, value):
        with open(resource_path("data/settings.txt"), "r") as file:
            lines = file.readlines()
            lines[1] = f"{value}"
        with open(resource_path("data/settings.txt"), "w") as file:
            file.writelines(lines)
        self.display_help = value