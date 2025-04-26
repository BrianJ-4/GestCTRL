import json

class PoseActionManager():
    def __init__(self):
        self.mappings = self.load_mappings()

    def load_mappings(self):
        try:
            with open("data/mappings.json", "r") as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print("Warning: mappings.json not found. Using empty mappings dictionary.")
            return {}
        
    def save_mappings(self):
        with open("data/mappings.json", "w") as file:
            json.dump(self.mappings, file)
    
    def get_pose_action(self, pose):
        self.mappings = self.load_mappings()
        return self.mappings.get(pose)
    
    def set_pose_action(self, pose, action):
        self.mappings[pose] = action
        self.save_mappings()

    def delete_mapping(self, pose):
        self.mappings[pose] = ""
        self.save_mappings()

    def delete_pose(self, pose):
        self.mappings.pop(pose)
        self.save_mappings()

    def get_mappings(self):
        return self.mappings
    
    def add_pose(self, pose):
        self.mappings[pose] = ""
        self.save_mappings()
        