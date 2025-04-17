import csv

class GestureManager:
    def __init__(self, gesture_file = "data/gestures.csv", pose_file = "data/poses.txt"):
        self.gesture_file = gesture_file
        self.pose_file = pose_file

    def get_all_poses(self):
        with open(self.pose_file, 'r') as file:
            return [line.strip() for line in file.readlines() if line.strip()]
        
    def add_pose(self, pose_name, processed_landmarks):
        # Add landmark data to gestures.csv
        with open(self.gesture_file, mode = 'a', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow([pose_name] + processed_landmarks)

        # Add pose name to poses.txt if its new
        poses = self.get_all_poses()
        if pose_name not in poses:
            with open(self.pose_file, 'a') as file:
                file.write(pose_name + "\n")

    def delete_pose(self, pose_name):
        # Remove all matching rows from gestures.csv
        with open(self.gesture_file, 'r') as file:
            rows = list(csv.reader(file))
        rows = [row for row in rows if row[0] != pose_name]
        with open(self.gesture_file, 'w', newline = '') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        # Remove pose name from poses.txt
        poses = self.get_all_poses()
        if pose_name in poses:
            poses.remove(pose_name)
            with open(self.pose_file, 'w') as file:
                for pose in poses:
                    file.write(pose + "\n")