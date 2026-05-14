import os
import shutil

# 🔹 CHANGE THIS to the folder that contains all subfolders
source_root = r"C:\esp32-csi-tool\Activity\WA"

# 🔹 CHANGE THIS to your output folder
target = r"C:\esp32-csi-tool\Activity\new1"

os.makedirs(target, exist_ok=True)

for folder in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            src = os.path.join(folder_path, file)

            # add folder name to avoid overwrite
            new_name = folder + "_" + file
            dst = os.path.join(target, new_name)

            shutil.copy2(src, dst)

print("✅ All files copied into one folder!")
