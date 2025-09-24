import os

# Directory containing the images (change if needed)
directory = "."

# List all .jpg files and sort them
jpg_files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]
jpg_files.sort()  # optional: sorts alphabetically

# Rename files sequentially
for i, filename in enumerate(jpg_files, start=1):
    old_path = os.path.join(directory, filename)
    new_filename = f"img{i}.jpg"
    new_path = os.path.join(directory, new_filename)
    os.rename(old_path, new_path)
    print(f"{filename} -> {new_filename}")
