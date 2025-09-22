

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: [Your Name]
# Last Modified: 2024-09-09

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import ultralytics
import cv2
import torch

def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")

def run_task1(image_path, config):
    # TODO: Implement task 1 here
    train_model = False

    # Check that cuda is enabled
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    if train_model:
        # Train model
        model = ultralytics.YOLO("yolo11s.pt")

        model.train(
            data="data.yaml",
            epochs=50,
            batch=8,
            imgsz=960,
            workers=2
        )

        # Evaluate model
        metrics = model.val(split="test")
        print(metrics)
    else:
        model = ultralytics.YOLO("runs/detect/train/weights/best.pt")

    # Ensure output folder exists
    os.makedirs("output/task1", exist_ok=True)

    # Iterate over all images in the input directory
    for filename in os.listdir(image_path):
        if filename.lower().startswith("img") and filename.lower().endswith(".jpg"):
            file_path = os.path.join(image_path, filename)
            results = model.predict(file_path, imgsz=960)

            if len(results[0].boxes) > 0:
                # Take the first detected box only
                box = results[0].boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                image = cv2.imread(file_path)
                x1, y1, x2, y2 = box.astype(int)
                cropped = image[y1:y2, x1:x2]

                # Generate output filename: imgX.jpg -> bnX.png
                index = ''.join(filter(str.isdigit, filename))
                output_img_path = os.path.join("output/task1", f"bn{index}.png")

                save_output(output_img_path, cropped, output_type='image')
            else:
                print(f"No building number detected in {filename}. No output created.")