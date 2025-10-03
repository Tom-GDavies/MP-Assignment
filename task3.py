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


# Author: Thomas Davies
# Last Modified: 2025-03-10

import os
import shutil
import re
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model


# Set to true to print the images
show_images = False


def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")

def preprocess_digit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        padded = np.zeros((28,28), dtype=np.uint8)
    else:
        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]

        scale = 20.0 / max(w, h)
        new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
        resized_digit = cv2.resize(digit, (new_w, new_h))

        pad_top = (28 - new_h) // 2
        pad_bottom = 28 - new_h - pad_top
        pad_left = (28 - new_w) // 2
        pad_right = 28 - new_w - pad_left

        padded = cv2.copyMakeBorder(resized_digit, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=0)

    scaled = padded.astype(np.float32) / 255.0
    scaled = np.expand_dims(scaled, axis=-1)
    return np.expand_dims(scaled, axis=0)


def run_task3(image_path, config):

    model_path = Path(__file__).resolve().parent / "digit_cnn.h5"

    if not model_path.exists():
        print(f"Error loading CNN, CNN does not exist at {model_path}. Please train the required model.")
    else:
        model = load_model(str(model_path))

        #####################################################
        # Inference on external images
        #####################################################
        for folder_name in os.listdir(image_path):
            if re.search(r"bn\d+", folder_name):

                index = ''.join(filter(str.isdigit, folder_name))

                output_dir = Path(__file__).resolve().parent / f"output/task3/bn{index}"
                if output_dir.exists():
                    shutil.rmtree(output_dir)         
                output_dir.mkdir(parents=True, exist_ok=True)

                folder_path = os.path.join(image_path, folder_name)

                for file_name in os.listdir(folder_path):
                    if re.search(r"c\d+\.png", file_name):

                        file_path = os.path.join(folder_path, file_name)
                        image = cv2.imread(file_path)

                        if image is None:
                            print(f"Failed to load image at : {file_path}")
                            continue

                        processed_image = preprocess_digit(image)

                        pred = model.predict(processed_image)
                        predicted_class = np.argmax(pred, axis=1)[0]

                        print("Predicted digit:", predicted_class)

                        if show_images:
                            cv2.imshow(f"{predicted_class}", (processed_image[0,:,:,0]*255).astype(np.uint8))
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        # Save output in the same folder, preserving cXX.txt naming
                        img_num = re.search(r"c(\d+)\.png", file_name).group(1)
                        output_txt_path = output_dir / f"c{img_num}.txt"
                        save_output(str(output_txt_path), str(predicted_class), output_type='txt')
                    else:
                        print(f"Error: invalid file name: {file_name}")
            else:
                print(f"Error: invalid folder name: {folder_name}")