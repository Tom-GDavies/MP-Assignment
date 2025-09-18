

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
# Last Modified: 2025-Aug-21

import os
import shutil
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt

# Set to true to print the images
show_images = True


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


def run_task2(image_path, config):
    # TODO: Implement task 2 here

    #####################################################
    # Import image
    #####################################################

    for file_name in os.listdir(image_path):
        if re.search(r"bn\d+\.png", file_name):
            file_path = os.path.join(image_path, file_name)
            image = cv2.imread(file_path)

            if image is None:
                print(f"Failed to load image at : {file_path}")
                continue

            resized_image = cv2.resize(image, (800,500))

            gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

            if show_images:
                cv2.imshow("Grayscale", gray)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #####################################################
            # Thresholding
            #####################################################

            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)

            if show_images:
                cv2.imshow("Thresholded", thresh)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #####################################################
            # Noise reduction and morphology
            #####################################################

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=8)

            if show_images:
                cv2.imshow("Cleaned 1", close)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
            close = cv2.morphologyEx(close , cv2.MORPH_OPEN, kernel, iterations=1)

            if show_images:
                cv2.imshow("Cleaned 1 open", close)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            inverted = cv2.bitwise_not(close)

            #####################################################
            # Find components
            #####################################################

            max_prop = 0.35
            min_prop = 0.04

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

            img_num = re.search(r"bn(\d+)\.png", file_name).group(1)

            folder = os.path.join("output/task2", f"bn{img_num}")

            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

            height, width = resized_image.shape[:2]
            img_area = height * width
            max_area = img_area * max_prop
            min_area = img_area * min_prop

            chars = []
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]

                if area > max_area or area < min_area or w < 10 or h < 10:
                    continue

                char = gray[y:y+h, x:x+w]
                chars.append((x, char))

            chars = sorted(chars, key=lambda c: c[0])

            #####################################################
            # Save each character
            #####################################################

            for i, (_, char) in enumerate(chars, start=1):
                save_output(os.path.join(folder, f"c{i:02d}.png"), char, output_type='image')

            if show_images:
                inverted = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
                for i in range(1, num_labels):
                    x, y, w, h, area = stats[i]
                    if not (area > max_area or area < min_area or w < 10 or h < 10):
                        cv2.rectangle(inverted, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow("Connected Components", inverted)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            
        else:
            print(f"Error: invalid file name: {file_name}")
