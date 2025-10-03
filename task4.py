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
import re
import shutil
import cv2
import numpy as np
import torch
import ultralytics
from pathlib import Path
from tensorflow.keras.models import load_model



print_images = False

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


def run_task4(image_path, config):
    # TODO: Implement task 4 here


    ##############################################################################
    # Load model
    ##############################################################################

    model_path = Path(__file__).resolve().parent / "runs/detect/train/weights/best.pt"
    
    if not model_path.exists():
        print(f"Error loading CNN, CNN does not exist at {model_path}. Please train the required model.")

    else:
        model = ultralytics.YOLO(str(model_path))
        ##############################################################################
        # For each image in provided folder
        ##############################################################################

        for folder_name in os.listdir(image_path):
            if re.search(r"bn\d+", folder_name):
                index = ''.join(filter(str.isdigit, folder_name))
                folder_path = os.path.join(image_path, folder_name)

                # Preserve the relative folder structure in output
                output_dir = Path(__file__).resolve().parent / f"output/task4/{folder_name}"
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(".jpg"):
                        file_path = os.path.join(folder_path, filename)

                        whole_building_number = extract_building_number(model, file_path)
                        if whole_building_number is None:
                            print(f"No building number detected in {filename}, skipping...")
                            continue

                        if print_images:
                            cv2.imshow("Building Number", whole_building_number)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        characters = extract_character(whole_building_number)

                        if print_images:
                            for i, char_img in enumerate(characters):
                                cv2.imshow(f"Character {i+1}", char_img)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()

                        if len(characters) < 3:
                            print("Less than 3 characters were extracted. Skipping...")
                            continue

                        whole_number = ""
                        for character in characters:
                            prediction = predict_character(character)
                            whole_number += str(prediction)

                        # Fix misclassified 7 -> 1 in first two digits
                        digits = list(whole_number)
                        if len(digits) > 0 and digits[0] == "7":
                            digits[0] = "1"
                        if len(digits) > 1 and digits[1] == "7":
                            digits[1] = "1"
                        whole_number = "".join(digits)

                        # Save output in same relative folder
                        name_portion, _ = os.path.splitext(filename)
                        output_txt_path = output_dir / f"{name_portion}.txt"
                        save_output(output_txt_path, whole_number, output_type='txt')
            else:
                print(f"Error: invalid folder name: {folder_name}")





def extract_building_number(model, image_path):
    results = model.predict(image_path, imgsz=960)

    if len(results[0].boxes) > 0:
        # Take the first detected box only
        box = results[0].boxes.xyxy[0].cpu().numpy()
        image = cv2.imread(image_path)
        x1, y1, x2, y2 = box.astype(int)
        cropped = image[y1:y2, x1:x2]

        return cropped






def extract_character(whole_number):
    
    characters = []

    resized_image = cv2.resize(whole_number, (800,500))

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    #equalised = cv2.equalizeHist(gray)
    equalised = gray

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(equalised, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    if print_images:
        cv2.imshow("Closed", closed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    regions = stable_mser_regions(closed)

    vis = resized_image.copy()

    for (x, y, w, h) in regions:
        colour = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)


    if print_images:
        cv2.imshow("MSER Regions", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    filtered = [(x, y, w, h) for (x, y, w, h) in regions if w <= h]

    filtered = non_max_suppression(filtered)

    filtered = list(filtered)
    filtered.sort(key=lambda c: c[2] * c[3], reverse=True)

    # IM GOING TO MAKE THE ASSUMPTION THAT THE BUILDING NUMBER DOESNT CONTAIN A LETTER FOR SIMPLICITY
    # IM AWARE THAT THEY CAN BUT THAT ADDS TOO MUCH COMPLEXITY 

    # If more than 3 characters are extracted then keep the 3 largest
    if len(filtered) > 3:
        filtered = filtered[:3]

    filtered.sort(key=lambda c: c[0])

    characters = [resized_image[y:y+h, x:x+w] for (x, y, w, h) in filtered]

    return characters




def stable_mser_regions(image):
    # Was testing something but ended up getting better results without different variations
    variations = [0.15]#, 0.2, 0.25, 0.3]
    all_bboxes = []

    image_area = image.shape[0] * image.shape[1]
    min_prop = 0.02
    max_prop = 0.3

    for var in variations:
        mser = cv2.MSER_create()
        mser.setDelta(7)
        mser.setMinArea(int(image_area * min_prop))
        mser.setMaxArea(int(image_area * max_prop))
        mser.setMaxVariation(var)
        mser.setMinDiversity(0.5)

        regions, _ = mser.detectRegions(image)

        bboxes = [cv2.boundingRect(p.reshape(-1,1,2)) for p in regions]

        image_h, image_w = image.shape[:2]
        min_area = int(image_h * image_w * min_prop)
        max_area = int(image_h * image_w * max_prop)

        if print_images:
            for p in regions:
                hull = cv2.convexHull(p.reshape(-1,1,2))
                cv2.polylines(image, [hull], 1, (0,255,0), 2)
            cv2.imshow("MSER raw", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        filtered_bboxes = []
        for (x, y, w, h) in bboxes:
            area = w * h
            if min_area <= area <= max_area:
                filtered_bboxes.append((x, y, w, h))

        all_bboxes.extend(filtered_bboxes)
        
    
    merged = merge_boxes(all_bboxes, overlap_thresh=0.5)
        

    return merged






def merge_boxes(boxes, overlap_thresh=0.7):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]

    merged = []

    used = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if used[i]:
            continue
        xi1, yi1, xi2, yi2 = x1[i], y1[i], x2[i], y2[i]
        for j in range(i+1, len(boxes)):
            if used[j]:
                continue
            xj1, yj1, xj2, yj2 = x1[j], y1[j], x2[j], y2[j]

            # Compute intersection
            xx1 = max(xi1, xj1)
            yy1 = max(yi1, yj1)
            xx2 = min(xi2, xj2)
            yy2 = min(yi2, yj2)
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            intersection = w * h

            area_i = (xi2 - xi1) * (yi2 - yi1)
            area_j = (xj2 - xj1) * (yj2 - yj1)
            overlap_ratio = intersection / min(area_i, area_j)

            if overlap_ratio > overlap_thresh:
                # Merge boxes
                xi1 = min(xi1, xj1)
                yi1 = min(yi1, yj1)
                xi2 = max(xi2, xj2)
                yi2 = max(yi2, yj2)
                used[j] = True

        merged.append((xi1, yi1, xi2 - xi1, yi2 - yi1))

    return merged






def non_max_suppression(boxes, overlapThreshold=0.5):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(areas)[::-1]

    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[idxs[1:]]

        idxs = idxs[np.where(overlap <= overlapThreshold)[0] + 1]

    return boxes[keep].astype("int")





def predict_character(character):
    model_path = Path(__file__).resolve().parent / "digit_cnn.h5"

    if not model_path.exists():
        print(f"Error loading CNN, CNN does not exist at {model_path}. Please train the required model.")
    else:
        model = load_model(str(model_path))

        #####################################################
        # Inference on external images
        #####################################################
        
        processed_image = preprocess_digit(character)

        pred = model.predict(processed_image)
        predicted_class = np.argmax(pred, axis=1)[0]

        print("Predicted digit:", predicted_class)

        if print_images:
            cv2.imshow(f"{predicted_class}", (processed_image[0,:,:,0]*255).astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return predicted_class

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