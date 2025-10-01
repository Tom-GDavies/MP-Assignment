

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
import ultralytics
import torch
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


print_images = True

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
    # Train or load model
    ##############################################################################
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

    ##############################################################################
    # For each image in provided folder
    ##############################################################################

    for filename in os.listdir(image_path):
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(image_path, filename)

            ##############################################################################
            # Extract the building number from the image
            ##############################################################################
            whole_building_number = extract_building_number(model, file_path)

            if print_images and whole_building_number is not None:
                cv2.imshow("Building Number", whole_building_number)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            ##############################################################################
            # Extract individual characters from building number
            ##############################################################################
            characters = extract_character(whole_building_number)

            if print_images:
                for i, char_img in enumerate(characters):
                    cv2.imshow(f"Character {i+1}", char_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            ##############################################################################
            # For each character
            ##############################################################################
            whole_number = ""

            for character in characters:
                ##############################################################################
                # Predict each character
                ##############################################################################
                prediction = predict_character(character)

                print("Predicted Character:", prediction)

                whole_number += str(prediction)

            ##############################################################################
            # Output whole building number
            ##############################################################################
            if whole_number != "":
                name_portion, _ = os.path.splitext(filename)
                new_filename = f"{name_portion}.txt"

                output_path = os.path.join("./output/task4", new_filename)

                save_output(output_path, whole_number, output_type='txt')





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

    if print_images:
        cv2.imshow("Grayscale", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)

    if print_images:
        cv2.imshow("Thresholded", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=8)

    if print_images:
        cv2.imshow("Cleaned 1", close)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    close = cv2.morphologyEx(close , cv2.MORPH_OPEN, kernel, iterations=1)

    if print_images:
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

    height, width = resized_image.shape[:2]
    img_area = height * width
    max_area = img_area * max_prop
    min_area = img_area * min_prop

    char_list = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area > max_area or area < min_area or w < 10 or h < 10:
            continue

        char_img = gray[y:y+h, x:x+w]
        char_list.append((x, char_img))

    char_list.sort(key=lambda c: c[0])

    characters = [c[1] for c in char_list]

    return characters





def predict_character(character):
    retrain_model = False
    if retrain_model or not os.path.exists("digit_cnn.h5"):
        model = retrain_character_model()
    else:
        model = load_model("digit_cnn.h5")

    #####################################################
    # Inference on external images
    #####################################################
    resized = cv2.resize(character, (32,32))

    thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(close)

    colourAgain = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)

    scaled = colourAgain.astype(np.float32) / 255.0
    expanded = np.expand_dims(scaled, axis=0)

    pred = model.predict(expanded)
    predicted_class = np.argmax(pred, axis=1)[0]

    return predicted_class

            



def retrain_character_model():
    #####################################################
    # Load MNIST data
    #####################################################
    ds_train, ds_test = tfds.load('svhn_cropped', split=['train', 'test'], as_supervised=True)


    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        label = tf.one_hot(label, depth=10)
        return image, label
    
    # Test data
    ds_test = ds_test.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)

    # Train and val data
    ds_train = ds_train.map(preprocess)
    ds_train = ds_train.shuffle(10000, reshuffle_each_iteration=True)

    ds_val = ds_train.take(5000)
    ds_train = ds_train.skip(5000)

    # Batch AFTER splitting
    ds_train = ds_train.batch(64).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.batch(64).prefetch(tf.data.AUTOTUNE)

    #####################################################
    # Define CNN model
    #####################################################
    model = models.Sequential([
        # input shape should be (32, 32, 3) for RGB
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    #####################################################
    # Train model
    #####################################################
    callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_val,
        callbacks=[callback]
    )

    test_loss, test_acc = model.evaluate(ds_test)
    print("Test accuracy:", test_acc, ", Test loss:", test_loss)


    model.save("digit_cnn.h5")

    return model