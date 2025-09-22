

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
import re
import cv2
import numpy as np
import glob
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# Set to true to print the images
show_images = True
retrain_model = True


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

def run_task3(image_path, config):
    # TODO: Implement task 3 here

    if retrain_model or not os.path.exists("digit_cnn.h5"):

        data_dir = "synthetic_digits"
        images = []
        labels = []

        #####################################################
        # Prepare training data
        #####################################################

        for filename in os.listdir(data_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(data_dir, filename)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (32,32))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=-1)  # only channel dim, shape -> (32,32,1)
                images.append(img)


                label = int(filename.split("_")[0])
                labels.append(label)

            else:
                print(f"File {filename} is not a png")


        X = np.array(images)
        X = X.reshape(-1, 32, 32, 1)

        y = np.array(labels)
        
        
        y = to_categorical(y, num_classes=10)

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)


        #####################################################
        # Train model
        #####################################################

        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
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

        
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=64,
            validation_data=(X_val, y_val)
        )

        callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[callback])

        test_loss, test_acc = model.evaluate(X_test, y_test)
        print("Test accuracy:", test_acc)

        model.save("digit_cnn.h5")
    else:
        model = load_model("digit_cnn.h5")


    for folder_name in os.listdir(image_path):
        if re.search(r"bn\d+", folder_name):

            folder_num = re.search(r"bn(\d+)", folder_name).group(1)
            folder = os.path.join("output/task3", f"bn{folder_num}")

            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
            
            new_image_path = os.path.join(image_path, folder_name)

            print(image_path)
            for file_name in os.listdir(new_image_path):
                if re.search(r"c\d+\.png", file_name):

                    #####################################################
                    # Import file
                    #####################################################

                    file_path = os.path.join(new_image_path, file_name)
                    image = cv2.imread(file_path)

                    if image is None:
                        print(f"Failed to load image at : {file_path}")
                        continue

                    #####################################################
                    # Data preprocessing
                    #####################################################
                    
                    gray = cv2.resize(image, (32,32))
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
                    gray = gray / 255.0
                    gray = np.expand_dims(gray, axis=-1)
                    gray = np.expand_dims(gray, axis=0)


                    #####################################################
                    # Prediction pipeline
                    #####################################################

                    pred = model.predict(gray)
                    predicted_class = np.argmax(pred, axis=1)[0]

                    print("Predicted digit:", predicted_class)

                    if show_images:
                        display_img = gray[0, :, :, 0]  # shape -> (32, 32)
                        cv2.imshow(f"{predicted_class}", display_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()


                    #####################################################
                    # Save output
                    #####################################################

                    img_num = re.search(r"c(\d+)\.png", file_name).group(1)

                    save_output(os.path.join(folder, f"c{img_num}.txt"), "num", output_type='txt')

                
                else:
                    print(f"Error: invalid file name: {folder_name}")
        else:
            print(f"Error: invalid folder name: {folder_name}")

