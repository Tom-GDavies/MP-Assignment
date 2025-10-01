import os
import shutil
import re
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set to true to print the images
show_images = False
retrain_model = True


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


def run_task3(image_path, config):

    if retrain_model or not os.path.exists("digit_cnn.h5"):

        #####################################################
        # Load MNIST data
        #####################################################
        ds_train, ds_test = tfds.load('svhn_cropped', split=['train', 'test'], as_supervised=True)


        def preprocess(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            label = tf.where(label == 10, tf.cast(0, label.dtype), label)
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
        # Data augmentation
        #####################################################

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )

        batch_size = 64

        x_train, y_train = [], []
        for img, lbl in ds_train.unbatch():
            x_train.append(img.numpy())
            y_train.append(lbl.numpy())

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

        #####################################################
        # Define CNN model
        #####################################################
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3,3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        #####################################################
        # Train model
        #####################################################
        callback = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        
        history = model.fit(
            train_generator,
            epochs=100,
            validation_data=ds_val,
            callbacks=[callback]
        )

        test_loss, test_acc = model.evaluate(ds_test)
        print("Test accuracy:", test_acc, ", Test loss:", test_loss)


        model.save("digit_cnn.h5")

    else:
        model = load_model("digit_cnn.h5")

    #####################################################
    # Inference on external images
    #####################################################
    for folder_name in os.listdir(image_path):
        if re.search(r"bn\d+", folder_name):

            folder_num = re.search(r"bn(\d+)", folder_name).group(1)
            folder = os.path.join("output/task3", f"bn{folder_num}")

            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
            
            new_image_path = os.path.join(image_path, folder_name)

            for file_name in os.listdir(new_image_path):
                if re.search(r"c\d+\.png", file_name):

                    file_path = os.path.join(new_image_path, file_name)
                    image = cv2.imread(file_path)

                    if image is None:
                        print(f"Failed to load image at : {file_path}")
                        continue

                    resized = cv2.resize(image, (32,32))

                    scaled = resized.astype(np.float32) / 255.0
                    expanded = np.expand_dims(scaled, axis=0)

                    pred = model.predict(expanded)
                    predicted_class = np.argmax(pred, axis=1)[0]

                    print("Predicted digit:", predicted_class)

                    if show_images:
                        display_img = image
                        cv2.imshow(f"{predicted_class}", display_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    img_num = re.search(r"c(\d+)\.png", file_name).group(1)
                    save_output(os.path.join(folder, f"c{img_num}.txt"), str(predicted_class), output_type='txt')
                else:
                    print(f"Error: invalid file name: {folder_name}")
        else:
            print(f"Error: invalid folder name: {folder_name}")
