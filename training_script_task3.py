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

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Set to true to print the images
show_images = False
retrain_model = False



def train_model():
    #####################################################
    # Load MNIST data
    #####################################################
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        label = tf.one_hot(label, depth=10)
        return image, label
    
    # Apply preprocessing
    ds_test = ds_test.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)
    ds_train = ds_train.map(preprocess).shuffle(10000, reshuffle_each_iteration=True)

    # Split into train/val
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
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
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