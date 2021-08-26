import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from dvclive.keras import DvcLiveCallback
from tensorflow.keras.callbacks import ModelCheckpoint



from tensorflow.keras.preprocessing import image

from pathlib import Path
from pydoc import locate

import yaml

PARAMETERS_FILE = "params.yaml"
with open(PARAMETERS_FILE) as file:
    print(f"Parsing parameters from \"{PARAMETERS_FILE}\"")
    params = yaml.safe_load(file)

ROOT_DIR = Path(params["root_dir"])

# Dataset parameters
RAW_DATASET_DIR = ROOT_DIR / params["sub_dir"]
DATASET_DIR = ROOT_DIR / params["split"]["dir"]

# Train parameters
BATCH_SIZE = params["train"]["batch_size"]
IMG_SIZE = tuple(params["train"]["img_size"])
LEARNING_RATE = params["train"]["learning_rate"]
TRAIN_DIR = ROOT_DIR / params["train"]["subdir"]["train"]
CHANNELS=params["train"]["n_channels"]
EPOCHS=params["train"]["epochs"]
ACTIVATION = params["train"]["activation"]

print(ACTIVATION)


def train() :
    


    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR / "train",
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
        ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR / "val",
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
        ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    data_augmentation = keras.Sequential(
                        [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(IMG_SIZE[0], IMG_SIZE[1],3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
                        ]
)
    model = Sequential([
    data_augmentation,
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    print(model.summary())
    
    callbacks = [
    # Use dvclive's Keras callback
    DvcLiveCallback(),
    ModelCheckpoint(str(TRAIN_DIR / "best_weights.h5"), save_best_only=True),
        ]
    history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=callbacks,
        )
    model.load_weights(str(TRAIN_DIR / "best_weights.h5"))
    tf.saved_model.save(model, str(TRAIN_DIR / "model"))




if __name__=='__main__':
    train()
    