from hydra import utils
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from dvclive.keras import DvcLiveCallback

# for hydra configuration
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.keras import WandbCallback

# improt logging
import logging

from tensorflow.keras.preprocessing import image

logger = logging.getLogger(__name__)
from pathlib import Path


@hydra.main(config_path="../config", config_name="config")
def train(cfg: DictConfig):

    config = {
        "lr": cfg.model.optimizer.lr,
        "base_filters": cfg.model.base_filters,
        "epochs": cfg.trainer.epochs,
    }
    # wandb
    wandb.init(project="dvc_wandbc", config=config)

    print(wandb.config)

    cfg.model.base_filters = wandb.config.base_filters
    cfg.model.optimizer.lr = wandb.config.lr
    cfg.trainer.epochs = wandb.config.epochs

    logger.info(OmegaConf.to_yaml(cfg))

    # cofig file for training
    batch_size = cfg.trainer.batch_size
    print(hydra.utils.get_original_cwd())
    train = Path(hydra.utils.get_original_cwd() / Path(cfg.data_.train))
    valid = Path(hydra.utils.get_original_cwd() / Path(cfg.data_.valid))
    print(f"path for training --{train}")
    print(f"path for valid--{valid}")

    image_size = (cfg.image.size, cfg.image.size)
    print(image_size)

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train, shuffle=True, batch_size=batch_size, image_size=image_size
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train, shuffle=True, batch_size=batch_size, image_size=image_size
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal_and_vertical", input_shape=(image_size[0], image_size[1], 3)
            ),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential(
        [
            data_augmentation,
            tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(
                16, 3, padding="same", activation=cfg.model.activation
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                cfg.model.base_filters,
                3,
                padding="same",
                activation=cfg.model.activation,
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                64, 3, padding="same", activation=cfg.model.activation
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=cfg.model.activation),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    print(model.summary())
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")
    callbacks = [WandbCallback()]

    model.fit(
        train_dataset,
        epochs=cfg.trainer.epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
    )

    # model.load_weights(hydra.utils.to_absolute_path("data/train/best_weights.h5"))
    # tf.saved_model.save(model,hydra.utils.to_absolute_path("data/train/model"))


if __name__ == "__main__":
    train()
