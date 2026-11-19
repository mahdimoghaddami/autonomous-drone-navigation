"""DroNet training pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from autonomous_drone_navigation.config import DroneConfig
from autonomous_drone_navigation.data import DroneDataGenerator, DroneDirectoryIterator
from autonomous_drone_navigation.logging_utils import DroneTrainingCallback, TabularLogger
from autonomous_drone_navigation.losses import HardMiningState, hard_mining_entropy, hard_mining_mse
from autonomous_drone_navigation.model import get_model, model_to_json, save_model_weights


def train_model(
    train_data_generator: DroneDirectoryIterator,
    val_data_generator: DroneDirectoryIterator,
    model: keras.Model,
    config: DroneConfig,
    initial_epoch: int = 0,
) -> keras.callbacks.History:
    """Compile and train DroNet with hard-mining losses and dynamic loss weighting."""
    state = HardMiningState(
        alpha=1.0,
        beta=0.0,
        k_mse=config.batch_size,
        k_entropy=config.batch_size,
    )

    optimizer = optimizers.Adam(learning_rate=1e-3, weight_decay=1e-5)
    model.compile(
        loss=[hard_mining_mse(state), hard_mining_entropy(state)],
        optimizer=optimizer,
    )

    best_weights_path = os.path.join(
        config.experiment_rootdir, "weights_{epoch:03d}.weights.h5"
    )
    write_best_model = ModelCheckpoint(
        filepath=best_weights_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    logger = TabularLogger(config.experiment_rootdir)
    save_model_and_loss = DroneTrainingCallback(
        filepath=config.experiment_rootdir,
        period=config.log_rate,
        batch_size=config.batch_size,
        state=state,
        logger=logger,
    )

    history = model.fit(
        train_data_generator,
        epochs=config.epochs,
        callbacks=[write_best_model, save_model_and_loss],
        validation_data=val_data_generator,
        initial_epoch=initial_epoch,
    )

    final_weights_path = os.path.join(config.experiment_rootdir, config.weights_fname)
    save_model_weights(model, final_weights_path)
    return history


def run_training(config: DroneConfig) -> keras.callbacks.History:
    """Prepare generators, build or restore DroNet, and run training."""
    exp_dir = Path(config.experiment_rootdir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_datagen = DroneDataGenerator(
        rotation_range=0.2,
        rescale=1.0 / 255.0,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )
    train_generator = train_datagen.flow_from_directory(
        config.train_dir,
        shuffle=True,
        color_mode=config.img_mode,
        target_size=(config.img_width, config.img_height),
        crop_size=(config.crop_img_height, config.crop_img_width),
        batch_size=config.batch_size,
    )

    val_datagen = DroneDataGenerator(rescale=1.0 / 255.0)
    val_generator = val_datagen.flow_from_directory(
        config.val_dir,
        shuffle=True,
        color_mode=config.img_mode,
        target_size=(config.img_width, config.img_height),
        crop_size=(config.crop_img_height, config.crop_img_width),
        batch_size=config.batch_size,
    )

    weights_path: Path | None = None
    initial_epoch = 0
    if config.restore_model:
        weights_path = exp_dir / config.weights_fname
        initial_epoch = config.initial_epoch

    model = get_model(
        img_width=config.crop_img_width,
        img_height=config.crop_img_height,
        img_channels=config.img_channels,
        output_dim=1,
        weights_path=weights_path,
    )

    json_model_path = exp_dir / config.json_model_fname
    model_to_json(model, json_model_path)

    return train_model(train_generator, val_generator, model, config, initial_epoch=initial_epoch)
