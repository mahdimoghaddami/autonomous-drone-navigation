"""DroNet ResNet-8 architecture and Keras 2/3 weight & JSON compatibility helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import h5py
import keras
from keras import regularizers
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from keras.models import Model


def _residual_block(x: keras.KerasTensor, filters: int, block_idx: int) -> keras.KerasTensor:
    """Build a pre-activation residual block with exact legacy layer names."""
    bn1_idx = 2 * block_idx - 1
    bn2_idx = 2 * block_idx
    conv_a_idx = 3 * block_idx - 1
    conv_b_idx = 3 * block_idx
    conv_skip_idx = 3 * block_idx + 1

    res = BatchNormalization(name=f"batch_normalization_{bn1_idx}")(x)
    res = Activation("relu", name=f"activation_{bn1_idx}")(res)
    res = Conv2D(
        filters,
        (3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"conv2d_{conv_a_idx}",
    )(res)

    res = BatchNormalization(name=f"batch_normalization_{bn2_idx}")(res)
    res = Activation("relu", name=f"activation_{bn2_idx}")(res)
    res = Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"conv2d_{conv_b_idx}",
    )(res)

    skip = Conv2D(
        filters,
        (1, 1),
        strides=(2, 2),
        padding="same",
        name=f"conv2d_{conv_skip_idx}",
    )(x)
    return Add(name=f"add_{block_idx}")([skip, res])


def resnet8(
    img_width: int = 200,
    img_height: int = 200,
    img_channels: int = 1,
    output_dim: int = 1,
    verbose: bool = False,
) -> Model:
    """Define the DroNet 8-layer residual CNN with steering and collision heads."""
    img_input = Input(shape=(img_height, img_width, img_channels), name="input_1")

    x1 = Conv2D(32, (5, 5), strides=(2, 2), padding="same", name="conv2d_1")(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="max_pooling2d_1")(x1)

    x3 = _residual_block(x1, filters=32, block_idx=1)
    x5 = _residual_block(x3, filters=64, block_idx=2)
    x7 = _residual_block(x5, filters=128, block_idx=3)

    x = Flatten(name="flatten_1")(x7)
    x = Activation("relu", name="activation_7")(x)
    x = Dropout(0.5, name="dropout_1")(x)

    steer = Dense(output_dim, name="dense_1")(x)

    coll = Dense(output_dim, name="dense_2")(x)
    coll = Activation("sigmoid", name="activation_8")(coll)

    model = Model(inputs=img_input, outputs=[steer, coll], name="model_1")
    if verbose:
        model.summary()
    return model


def load_model_weights(model: Model, weights_path: str | Path) -> None:
    """Load model weights from either `.h5` (legacy HDF5) or `.weights.h5`."""
    model.load_weights(str(weights_path))


def save_model_weights(model: Model, weights_path: str | Path) -> None:
    """Save model weights supporting both `.weights.h5` and legacy `.h5` filenames."""
    path_str = str(weights_path)
    if path_str.endswith(".weights.h5"):
        model.save_weights(path_str)
        return

    if path_str.endswith(".h5"):
        from keras.src.legacy.saving import legacy_h5_format

        with h5py.File(path_str, "w") as h5f:
            legacy_h5_format.save_weights_to_hdf5_group(h5f, model)
        return

    model.save_weights(path_str)


def get_model(
    img_width: int,
    img_height: int,
    img_channels: int,
    output_dim: int = 1,
    weights_path: str | Path | None = None,
    verbose: bool = False,
) -> Model:
    """Initialize DroNet and optionally restore weights from disk."""
    model = resnet8(img_width, img_height, img_channels, output_dim, verbose=verbose)
    if weights_path is not None:
        load_model_weights(model, weights_path)
    return model


def model_to_json(model: Model, json_model_path: str | Path) -> None:
    """Serialize a Keras model architecture into JSON."""
    Path(json_model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_model_path, "w", encoding="utf-8") as f:
        f.write(model.to_json())


def json_to_model(json_model_path: str | Path) -> Model:
    """Load a DroNet model from either a Keras 3 JSON or a legacy Keras 2 JSON file."""
    with open(json_model_path, "r", encoding="utf-8") as f:
        raw = f.read()

    payload = json.loads(raw)
    keras_version = str(payload.get("keras_version", ""))
    if payload.get("class_name") == "Model" or keras_version.startswith("2."):
        layers = payload.get("config", {}).get("layers", [])
        input_shape = [None, 200, 200, 1]
        output_dim = 1
        for layer in layers:
            if layer.get("class_name") == "InputLayer":
                input_shape = layer.get("config", {}).get("batch_input_shape", input_shape)
            elif layer.get("name") == "dense_1":
                output_dim = int(layer.get("config", {}).get("units", output_dim))
        img_height, img_width, img_channels = (
            int(input_shape[1]),
            int(input_shape[2]),
            int(input_shape[3]),
        )
        return resnet8(img_width=img_width, img_height=img_height, img_channels=img_channels, output_dim=output_dim)

    return keras.models.model_from_json(raw)
