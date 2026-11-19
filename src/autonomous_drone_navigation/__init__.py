"""DroNet: Learning to Fly by Driving (CSI 5170 autonomous drone navigation package)."""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from autonomous_drone_navigation.cli import main
from autonomous_drone_navigation.config import TEST_PHASE, TRAIN_PHASE, DroneConfig
from autonomous_drone_navigation.control import ControlCommand, DroneControlConfig, DroneController
from autonomous_drone_navigation.data import DroneDataGenerator, DroneDirectoryIterator
from autonomous_drone_navigation.losses import (
    HardMiningEntropyLoss,
    HardMiningMSELoss,
    HardMiningState,
    hard_mining_entropy,
    hard_mining_mse,
)
from autonomous_drone_navigation.model import (
    get_model,
    json_to_model,
    load_model_weights,
    model_to_json,
    resnet8,
    save_model_weights,
)

__all__ = [
    "ControlCommand",
    "DroneConfig",
    "DroneControlConfig",
    "DroneController",
    "DroneDataGenerator",
    "DroneDirectoryIterator",
    "HardMiningEntropyLoss",
    "HardMiningMSELoss",
    "HardMiningState",
    "TEST_PHASE",
    "TRAIN_PHASE",
    "get_model",
    "hard_mining_entropy",
    "hard_mining_mse",
    "json_to_model",
    "load_model_weights",
    "main",
    "model_to_json",
    "resnet8",
    "save_model_weights",
]
