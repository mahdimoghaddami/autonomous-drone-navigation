"""Configuration constants and dataclasses for DroNet."""

from dataclasses import dataclass

TEST_PHASE = 0
TRAIN_PHASE = 1


@dataclass
class DroneConfig:
    """Configuration parameters for DroNet training, evaluation, and plotting."""

    img_width: int = 320
    img_height: int = 240
    crop_img_width: int = 200
    crop_img_height: int = 200
    img_mode: str = "grayscale"

    batch_size: int = 32
    epochs: int = 100
    log_rate: int = 10
    initial_epoch: int = 0

    experiment_rootdir: str = "./model"
    train_dir: str = "../training"
    val_dir: str = "../validation"
    test_dir: str = "../testing"

    restore_model: bool = False
    weights_fname: str = "model_weights.h5"
    json_model_fname: str = "model_struct.json"

    @property
    def img_channels(self) -> int:
        if self.img_mode == "rgb":
            return 3
        if self.img_mode == "grayscale":
            return 1
        raise ValueError(f"Unidentified image mode {self.img_mode!r}: use 'grayscale' or 'rgb'")
