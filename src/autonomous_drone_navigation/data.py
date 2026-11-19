"""Dataset generators, augmentation, and batch evaluation utilities."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import cv2
import keras
import numpy as np
from keras.utils import Progbar

from autonomous_drone_navigation import preprocessing


class DroneDataGenerator:
    """Real-time image augmentation and directory flow factory for DroNet."""

    def __init__(
        self,
        rotation_range: float = 0.0,
        width_shift_range: float = 0.0,
        height_shift_range: float = 0.0,
        rescale: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.rescale = rescale
        self.rng = np.random.default_rng(seed)

    def random_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply random affine rotation and translation to a single image."""
        if (
            self.rotation_range == 0.0
            and self.width_shift_range == 0.0
            and self.height_shift_range == 0.0
        ):
            return x

        h, w = x.shape[0], x.shape[1]
        angle = float(self.rng.uniform(-self.rotation_range, self.rotation_range)) if self.rotation_range else 0.0
        tx = (
            float(self.rng.uniform(-self.width_shift_range, self.width_shift_range) * w)
            if self.width_shift_range
            else 0.0
        )
        ty = (
            float(self.rng.uniform(-self.height_shift_range, self.height_shift_range) * h)
            if self.height_shift_range
            else 0.0
        )

        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        transformed = cv2.warpAffine(
            x,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if x.ndim == 3 and transformed.ndim == 2:
            transformed = transformed[:, :, None]
        return transformed.astype(np.float32)

    def standardize(self, x: np.ndarray) -> np.ndarray:
        """Apply configured pixel rescaling."""
        if self.rescale is not None:
            return (x * np.float32(self.rescale)).astype(np.float32)
        return x.astype(np.float32)

    def flow_from_directory(
        self,
        directory: str | Path,
        target_size: tuple[int, int] = (320, 240),
        crop_size: tuple[int, int] = (200, 200),
        color_mode: str = "grayscale",
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        follow_links: bool = False,
    ) -> DroneDirectoryIterator:
        return DroneDirectoryIterator(
            directory=directory,
            image_data_generator=self,
            target_size=target_size,
            crop_size=crop_size,
            color_mode=color_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            follow_links=follow_links,
        )


class DroneDirectoryIterator(keras.utils.PyDataset):
    """Manage loading DroNet steering and collision experiment directories."""

    def __init__(
        self,
        directory: str | Path,
        image_data_generator: DroneDataGenerator,
        target_size: tuple[int, int] = (320, 240),
        crop_size: tuple[int, int] = (200, 200),
        color_mode: str = "grayscale",
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        follow_links: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.directory = str(directory)
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.crop_size = tuple(crop_size)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.follow_links = follow_links
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()

        if color_mode not in {"rgb", "grayscale"}:
            raise ValueError(f"Invalid color mode {color_mode!r}; expected 'rgb' or 'grayscale'.")
        self.color_mode = color_mode
        channels = 3 if color_mode == "rgb" else 1
        # crop_size is passed as (height, width) matching legacy calls
        self.image_shape = (self.crop_size[0], self.crop_size[1], channels)

        self.samples = 0
        self.formats = {"png", "jpg", "jpeg"}
        self.filenames: list[str] = []
        self.ground_truth_list: list[float] = []
        self.exp_type: list[int] = []

        experiments = [
            subdir
            for subdir in sorted(os.listdir(self.directory))
            if os.path.isdir(os.path.join(self.directory, subdir))
        ]
        self.num_experiments = len(experiments)

        for subdir in experiments:
            self._decode_experiment_dir(os.path.join(self.directory, subdir))

        self.ground_truth = np.asarray(self.ground_truth_list, dtype=np.float32)
        if self.samples == 0:
            raise ValueError(f"Did not find any data in {self.directory}")

        print(f"Found {self.samples} images belonging to {self.num_experiments} experiments.")

        self._indices = np.arange(self.samples)
        if self.shuffle:
            self.rng.shuffle(self._indices)
        self._cursor = 0

    def _recursive_list(self, subpath: str) -> list[tuple[str, list[str], list[str]]]:
        return sorted(os.walk(subpath, followlinks=self.follow_links), key=lambda tpl: tpl[0])

    @staticmethod
    def _sort_key(fname: str) -> tuple[int, str]:
        match = re.search(r"\d+", fname)
        return (int(match.group()) if match else 0, fname)

    def _decode_experiment_dir(self, dir_subpath: str) -> None:
        steerings_filename = os.path.join(dir_subpath, "sync_steering.txt")
        labels_filename = os.path.join(dir_subpath, "labels.txt")

        if os.path.exists(steerings_filename):
            ground_truth = np.atleast_1d(
                np.loadtxt(steerings_filename, usecols=0, delimiter=",", skiprows=1)
            )
            exp_type = 1
        elif os.path.exists(labels_filename):
            ground_truth = np.atleast_1d(np.loadtxt(labels_filename, usecols=0))
            exp_type = 0
        else:
            raise IOError(f"Neither steerings nor labels found in dir {dir_subpath}")

        image_dir_path = os.path.join(dir_subpath, "images")
        for root, _, files in self._recursive_list(image_dir_path):
            sorted_files = sorted(files, key=self._sort_key)
            valid_idx = 0
            for fname in sorted_files:
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext not in self.formats:
                    continue
                absolute_path = os.path.join(root, fname)
                self.filenames.append(os.path.relpath(absolute_path, self.directory))
                self.ground_truth_list.append(float(ground_truth[valid_idx]))
                self.exp_type.append(exp_type)
                self.samples += 1
                valid_idx += 1

    def __len__(self) -> int:
        return int(np.ceil(self.samples / self.batch_size))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.samples)
        index_array = self._indices[start:end]
        return self._get_batches_of_transformed_samples(index_array)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self._indices)
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def __iter__(self) -> DroneDirectoryIterator:
        return self

    def __next__(self) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        return self.next()

    def next(self) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        with self.lock:
            if self._cursor >= len(self):
                self.on_epoch_end()
            batch = self[self._cursor]
            self._cursor += 1
            return batch

    def _get_batches_of_transformed_samples(
        self, index_array: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        current_batch_size = index_array.shape[0]
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=np.float32)
        batch_steer = np.zeros((current_batch_size, 2), dtype=np.float32)
        batch_coll = np.zeros((current_batch_size, 2), dtype=np.float32)

        grayscale = self.color_mode == "grayscale"
        crop_wh = (self.crop_size[1], self.crop_size[0])

        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = preprocessing.load_img(
                os.path.join(self.directory, fname),
                grayscale=grayscale,
                target_size=self.target_size,
                crop_size=crop_wh,
            )
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.exp_type[j] == 1:
                batch_steer[i, 0] = 1.0
                batch_steer[i, 1] = self.ground_truth[j]
                batch_coll[i] = np.array([1.0, 0.0], dtype=np.float32)
            else:
                batch_steer[i] = np.array([0.0, 0.0], dtype=np.float32)
                batch_coll[i, 0] = 0.0
                batch_coll[i, 1] = self.ground_truth[j]

        return batch_x, (batch_steer, batch_coll)


def compute_predictions_and_gt(
    model: keras.Model,
    generator: DroneDirectoryIterator,
    steps: int,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate model predictions, ground truth labels, and task masks over `steps` batches."""
    generator.reset()
    all_steer_preds: list[np.ndarray] = []
    all_coll_preds: list[np.ndarray] = []
    all_steer_gt: list[np.ndarray] = []
    all_coll_gt: list[np.ndarray] = []
    all_ts: list[np.ndarray] = []

    progbar = Progbar(target=steps) if verbose == 1 else None

    for step in range(steps):
        x, gt_lab = next(generator)
        outs = model.predict_on_batch(x)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]

        all_steer_preds.append(np.asarray(outs[0]).reshape(-1))
        all_coll_preds.append(np.asarray(outs[1]).reshape(-1))
        all_steer_gt.append(np.asarray(gt_lab[0][:, 1]).reshape(-1))
        all_coll_gt.append(np.asarray(gt_lab[1][:, 1]).reshape(-1))
        all_ts.append(np.asarray(gt_lab[0][:, 0]).reshape(-1))

        if progbar is not None:
            progbar.update(step + 1)

    predictions = np.column_stack(
        [np.concatenate(all_steer_preds), np.concatenate(all_coll_preds)]
    )
    ground_truth = np.column_stack(
        [np.concatenate(all_steer_gt), np.concatenate(all_coll_gt)]
    )
    t_mask = np.concatenate(all_ts)
    return predictions, ground_truth, t_mask


def write_to_file(dictionary: dict[str, Any], fname: str | Path) -> None:
    """Write a dictionary to a JSON file."""
    path = Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2)
    print(f"Written file {path}")
