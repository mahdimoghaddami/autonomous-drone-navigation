"""Image loading, cropping, frame preprocessing, and Udacity timestamp synchronization."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np


def central_image_crop(img: np.ndarray, crop_width: int = 150, crop_height: int = 150) -> np.ndarray:
    """Crop the input image centered horizontally and anchored at the bottom."""
    height, width = img.shape[0], img.shape[1]
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"Crop size ({crop_width}x{crop_height}) exceeds image dimensions ({width}x{height})."
        )
    half_width = width // 2
    left = half_width - (crop_width // 2)
    right = left + crop_width
    top = height - crop_height
    return img[top:height, left:right]


def preprocess_frame(
    img: np.ndarray,
    target_size: tuple[int, int] | None = (320, 240),
    crop_size: tuple[int, int] | None = (200, 200),
    grayscale: bool = True,
    rescale: float | None = 1.0 / 255.0,
) -> np.ndarray:
    """Preprocess an in-memory BGR or grayscale frame for DroNet inference."""
    frame = img
    if grayscale:
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]
    else:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        target_w, target_h = target_size
        if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h))

    if crop_size is not None:
        frame = central_image_crop(frame, crop_size[0], crop_size[1])

    if grayscale and frame.ndim == 2:
        frame = frame.reshape((frame.shape[0], frame.shape[1], 1))

    arr = np.asarray(frame, dtype=np.float32)
    if rescale is not None:
        arr = arr * np.float32(rescale)
    return arr


def load_img(
    path: str | Path,
    grayscale: bool = False,
    target_size: tuple[int, int] | None = None,
    crop_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load an image from disk, resize, bottom-center crop, and return as float32."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return preprocess_frame(
        img,
        target_size=target_size,
        crop_size=crop_size,
        grayscale=grayscale,
        rescale=None,
    )


def _extract_timestamps_from_csv(csv_path: Path) -> np.ndarray:
    """Load integer timestamps from column 1 of interpolated.csv."""
    stamps = np.loadtxt(csv_path, usecols=1, delimiter=",", skiprows=1, dtype=np.int64)
    return np.atleast_1d(stamps)


def _match_timestamp_indices(image_stamps: np.ndarray, steer_stamps: np.ndarray) -> np.ndarray:
    """Match each image timestamp to its exact (or nearest) steering CSV index."""
    indices: list[int] = []
    for stamp in image_stamps:
        dist = np.abs(stamp - steer_stamps)
        exact = np.where(dist == 0)[0]
        if len(exact) > 0:
            indices.append(int(exact[0]))
        else:
            indices.append(int(np.argmin(dist)))
    return np.asarray(indices, dtype=np.int64)


def match_timestamps(dataset_folder: str | Path) -> list[Path]:
    """Synchronize Udacity image filenames with `interpolated.csv` into `sync_steering.txt`."""
    folder = Path(dataset_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

    written: list[Path] = []
    for exp_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        image_paths = sorted(glob.glob(os.path.join(exp_dir, "images", "*.png")))
        if not image_paths:
            continue

        im_stamps = sorted(
            int(re.sub(r"\.png$", "", os.path.basename(p))) for p in image_paths
        )
        csv_path = exp_dir / "interpolated.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing interpolated.csv in {exp_dir}")

        steer_stamps = _extract_timestamps_from_csv(csv_path)
        match_idx = _match_timestamp_indices(np.asarray(im_stamps, dtype=np.int64), steer_stamps)

        mat = np.loadtxt(csv_path, usecols=(6, 7, 8, 9, 10, 11), skiprows=1, delimiter=",")
        if mat.ndim == 1:
            mat = mat[None, :]
        sync_steer = mat[match_idx, :]

        out_path = exp_dir / "sync_steering.txt"
        np.savetxt(
            out_path,
            sync_steer,
            delimiter=",",
            header="angle,torque,speed,lat,long,alt",
        )
        written.append(out_path)

    return written
