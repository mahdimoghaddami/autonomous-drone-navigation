"""Unit and integration tests for the recreated DroNet package."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from autonomous_drone_navigation.cli import main
from autonomous_drone_navigation.config import DroneConfig
from autonomous_drone_navigation.control import DroneControlConfig, DroneController
from autonomous_drone_navigation.evaluation import run_evaluation
from autonomous_drone_navigation.model import (
    json_to_model,
    load_model_weights,
    model_to_json,
    resnet8,
    save_model_weights,
)
from autonomous_drone_navigation.plotting import plot_loss, plot_results
from autonomous_drone_navigation.preprocessing import (
    central_image_crop,
    load_img,
    match_timestamps,
    preprocess_frame,
)
from autonomous_drone_navigation.training import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_JSON = REPO_ROOT / "models" / "model_struct.json"
PRETRAINED_WEIGHTS = REPO_ROOT / "models" / "model_weights.h5"


def _create_synthetic_dataset(root: Path, n_frames: int = 4) -> None:
    """Create a synthetic dataset with one steering experiment and one collision experiment."""
    steer_exp = root / "exp_steering" / "images"
    steer_exp.mkdir(parents=True, exist_ok=True)
    steer_lines = ["angle,torque,speed,lat,long,alt"]
    for idx in range(n_frames):
        frame = np.full((240, 320, 3), fill_value=40 + idx * 30, dtype=np.uint8)
        cv2.imwrite(str(steer_exp / f"frame_{idx:04d}.png"), frame)
        steer_lines.append(f"{0.05 * (idx - 1):.4f},0.0,1.0,0.0,0.0,0.0")
    (root / "exp_steering" / "sync_steering.txt").write_text("\n".join(steer_lines) + "\n", encoding="utf-8")

    coll_exp = root / "exp_collision" / "images"
    coll_exp.mkdir(parents=True, exist_ok=True)
    coll_lines = []
    for idx in range(n_frames):
        frame = np.full((240, 320, 3), fill_value=200 - idx * 40, dtype=np.uint8)
        cv2.imwrite(str(coll_exp / f"frame_{idx:04d}.png"), frame)
        coll_lines.append(str(idx % 2))
    (root / "exp_collision" / "labels.txt").write_text("\n".join(coll_lines) + "\n", encoding="utf-8")


def test_pretrained_model_loads_exact_weights_and_infers() -> None:
    model = json_to_model(PRETRAINED_JSON)
    load_model_weights(model, PRETRAINED_WEIGHTS)

    with h5py.File(PRETRAINED_WEIGHTS, "r") as f:
        for layer_name in ["conv2d_1", "conv2d_3", "conv2d_4", "dense_1", "dense_2"]:
            model_w = model.get_layer(layer_name).get_weights()[0]
            h5_w = f[layer_name][layer_name]["kernel:0"][:]
            assert np.allclose(model_w, h5_w)

    batch = np.zeros((2, 200, 200, 1), dtype=np.float32)
    steer, coll = model.predict_on_batch(batch)
    assert steer.shape == (2, 1)
    assert coll.shape == (2, 1)
    assert np.all((coll >= 0.0) & (coll <= 1.0))


def test_weight_and_json_roundtrip(tmp_path: Path) -> None:
    model = resnet8(img_width=200, img_height=200, img_channels=1, output_dim=1)
    json_path = tmp_path / "model_struct.json"
    model_to_json(model, json_path)
    reloaded = json_to_model(json_path)

    h5_path = tmp_path / "weights.h5"
    weights_h5_path = tmp_path / "weights.weights.h5"
    save_model_weights(model, h5_path)
    save_model_weights(model, weights_h5_path)

    load_model_weights(reloaded, h5_path)
    assert np.allclose(
        model.get_layer("conv2d_1").get_weights()[0],
        reloaded.get_layer("conv2d_1").get_weights()[0],
    )
    load_model_weights(reloaded, weights_h5_path)
    assert np.allclose(
        model.get_layer("dense_1").get_weights()[0],
        reloaded.get_layer("dense_1").get_weights()[0],
    )


def test_image_preprocessing_and_timestamp_matching(tmp_path: Path) -> None:
    img = np.arange(240 * 320, dtype=np.uint8).reshape(240, 320) % 255
    cropped = central_image_crop(img, crop_width=200, crop_height=200)
    assert cropped.shape == (200, 200)

    processed = preprocess_frame(
        img, target_size=(320, 240), crop_size=(200, 200), grayscale=True, rescale=1.0 / 255.0
    )
    assert processed.shape == (200, 200, 1)
    assert processed.dtype == np.float32
    assert np.all((processed >= 0.0) & (processed <= 1.0))

    img_path = tmp_path / "sample.png"
    cv2.imwrite(str(img_path), img)
    loaded = load_img(img_path, grayscale=True, target_size=(320, 240), crop_size=(200, 200))
    assert loaded.shape == (200, 200, 1)

    bag_dir = tmp_path / "udacity" / "bag_01"
    (bag_dir / "images").mkdir(parents=True, exist_ok=True)
    for ts in [1000, 2000]:
        cv2.imwrite(str(bag_dir / "images" / f"{ts}.png"), img)
    csv_content = "\n".join(
        [
            "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt",
            "0,1000,320,240,center,1000.png,-0.12,0.1,5.0,0.0,0.0,0.0",
            "1,1500,320,240,center,1500.png,0.00,0.0,5.0,0.0,0.0,0.0",
            "2,2000,320,240,center,2000.png,0.24,-0.2,5.5,0.0,0.0,0.0",
        ]
    )
    (bag_dir / "interpolated.csv").write_text(csv_content + "\n", encoding="utf-8")

    written = match_timestamps(tmp_path / "udacity")
    assert len(written) == 1
    synced = np.loadtxt(written[0], delimiter=",", skiprows=1)
    assert synced.shape == (2, 6)
    assert np.allclose(synced[:, 0], [-0.12, 0.24])


def test_training_evaluation_and_plotting_pipeline(tmp_path: Path) -> None:
    train_dir = tmp_path / "training"
    val_dir = tmp_path / "validation"
    test_dir = tmp_path / "testing"
    exp_dir = tmp_path / "experiment"

    _create_synthetic_dataset(train_dir, n_frames=4)
    _create_synthetic_dataset(val_dir, n_frames=4)
    _create_synthetic_dataset(test_dir, n_frames=4)

    config = DroneConfig(
        batch_size=4,
        epochs=2,
        log_rate=1,
        experiment_rootdir=str(exp_dir),
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        test_dir=str(test_dir),
    )

    run_training(config)
    assert (exp_dir / "log.txt").exists()
    assert (exp_dir / "model_struct.json").exists()
    assert (exp_dir / "model_weights.h5").exists()

    eval_results = run_evaluation(config)
    assert "test_regression.json" in eval_results
    assert "test_classification.json" in eval_results
    assert (exp_dir / "predicted_and_real_steerings.json").exists()
    assert (exp_dir / "predicted_and_real_labels.json").exists()

    loss_fig = plot_loss(exp_dir, show=False)
    res_figs = plot_results(exp_dir, normalize=True)
    assert loss_fig.exists() and loss_fig.stat().st_size > 0
    assert res_figs["histograms"].exists() and res_figs["histograms"].stat().st_size > 0
    assert res_figs["confusion"].exists() and res_figs["confusion"].stat().st_size > 0


def test_drone_controller_and_cli_demo(tmp_path: Path) -> None:
    controller = DroneController(
        DroneControlConfig(
            alpha_velocity=0.7,
            alpha_yaw=0.5,
            max_forward_index=0.4,
            critical_prob_coll=0.7,
        )
    )
    cmd_clear = controller.step(steering_angle=0.5, collision_prob=0.0)
    assert np.isclose(cmd_clear.desired_forward_velocity, 0.4)
    assert cmd_clear.desired_angular_velocity > 0.0
    assert cmd_clear.desired_yaw_rad > 0.0

    cmd_stop_1 = controller.step(steering_angle=-0.2, collision_prob=1.0)
    assert cmd_stop_1.desired_forward_velocity < cmd_clear.desired_forward_velocity
    cmd_stop_2 = controller.step(steering_angle=-0.2, collision_prob=1.0)
    assert cmd_stop_2.desired_forward_velocity == 0.0

    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    out_json = tmp_path / "demo_pred.json"
    exit_code = main(["predict", "--demo", "--output-json", str(out_json)])
    assert exit_code == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert "steering_angle" in data[0]
    assert 0.0 <= data[0]["collision_prob"] <= 1.0
