"""Command-line interface for DroNet training, evaluation, plotting, and inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autonomous_drone_navigation.config import DroneConfig
from autonomous_drone_navigation.control import DroneControlConfig, DroneController
from autonomous_drone_navigation.evaluation import run_evaluation
from autonomous_drone_navigation.model import get_model, json_to_model, load_model_weights
from autonomous_drone_navigation.plotting import plot_loss, plot_results
from autonomous_drone_navigation.preprocessing import load_img, match_timestamps, preprocess_frame
from autonomous_drone_navigation.training import run_training


def _default_repo_model_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root / "models" / "model_struct.json",
        repo_root / "models" / "model_weights.h5",
    )


def _add_shared_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--img-width", "--img_width", dest="img_width", type=int, default=320)
    parser.add_argument("--img-height", "--img_height", dest="img_height", type=int, default=240)
    parser.add_argument(
        "--crop-img-width", "--crop_img_width", dest="crop_img_width", type=int, default=200
    )
    parser.add_argument(
        "--crop-img-height", "--crop_img_height", dest="crop_img_height", type=int, default=200
    )
    parser.add_argument(
        "--img-mode", "--img_mode", dest="img_mode", choices=["grayscale", "rgb"], default="grayscale"
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument(
        "--experiment-rootdir",
        "--experiment_rootdir",
        dest="experiment_rootdir",
        default="./model",
    )
    parser.add_argument(
        "--weights-fname",
        "--weights_fname",
        dest="weights_fname",
        default="model_weights.h5",
    )
    parser.add_argument(
        "--json-model-fname",
        "--json_model_fname",
        dest="json_model_fname",
        default="model_struct.json",
    )


def _config_from_args(args: argparse.Namespace) -> DroneConfig:
    return DroneConfig(
        img_width=getattr(args, "img_width", 320),
        img_height=getattr(args, "img_height", 240),
        crop_img_width=getattr(args, "crop_img_width", 200),
        crop_img_height=getattr(args, "crop_img_height", 200),
        img_mode=getattr(args, "img_mode", "grayscale"),
        batch_size=getattr(args, "batch_size", 32),
        epochs=getattr(args, "epochs", 100),
        log_rate=getattr(args, "log_rate", 10),
        initial_epoch=getattr(args, "initial_epoch", 0),
        experiment_rootdir=getattr(args, "experiment_rootdir", "./model"),
        train_dir=getattr(args, "train_dir", "../training"),
        val_dir=getattr(args, "val_dir", "../validation"),
        test_dir=getattr(args, "test_dir", "../testing"),
        restore_model=getattr(args, "restore_model", False),
        weights_fname=getattr(args, "weights_fname", "model_weights.h5"),
        json_model_fname=getattr(args, "json_model_fname", "model_struct.json"),
    )


def _build_demo_frame(height: int = 240, width: int = 320) -> np.ndarray:
    """Create a deterministic synthetic corridor-like frame for `--demo` inference."""
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    x = np.abs(np.linspace(-1.0, 1.0, width, dtype=np.float32))[None, :]
    gray = np.clip(0.55 * y + 110.0 * x, 0, 255).astype(np.uint8)
    return gray


def _run_predict(args: argparse.Namespace) -> int:
    default_json, default_weights = _default_repo_model_paths()
    json_path = Path(args.model_json) if args.model_json else default_json
    weights_path = Path(args.weights) if args.weights else default_weights

    if json_path.exists():
        model = json_to_model(json_path)
        if weights_path.exists():
            load_model_weights(model, weights_path)
    else:
        channels = 3 if args.img_mode == "rgb" else 1
        model = get_model(
            img_width=args.crop_img_width,
            img_height=args.crop_img_height,
            img_channels=channels,
            output_dim=1,
            weights_path=weights_path if weights_path.exists() else None,
        )

    controller = DroneController(
        DroneControlConfig(
            alpha_velocity=args.alpha_velocity,
            alpha_yaw=args.alpha_yaw,
            max_forward_index=args.max_forward_index,
            critical_prob_coll=args.critical_prob,
        )
    )

    grayscale = args.img_mode == "grayscale"
    target_size = (args.img_width, args.img_height)
    crop_size = (args.crop_img_width, args.crop_img_height)

    frames: list[tuple[str, np.ndarray]] = []
    if args.demo or not args.images:
        demo_raw = _build_demo_frame(args.img_height, args.img_width)
        frames.append(
            (
                "synthetic_demo_frame",
                preprocess_frame(
                    demo_raw,
                    target_size=target_size,
                    crop_size=crop_size,
                    grayscale=grayscale,
                    rescale=1.0 / 255.0,
                ),
            )
        )
    else:
        for img_path in args.images:
            arr = load_img(
                img_path,
                grayscale=grayscale,
                target_size=target_size,
                crop_size=crop_size,
            )
            arr = arr * np.float32(1.0 / 255.0)
            frames.append((str(img_path), arr))

    records: list[dict] = []
    for label, tensor in frames:
        outs = model.predict_on_batch(tensor[None, ...])
        steer = float(np.asarray(outs[0]).reshape(-1)[0])
        coll = float(np.asarray(outs[1]).reshape(-1)[0])
        cmd = controller.step(steer, coll)
        record = {"input": label, **asdict(cmd)}
        records.append(record)
        print(
            f"[{label}] steering={cmd.steering_angle:+.4f}, "
            f"collision_prob={cmd.collision_prob:.4f}, "
            f"forward_vel={cmd.desired_forward_velocity:.4f}, "
            f"angular_vel={cmd.desired_angular_velocity:+.4f}, "
            f"yaw_rad={cmd.desired_yaw_rad:+.4f}"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved prediction results to {out_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dronet",
        description="DroNet: Learning to Fly by Driving (CNN training, evaluation, plotting, and inference)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # predict
    predict_p = subparsers.add_parser(
        "predict",
        help="Run DroNet inference and velocity/yaw control calculation on image(s) or a demo frame",
    )
    predict_p.add_argument("images", nargs="*", help="Input image path(s)")
    predict_p.add_argument("--demo", action="store_true", help="Run inference on a synthetic demo frame")
    predict_p.add_argument("--weights", default=None, help="Path to HDF5 weights (defaults to models/model_weights.h5)")
    predict_p.add_argument("--model-json", default=None, help="Path to model JSON (defaults to models/model_struct.json)")
    predict_p.add_argument("--output-json", default=None, help="Optional JSON path to save predictions")
    predict_p.add_argument("--alpha-velocity", type=float, default=0.7)
    predict_p.add_argument("--alpha-yaw", type=float, default=0.5)
    predict_p.add_argument("--max-forward-index", type=float, default=0.4)
    predict_p.add_argument("--critical-prob", type=float, default=0.7)
    _add_shared_config_args(predict_p)

    # train
    train_p = subparsers.add_parser("train", help="Train DroNet on steering and collision datasets")
    _add_shared_config_args(train_p)
    train_p.add_argument("--train-dir", "--train_dir", dest="train_dir", default="../training")
    train_p.add_argument("--val-dir", "--val_dir", dest="val_dir", default="../validation")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--log-rate", "--log_rate", dest="log_rate", type=int, default=10)
    train_p.add_argument("--initial-epoch", "--initial_epoch", dest="initial_epoch", type=int, default=0)
    train_p.add_argument("--restore-model", "--restore_model", dest="restore_model", action="store_true")

    # evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluate trained DroNet on a test directory")
    _add_shared_config_args(eval_p)
    eval_p.add_argument("--test-dir", "--test_dir", dest="test_dir", default="../testing")

    # plot-loss
    loss_p = subparsers.add_parser("plot-loss", help="Plot training/validation loss curves from log.txt")
    loss_p.add_argument(
        "--experiment-rootdir",
        "--experiment_rootdir",
        dest="experiment_rootdir",
        default="./model",
    )
    loss_p.add_argument("--show", action="store_true", help="Display the figure interactively")

    # plot-results
    res_p = subparsers.add_parser(
        "plot-results", help="Plot steering histograms and collision confusion matrix"
    )
    res_p.add_argument(
        "--experiment-rootdir",
        "--experiment_rootdir",
        dest="experiment_rootdir",
        default="./model",
    )
    res_p.add_argument("--no-normalize", action="store_true", help="Plot raw confusion counts")

    # match-timestamps
    ts_p = subparsers.add_parser(
        "match-timestamps",
        help="Match Udacity interpolated.csv timestamps with image frames into sync_steering.txt",
    )
    ts_p.add_argument("folder", help="Root folder containing Udacity bag experiment subdirectories")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        return _run_predict(args)
    if args.command == "train":
        run_training(_config_from_args(args))
        return 0
    if args.command == "evaluate":
        run_evaluation(_config_from_args(args))
        return 0
    if args.command == "plot-loss":
        out = plot_loss(args.experiment_rootdir, show=args.show)
        print(f"Saved loss plot to {out}")
        return 0
    if args.command == "plot-results":
        outputs = plot_results(args.experiment_rootdir, normalize=not args.no_normalize)
        for kind, path in outputs.items():
            print(f"Saved {kind} plot to {path}")
        return 0
    if args.command == "match-timestamps":
        written = match_timestamps(args.folder)
        print(f"Generated {len(written)} sync_steering.txt file(s).")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
