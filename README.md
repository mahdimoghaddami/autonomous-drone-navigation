# Autonomous Drone Navigation (DroNet)

**Authors:** Mahdi Moghaddami and Emmanuel Musvaire  
**Course:** CSI 5170 Machine Learning, Oakland University  
**Reference:** Loquercio, Maqueda, Del Blanco, and Scaramuzza, *"DroNet: Learning to Fly by Driving"*, IEEE Robotics and Automation Letters (RA-L), 2018.

[Final Presentation (PDF)](docs/Project_Final_Group_13.pdf) | [Project Proposal (PDF)](docs/Project_Proposal_Group_13.pdf) | [Pretrained Weights](models/model_weights.h5)

## Overview

DroNet is a compact 8-layer residual convolutional neural network (`resnet8`) trained on ground-vehicle datasets (Udacity car steering angles and bicycle-mounted GoPro collision sequences) to predict two navigation outputs from a single onboard `200x200` grayscale camera frame:

1. **Steering angle** ($s_k$): Continuous regression head (`dense_1`) trained with hard-mined Mean Squared Error (MSE) to follow lanes and steer safely around obstacles.
2. **Collision probability** ($p_t \in [0, 1]$): Binary classification head (`dense_2` $\to$ `sigmoid`) trained with hard-mined Binary Cross-Entropy (BCE) to detect imminent obstacles.

During training, total loss combines both tasks with scheduled classification weight and exponential hard-mining decay after epoch 30:

$$L_{\text{tot}} = L_{\text{MSE}} + \max\left(0,\; 1 - e^{-0.1\,(\text{epoch} - 10)}\right)L_{\text{BCE}}$$

During flight control (`DroneController` and `ros/dronet_control`), raw predictions are smoothed with low-pass filters into forward velocity $v_k$ and yaw angle $\theta_k$:

$$v_k = (1 - \alpha)v_{k-1} + \alpha (1 - p_t)V_{\max}, \qquad \theta_k = (1 - \beta)\theta_{k-1} + \beta \frac{\pi}{2}s_k$$

## Reported Results

| Metric | Our Results | Paper Baseline (Loquercio et al., 2018) |
|---|---:|---:|
| Collision Average Accuracy | **95.3%** | 95.4% |
| Collision F1-Score | **0.895** | 0.901 |
| Steering RMSE | **0.181** | 0.179 |

- Confusion matrix: [docs/figures/confusion_matrix.png](docs/figures/confusion_matrix.png)
- Steering distribution: [docs/figures/steering_histogram.png](docs/figures/steering_histogram.png)

## Repository Structure

```text
.
├── src/autonomous_drone_navigation/
│   ├── __init__.py        # Public package exports (Keras 3 + PyTorch backend)
│   ├── cli.py             # `dronet` CLI entry point
│   ├── config.py          # Hyperparameters and path configuration (`DroneConfig`)
│   ├── control.py         # Low-pass filtered forward velocity & yaw controller
│   ├── data.py            # Augmentation (`DroneDataGenerator`) and batch iterator
│   ├── evaluation.py      # Regression & classification metrics + baselines
│   ├── logging_utils.py   # Tabular TSV logger & dynamic hard-mining callback
│   ├── losses.py          # Hard-mined MSE and BCE losses
│   ├── model.py           # ResNet-8 architecture and Keras 2/3 HDF5/JSON loaders
│   ├── plotting.py        # Loss curves, steering histograms, and confusion matrix
│   ├── preprocessing.py   # Frame cropping/resizing & Udacity timestamp matching
│   └── training.py        # End-to-end training loop
├── models/
│   ├── model_struct.json  # Legacy DroNet architecture definition
│   └── model_weights.h5   # Pretrained DroNet weights
├── ros/                   # Parrot Bebop ROS perception & C++ control packages
├── docs/                  # Course proposal, final presentation, and result plots
├── tests/                 # Automated unit & end-to-end pipeline tests
├── pyproject.toml         # uv project configuration
└── uv.lock                # Locked dependency graph
```

## Setup with `uv`

Requires **Python 3.13 or 3.14** and [uv](https://docs.astral.sh/uv/). The project uses **Keras 3** configured with the **PyTorch** backend (`KERAS_BACKEND=torch`), enabling full compatibility with modern Python and Windows/Linux without legacy TensorFlow 1.x or Python 2 dependencies.

```bash
uv sync --locked
```

Run the test suite to verify pretrained weight loading, preprocessing, training, evaluation, plotting, and control logic:

```bash
uv run pytest -v
```

## CLI Usage

Run pretrained inference and the UAV velocity/yaw controller on a synthetic demo frame or on custom images:

```bash
uv run dronet predict --demo
uv run dronet predict path/to/frame1.png path/to/frame2.png --output-json runs/predictions.json
```

Train from scratch (or resume with `--restore-model`) using directories organized as `<split>/<experiment>/{images/, sync_steering.txt | labels.txt}`:

```bash
uv run dronet train \
  --experiment-rootdir runs/exp1 \
  --train-dir path/to/training \
  --val-dir path/to/validation \
  --batch-size 32 \
  --epochs 100 \
  --log-rate 10
```

Evaluate on a test split, generate loss curves, and plot steering histograms / collision confusion matrix:

```bash
uv run dronet evaluate --experiment-rootdir runs/exp1 --test-dir path/to/testing
uv run dronet plot-loss --experiment-rootdir runs/exp1
uv run dronet plot-results --experiment-rootdir runs/exp1
```

Synchronize raw Udacity driving bag timestamps (`interpolated.csv` + `images/*.png` $\to$ `sync_steering.txt`):

```bash
uv run dronet match-timestamps path/to/udacity_bags
```

## ROS Deployment (Parrot Bebop)

The original ROS Catkin packages (`dronet_perception` and `dronet_control`) and Bebop configuration (`configs/outdoor.yaml`) are preserved in [ros/](ros/) with setup and launch instructions in [ros/README.md](ros/README.md).
