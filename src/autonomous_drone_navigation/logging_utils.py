"""Training callbacks, tabular TSV logging, and checkpoint saving."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import TextIO

import keras
import numpy as np

from autonomous_drone_navigation.losses import HardMiningState
from autonomous_drone_navigation.model import save_model_weights


class TabularLogger:
    """Tab-separated training logger writing `<output_dir>/log.txt`."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "log.txt"
        self.output_file: TextIO = open(self.log_path, "w", encoding="utf-8")
        atexit.register(self.close)
        self.first_row = True
        self.headers: list[str] = []
        self.current_row: dict[str, float | str] = {}

    def close(self) -> None:
        if not self.output_file.closed:
            self.output_file.close()

    def log_tabular(self, key: str, val: float | str) -> None:
        if self.first_row:
            self.headers.append(key)
        elif key not in self.headers:
            raise KeyError(f"Key {key!r} was not included in the first iteration")
        if key in self.current_row:
            raise KeyError(f"Key {key!r} already set for this iteration")
        self.current_row[key] = val

    def dump_tabular(self) -> None:
        vals: list[float | str] = []
        print("-" * 37)
        for key in self.headers:
            val = self.current_row.get(key, "")
            valstr = f"{float(val):8.3g}" if isinstance(val, (int, float, np.floating)) else str(val)
            print(f"| {key:>15s} | {valstr:>15s} |")
            vals.append(val)
        print("-" * 37)

        if self.first_row:
            self.output_file.write("\t".join(self.headers) + "\n")
        self.output_file.write("\t".join(map(str, vals)) + "\n")
        self.output_file.flush()
        self.current_row.clear()
        self.first_row = False


class DroneTrainingCallback(keras.callbacks.Callback):
    """Dynamic loss-weight & hard-mining scheduler plus periodic checkpointing."""

    def __init__(
        self,
        filepath: str | Path,
        period: int,
        batch_size: int,
        state: HardMiningState,
        logger: TabularLogger | None = None,
    ) -> None:
        super().__init__()
        self.filepath = str(filepath)
        self.period = max(1, int(period))
        self.batch_size = int(batch_size)
        self.state = state
        self.logger = logger or TabularLogger(self.filepath)

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self.state.alpha = 1.0
        self.state.beta = float(np.maximum(0.0, 1.0 - np.exp(-1.0 / 10.0 * (epoch - 10))))

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        train_loss = float(logs.get("loss", np.nan))
        val_loss = float(logs.get("val_loss", np.nan))

        self.logger.log_tabular("train_loss", train_loss)
        self.logger.log_tabular("val_loss", val_loss)
        self.logger.dump_tabular()

        if (epoch + 1) % self.period == 0:
            filename = os.path.join(self.filepath, f"model_weights_{epoch}.h5")
            save_model_weights(self.model, filename)
            print(f"Saved model at {filename}")

        mse_k = self.batch_size - (self.batch_size - 10) * np.maximum(
            0.0, 1.0 - np.exp(-1.0 / 30.0 * (epoch - 30.0))
        )
        entropy_k = self.batch_size - (self.batch_size - 5) * np.maximum(
            0.0, 1.0 - np.exp(-1.0 / 30.0 * (epoch - 30.0))
        )
        self.state.k_mse = max(1, int(np.round(mse_k)))
        self.state.k_entropy = max(1, int(np.round(entropy_k)))

    def on_train_end(self, logs: dict | None = None) -> None:
        self.logger.close()


MyCallback = DroneTrainingCallback
