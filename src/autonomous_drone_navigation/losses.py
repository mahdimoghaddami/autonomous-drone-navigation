"""Hard-mining loss functions and dynamic scheduling state for DroNet."""

from __future__ import annotations

from dataclasses import dataclass

import keras
from keras import ops


@dataclass
class HardMiningState:
    """Mutable loss weights and hard-mining top-k counts shared across training callbacks."""

    alpha: float = 1.0
    beta: float = 0.0
    k_mse: int = 32
    k_entropy: int = 32


class HardMiningMSELoss(keras.losses.Loss):
    """Hard-mined Mean Squared Error computed exclusively on steering samples (t = 1)."""

    def __init__(self, state: HardMiningState, name: str = "hard_mining_mse") -> None:
        super().__init__(name=name)
        self.state = state

    def call(self, y_true, y_pred):
        t = ops.cast(y_true[:, 0], "float32")
        true_steer = ops.cast(y_true[:, 1], "float32")
        pred_steer = ops.squeeze(ops.cast(y_pred, "float32"), axis=-1)

        mask = ops.cast(ops.equal(t, 1.0), "float32")
        n_samples = ops.cast(ops.sum(mask), "int32")
        l_steer = mask * ops.square(pred_steer - true_steer)

        batch_n = ops.shape(t)[0]
        k_target = ops.cast(self.state.k_mse, "int32")
        k_val = ops.minimum(k_target, ops.maximum(n_samples, 1))
        k_val = ops.minimum(k_val, batch_n)

        top_vals, _ = ops.top_k(l_steer, k=k_val)
        hard_loss = ops.sum(top_vals) / ops.cast(ops.maximum(k_target, 1), "float32")
        weighted = hard_loss * ops.cast(self.state.alpha, "float32")
        return ops.where(ops.equal(n_samples, 0), 0.0, weighted)


class HardMiningEntropyLoss(keras.losses.Loss):
    """Hard-mined Binary Cross-Entropy computed exclusively on collision samples (t = 0)."""

    def __init__(self, state: HardMiningState, name: str = "hard_mining_entropy") -> None:
        super().__init__(name=name)
        self.state = state

    def call(self, y_true, y_pred):
        t = ops.cast(y_true[:, 0], "float32")
        true_coll = ops.cast(y_true[:, 1], "float32")
        pred_coll = ops.squeeze(ops.cast(y_pred, "float32"), axis=-1)

        mask = ops.cast(ops.equal(t, 0.0), "float32")
        n_samples = ops.cast(ops.sum(mask), "int32")

        pred_clipped = ops.clip(pred_coll, 1e-7, 1.0 - 1e-7)
        bce = -(true_coll * ops.log(pred_clipped) + (1.0 - true_coll) * ops.log(1.0 - pred_clipped))
        l_coll = mask * bce

        batch_n = ops.shape(t)[0]
        k_target = ops.cast(self.state.k_entropy, "int32")
        k_val = ops.minimum(k_target, ops.maximum(n_samples, 1))
        k_val = ops.minimum(k_val, batch_n)

        top_vals, _ = ops.top_k(l_coll, k=k_val)
        hard_loss = ops.sum(top_vals) / ops.cast(ops.maximum(k_target, 1), "float32")
        weighted = hard_loss * ops.cast(self.state.beta, "float32")
        return ops.where(ops.equal(n_samples, 0), 0.0, weighted)


def hard_mining_mse(state: HardMiningState) -> HardMiningMSELoss:
    """Factory wrapper matching legacy `hard_mining_mse` signature."""
    return HardMiningMSELoss(state)


def hard_mining_entropy(state: HardMiningState) -> HardMiningEntropyLoss:
    """Factory wrapper matching legacy `hard_mining_entropy` signature."""
    return HardMiningEntropyLoss(state)
