"""Pure-Python port of the DroNet deep_navigation velocity and yaw controller."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DroneControlConfig:
    """Parameters matching `dronet_control/deep_navigation` and the DroNet paper."""

    alpha_velocity: float = 0.7
    alpha_yaw: float = 0.5
    max_forward_index: float = 0.4
    critical_prob_coll: float = 0.7


@dataclass
class ControlCommand:
    """Filtered forward velocity and yaw command produced from a DroNet prediction."""

    steering_angle: float
    collision_prob: float
    desired_forward_velocity: float
    desired_angular_velocity: float
    desired_yaw_rad: float


class DroneController:
    """Low-pass filtered UAV forward velocity and steering angle controller."""

    def __init__(self, config: DroneControlConfig | None = None) -> None:
        self.config = config or DroneControlConfig()
        self.desired_forward_velocity = float(self.config.max_forward_index)
        self.desired_angular_velocity = 0.0
        self.desired_yaw_rad = 0.0

    def reset(self) -> None:
        self.desired_forward_velocity = float(self.config.max_forward_index)
        self.desired_angular_velocity = 0.0
        self.desired_yaw_rad = 0.0

    def step(self, steering_angle: float, collision_prob: float) -> ControlCommand:
        """Compute next filtered forward velocity and angular/yaw command."""
        clipped_steer = max(-1.0, min(1.0, float(steering_angle)))
        clipped_coll = max(0.0, min(1.0, float(collision_prob)))

        desired_forward_m = max(0.0, (1.0 - clipped_coll) * self.config.max_forward_index)

        alpha_v = self.config.alpha_velocity
        self.desired_forward_velocity = (
            (1.0 - alpha_v) * self.desired_forward_velocity + alpha_v * desired_forward_m
        )

        if self.desired_forward_velocity < (
            (1.0 - self.config.critical_prob_coll) * self.config.max_forward_index
        ):
            self.desired_forward_velocity = 0.0

        alpha_y = self.config.alpha_yaw
        self.desired_angular_velocity = (
            (1.0 - alpha_y) * self.desired_angular_velocity + alpha_y * clipped_steer
        )
        self.desired_yaw_rad = (
            (1.0 - alpha_y) * self.desired_yaw_rad + alpha_y * (math.pi / 2.0) * clipped_steer
        )

        return ControlCommand(
            steering_angle=clipped_steer,
            collision_prob=clipped_coll,
            desired_forward_velocity=self.desired_forward_velocity,
            desired_angular_velocity=self.desired_angular_velocity,
            desired_yaw_rad=self.desired_yaw_rad,
        )
