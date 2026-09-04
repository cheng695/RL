"""Check the cubic wheel action mapping used by WheelLeg."""

from __future__ import annotations

import math

import torch


SCALE = 30.0
BETA = 0.8


def cubic_shape(action: torch.Tensor, beta: float = BETA) -> torch.Tensor:
    action = torch.clamp(torch.tanh(action) / math.tanh(1.0), -1.0, 1.0)
    return (1.0 - beta) * action + beta * action**3


def main() -> None:
    samples = torch.tensor([-1.0, -0.8, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 0.8, 1.0])
    saturated = torch.clamp(torch.tanh(samples) / math.tanh(1.0), -1.0, 1.0)
    shaped = cubic_shape(samples)
    target = SCALE * shaped

    expected_shaped = (1.0 - BETA) * saturated + BETA * saturated**3
    expected_target = SCALE * expected_shaped
    if not torch.allclose(shaped, expected_shaped, atol=1.0e-5):
        raise AssertionError(f"Unexpected shaped actions: {shaped}")
    if not torch.allclose(target, expected_target, atol=1.0e-4):
        raise AssertionError(f"Unexpected targets: {target}")
    if not torch.allclose(cubic_shape(samples), -cubic_shape(-samples), atol=1.0e-6):
        raise AssertionError("Cubic wheel mapping is not odd symmetric.")

    dense = torch.linspace(-1.0, 1.0, 1001)
    if torch.any(torch.diff(cubic_shape(dense)) < -1.0e-7):
        raise AssertionError("Cubic wheel mapping is not monotonic on [-1, 1].")

    print("raw_a   tanh_norm shaped     target(rad/s)")
    for a, sat, s, t in zip(samples, saturated, shaped, target):
        print(f"{a.item():+5.1f}  {sat.item():+8.4f}  {s.item():+8.4f}  {t.item():+10.3f}")


if __name__ == "__main__":
    main()
