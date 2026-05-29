"""Shared fixtures: deterministic limits and synthetic episodes (no torch, no network)."""

from __future__ import annotations

import pytest

from demoforge.engine import RobotLimits
from demoforge.io import make_mock_episode


@pytest.fixture
def so101() -> RobotLimits:
    return RobotLimits.from_preset("so101")


@pytest.fixture
def jerky_episode(so101):
    """A position-clean but dynamically jerky episode (the case re-timing is meant to fix)."""
    return make_mock_episode(
        n_frames=80,
        n_joints=so101.n,
        fps=20.0,
        seed=0,
        joint_names=so101.names,
        pos_bounds=so101.pos_bounds,
    )
