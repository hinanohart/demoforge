"""Intermediate representation shared by the engine, I/O and health layers.

These dataclasses are LeRobot-free: they hold NumPy arrays and plain metadata. The I/O
layer fills :class:`RawEpisode` from a dataset (or a mock); :func:`demoforge.retime_episode`
produces :class:`RetimeResult` objects and a :class:`DemoHealth` sidecar record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["RawEpisode", "RetimeConfig", "RetimeResult", "DemoHealth"]


@dataclass
class RawEpisode:
    """One recorded episode's action stream as samples of a geometric path."""

    actions: np.ndarray  # (T, n) joint positions
    timestamps: np.ndarray  # (T,) seconds, monotone
    fps: float
    episode_index: int = 0
    joint_names: list[str] = field(default_factory=list)
    action_is_pad: np.ndarray | None = None  # (T,) bool

    def unpadded(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (actions, timestamps) with padded frames removed."""
        if self.action_is_pad is None:
            return self.actions, self.timestamps
        keep = ~np.asarray(self.action_is_pad, dtype=bool)
        return self.actions[keep], self.timestamps[keep]

    @property
    def n_frames(self) -> int:
        return int(self.actions.shape[0])

    @property
    def duration(self) -> float:
        ts = self.timestamps
        return float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0


@dataclass(frozen=True)
class RetimeConfig:
    """Configuration for :func:`demoforge.retime_episode`."""

    mode: str = "keep_count"  # "keep_count" (exact positions, retime only) | "resample"
    backend: str = "topp"
    speeds: tuple[float, ...] = (1.0,)
    preserve_contact: str | None = "gripper"  # joint name substring, or None to disable
    target_fps: float | None = None  # resample mode only; default: input fps
    n_grid: int = 200
    path_tol: float = 0.02  # feasibility relative tolerance
    seed: int = 0  # reserved; the pipeline is deterministic and does not sample

    def __post_init__(self) -> None:
        if self.mode not in ("keep_count", "resample"):
            raise ValueError(f"mode must be 'keep_count' or 'resample', got {self.mode!r}")
        if self.backend not in ("topp", "numpy"):
            raise ValueError(f"backend must be 'topp' or 'numpy', got {self.backend!r}")
        if not self.speeds or any(s <= 0 for s in self.speeds):
            raise ValueError("speeds must be a non-empty tuple of positive floats")


@dataclass
class RetimeResult:
    """One emitted (re-timed) variant of an episode."""

    speed: float
    actions: np.ndarray  # (M, n)
    timestamps: np.ndarray  # (M,)
    fps: float
    source: str  # e.g. "synthetic:retime@1.0" or "real:retime@0.8"
    clamped: bool  # requested faster than feasible-optimal -> clamped
    outcome: str  # "retimed" | "infeasible_passthrough"

    @property
    def n_frames(self) -> int:
        return int(self.actions.shape[0])

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0


@dataclass
class DemoHealth:
    """Per-episode triage telemetry — the *transform* before/after delta.

    This is NOT a quality score and NOT a predictor of downstream training outcomes. It reports what the
    re-timing did, so a human can triage which demos to keep. There is deliberately no
    single aggregate number.
    """

    episode_index: int
    source: str
    backend: str
    frames_in: int
    frames_out: int
    fps_in: float
    fps_out: float
    duration_in: float
    duration_out: float
    limit_violations_before: int  # dynamic (vel/accel/jerk) — what re-timing controls
    limit_violations_after: int
    position_violations_before: int  # position-box (input-determined; re-timing cannot fix)
    position_violations_after: int
    max_vel_before: float
    max_vel_after: float
    max_acc_before: float
    max_acc_after: float
    max_jerk_before: float
    max_jerk_after: float
    path_deviation_max: float
    contact_segments_locked: int
    contact_segments: list[dict[str, Any]]
    retime_outcome: str
    data_integrity: dict[str, Any]
    triage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
