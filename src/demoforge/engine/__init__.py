"""demoforge.engine — the LeRobot-free trajectory core.

Nothing in this subpackage imports LeRobot, torch, or pyarrow. It operates purely on
NumPy arrays of joint positions and a :class:`RobotLimits` contract, so it can be reused
standalone (e.g. by other robotics tools) without pulling in dataset machinery.

Pipeline: :func:`extract_path` -> :func:`detect_contact_segments` -> :func:`parameterize`
-> :func:`preserve_contact` -> :func:`bound_jerk` -> :func:`sample_trajectory` ->
:func:`check_feasible`.
"""

from __future__ import annotations

from .feasibility import FeasibilityReport, check_feasible
from .jerk import bound_jerk, enforce_feasibility, min_jerk_trajectory
from .limits import JointLimits, RobotLimits, list_presets
from .path import (
    ContactSegment,
    PathSpline,
    detect_contact_frames,
    detect_contact_segments,
    extract_path,
)
from .retime import TimeLaw, parameterize, preserve_contact, sample_trajectory

__all__ = [
    "JointLimits",
    "RobotLimits",
    "list_presets",
    "PathSpline",
    "ContactSegment",
    "extract_path",
    "detect_contact_segments",
    "detect_contact_frames",
    "TimeLaw",
    "parameterize",
    "preserve_contact",
    "sample_trajectory",
    "bound_jerk",
    "enforce_feasibility",
    "min_jerk_trajectory",
    "FeasibilityReport",
    "check_feasible",
]
