"""Geometric path extraction (LeRobot-free).

The recorded action stream is treated as samples of a *geometric path* in joint space.
We fit an **interpolating** (not smoothing) cubic spline over arc-length, so every
recorded waypoint is a knot: the spatial shape — including delicate contact micro-motion —
is preserved exactly. Re-timing then only changes how fast the path is traversed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = [
    "PathSpline",
    "ContactSegment",
    "extract_path",
    "detect_contact_segments",
    "detect_contact_frames",
]


@dataclass(frozen=True)
class PathSpline:
    """Interpolating cubic spline of joint positions over arc-length ``s in [0, L]``.

    ``waypoint_s`` maps each (de-padded, de-duplicated) input waypoint to its arc-length,
    so segment indices found on the raw signal can be expressed as arc-length intervals.
    """

    spline: CubicSpline
    length: float
    waypoint_s: np.ndarray  # (K,) arc-length of each retained waypoint
    n_joints: int
    continuous_mask: np.ndarray

    def q(self, s: np.ndarray) -> np.ndarray:
        return np.atleast_2d(self.spline(np.clip(s, 0.0, self.length)))

    def dqds(self, s: np.ndarray) -> np.ndarray:
        return np.atleast_2d(self.spline(np.clip(s, 0.0, self.length), 1))

    def d2qds2(self, s: np.ndarray) -> np.ndarray:
        return np.atleast_2d(self.spline(np.clip(s, 0.0, self.length), 2))


@dataclass(frozen=True)
class ContactSegment:
    """An arc-length interval to protect from temporal compression, with a reason."""

    s_start: float
    s_end: float
    reason: str  # "dwell" | "contact_joint"


def _dedupe(q: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop consecutive near-duplicate waypoints (degenerate zero-length segments).

    Returns the kept positions and the indices into the original array that were kept.
    The first and last waypoints are always kept.
    """
    keep = [0]
    last = q[0]
    for i in range(1, len(q)):
        if np.linalg.norm(q[i] - last) > eps:
            keep.append(i)
            last = q[i]
    if keep[-1] != len(q) - 1:
        keep.append(len(q) - 1)
    idx = np.array(keep, dtype=int)
    return q[idx], idx


def extract_path(
    positions: np.ndarray,
    continuous_mask: np.ndarray | None = None,
    *,
    dedupe_eps: float = 1e-6,
) -> PathSpline:
    """Build an interpolating arc-length cubic spline from a (T, n) action array.

    Continuous joints are unwrapped before measuring arc-length so a +pi/-pi wrap does
    not create a spurious long segment. Requires at least 2 distinct waypoints.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError(f"positions must be (T, n), got shape {positions.shape}")
    T, n = positions.shape
    if T < 2:
        raise ValueError("need at least 2 waypoints to form a path")
    if continuous_mask is None:
        continuous_mask = np.zeros(n, dtype=bool)
    continuous_mask = np.asarray(continuous_mask, dtype=bool)

    q = positions.copy()
    if continuous_mask.any():
        q[:, continuous_mask] = np.unwrap(q[:, continuous_mask], axis=0)

    q_kept, _ = _dedupe(q, dedupe_eps)
    if len(q_kept) < 2:
        raise ValueError("path collapses to a single point after de-duplication")

    seg = np.linalg.norm(np.diff(q_kept, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(s[-1])
    if length <= 0:
        raise ValueError("path has zero arc-length")

    spline = CubicSpline(s, q_kept, bc_type="natural", extrapolate=False)
    return PathSpline(
        spline=spline,
        length=length,
        waypoint_s=s,
        n_joints=n,
        continuous_mask=continuous_mask,
    )


def detect_contact_segments(
    positions: np.ndarray,
    timestamps: np.ndarray,
    path: PathSpline,
    *,
    contact_joint_indices: tuple[int, ...] = (),
    dwell_speed_frac: float = 0.12,
    contact_joint_speed_frac: float = 0.25,
) -> list[ContactSegment]:
    """Find segments to protect, on the *raw* signal (before any retiming).

    Two heuristics, both expressed as arc-length intervals:

    * **dwell** — joint-space speed drops below ``dwell_speed_frac`` of the median speed
      (a deliberate pause / fine alignment).
    * **contact_joint** — a designated joint (e.g. the gripper) is actively moving faster
      than ``contact_joint_speed_frac`` of its own peak speed (grasp / release).

    Detection runs on the de-padded raw waypoints; intervals are mapped to arc-length via
    the path's ``waypoint_s``. Overlapping intervals are merged.
    """
    positions = np.asarray(positions, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    T = len(positions)
    if T < 3:
        return []
    dt = np.diff(timestamps)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = step / dt  # (T-1,)
    # map raw index -> arc-length using a monotone interpolation onto path knots
    raw_s = np.concatenate([[0.0], np.cumsum(step)])
    if raw_s[-1] > 0:
        raw_s = raw_s / raw_s[-1] * path.length

    intervals: list[tuple[float, float, str]] = []

    med = np.median(speed[speed > 0]) if np.any(speed > 0) else 0.0
    if med > 0:
        dwell = speed < dwell_speed_frac * med
        for a, b in _runs(dwell):
            intervals.append((raw_s[a], raw_s[min(b + 1, T - 1)], "dwell"))

    for j in contact_joint_indices:
        if j < 0 or j >= positions.shape[1]:
            continue
        jspeed = np.abs(np.diff(positions[:, j])) / dt
        peak = jspeed.max()
        if peak > 0:
            active = jspeed > contact_joint_speed_frac * peak
            for a, b in _runs(active):
                intervals.append((raw_s[a], raw_s[min(b + 1, T - 1)], "contact_joint"))

    return _merge(intervals)


def detect_contact_frames(
    positions: np.ndarray,
    timestamps: np.ndarray,
    *,
    contact_joint_indices: tuple[int, ...] = (),
    dwell_speed_frac: float = 0.12,
    contact_joint_speed_frac: float = 0.25,
) -> list[tuple[int, int, str]]:
    """Same heuristics as :func:`detect_contact_segments`, but as *frame-index* ranges
    ``(start, end, reason)`` (inclusive). Frame indices survive zero-motion dwells, which
    collapse to a single point in arc-length space — needed for keep_count preservation.
    """
    positions = np.asarray(positions, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    T = len(positions)
    if T < 3:
        return []
    dt = np.diff(timestamps)
    pos_dt = dt[dt > 0]
    dt[dt <= 0] = np.median(pos_dt) if pos_dt.size else 1.0
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = step / dt
    out: list[tuple[int, int, str]] = []
    med = np.median(speed[speed > 0]) if np.any(speed > 0) else 0.0
    if med > 0:
        for a, b in _runs(speed < dwell_speed_frac * med):
            out.append((a, min(b + 1, T - 1), "dwell"))
    for j in contact_joint_indices:
        if 0 <= j < positions.shape[1]:
            jspeed = np.abs(np.diff(positions[:, j])) / dt
            peak = jspeed.max()
            if peak > 0:
                for a, b in _runs(jspeed > contact_joint_speed_frac * peak):
                    out.append((a, min(b + 1, T - 1), "contact_joint"))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [a, b] (inclusive) of consecutive True values."""
    runs = []
    a = None
    for i, v in enumerate(mask):
        if v and a is None:
            a = i
        elif not v and a is not None:
            runs.append((a, i - 1))
            a = None
    if a is not None:
        runs.append((a, len(mask) - 1))
    return runs


def _merge(intervals: list[tuple[float, float, str]]) -> list[ContactSegment]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: x[0])
    out: list[ContactSegment] = [ContactSegment(ordered[0][0], ordered[0][1], ordered[0][2])]
    for s0, s1, reason in ordered[1:]:
        cur = out[-1]
        if s0 <= cur.s_end:
            new_reason = cur.reason if reason in cur.reason else f"{cur.reason}+{reason}"
            out[-1] = ContactSegment(cur.s_start, max(cur.s_end, s1), new_reason)
        else:
            out.append(ContactSegment(s0, s1, reason))
    return out
