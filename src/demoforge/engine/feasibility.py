"""Feasibility checking on the *emitted discrete trajectory* (LeRobot-free).

Every claim demoforge makes about limits is measured here, the same way for the README
numbers, the property tests, and the health sidecar: finite-difference the sampled
trajectory and compare against the configured limits. There is no separate "theoretical"
feasibility path that could disagree with what is actually emitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .limits import RobotLimits

__all__ = ["FeasibilityReport", "check_feasible"]


@dataclass(frozen=True)
class FeasibilityReport:
    """Per-quantity maxima and violation counts on a sampled trajectory."""

    max_vel: float
    max_acc: float
    max_jerk: float
    max_pos_excess: float  # how far any joint exceeds its position box (0 if inside)
    n_vel_violations: int
    n_acc_violations: int
    n_jerk_violations: int
    n_pos_violations: int
    tol: float

    @property
    def feasible(self) -> bool:
        """All limits respected, including the (input-determined) position box."""
        return self.dynamic_feasible and self.n_pos_violations == 0

    @property
    def dynamic_feasible(self) -> bool:
        """Velocity/acceleration/jerk respected — the limits re-timing actually controls.

        Position-box compliance depends on the recorded joint values, which re-timing never
        changes, so it is reported separately and is not part of demoforge's re-timing claim.
        """
        return (
            self.n_vel_violations == 0
            and self.n_acc_violations == 0
            and self.n_jerk_violations == 0
        )

    @property
    def n_dynamic_violations(self) -> int:
        return self.n_vel_violations + self.n_acc_violations + self.n_jerk_violations

    @property
    def total_violations(self) -> int:
        return self.n_dynamic_violations + self.n_pos_violations


def check_feasible(
    t: np.ndarray, q: np.ndarray, limits: RobotLimits, *, tol: float = 0.02
) -> FeasibilityReport:
    """Finite-difference ``q(t)`` and count limit violations (relative tolerance ``tol``).

    A sample violates a limit if it exceeds ``limit * (1 + tol)``. Position is checked as a
    box; continuous joints are exempt from position checks. Needs at least 4 samples for a
    meaningful third derivative.
    """
    t = np.asarray(t, dtype=float)
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if q.shape[0] < 4:
        raise ValueError("need >= 4 samples to estimate jerk")
    if q.shape[1] != limits.n:
        raise ValueError(f"q has {q.shape[1]} joints but limits has {limits.n}")

    qd = np.gradient(q, t, axis=0, edge_order=2)
    qdd = np.gradient(qd, t, axis=0, edge_order=2)
    qddd = np.gradient(qdd, t, axis=0, edge_order=2)

    vel = np.abs(qd)
    acc = np.abs(qdd)
    jerk = np.abs(qddd)

    vlim = limits.vel[None, :] * (1 + tol)
    alim = limits.acc[None, :] * (1 + tol)
    jlim = limits.jerk[None, :] * (1 + tol)

    n_vel = int(np.any(vel > vlim, axis=1).sum())
    n_acc = int(np.any(acc > alim, axis=1).sum())
    n_jerk = int(np.any(jerk > jlim, axis=1).sum())

    # position box (continuous joints exempt)
    bounds = limits.pos_bounds  # (n, 2)
    cont = limits.continuous_mask
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-9)
    lo_excess = (bounds[None, :, 0] - q) / span[None, :]
    hi_excess = (q - bounds[None, :, 1]) / span[None, :]
    pos_excess = np.maximum(np.maximum(lo_excess, hi_excess), 0.0)
    pos_excess[:, cont] = 0.0
    n_pos = int(np.any(pos_excess > tol, axis=1).sum())

    return FeasibilityReport(
        max_vel=float(vel.max()),
        max_acc=float(acc.max()),
        max_jerk=float(jerk.max()),
        max_pos_excess=float(pos_excess.max()),
        n_vel_violations=n_vel,
        n_acc_violations=n_acc,
        n_jerk_violations=n_jerk,
        n_pos_violations=n_pos,
        tol=tol,
    )
