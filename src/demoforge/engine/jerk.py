"""Post-hoc feasibility enforcement and the minimum-jerk primitive (LeRobot-free).

TOPP-RA produces a velocity/acceleration-optimal time-law, but the *finite-difference*
derivatives of the trajectory actually emitted (after smooth resampling at a target fps) can
slightly exceed limits — TOPP-RA enforces its constraints at grid/collocation points and has
no native jerk constraint at all. :func:`enforce_feasibility` closes that gap by **time
dilation** measured on the emitted trajectory itself:

* a sample exceeding a limit is slowed by a factor ``d`` where, under time scaling, velocity
  shrinks as ``1/d``, acceleration as ``1/d**2`` and jerk as ``1/d**3``;
* the per-sample requirement is therefore ``max(vel_ratio, sqrt(acc_ratio), cbrt(jerk_ratio))``;
* a few local-dilation passes remove most overshoot, then a **global** dilation by the worst
  remaining factor *guarantees* every sample is within limits (dilation only ever slows down,
  so it can never create a new violation).

demoforge thus never claims a jerk-constrained TOPP-RA — it claims a measured, feasible emitted
trajectory, verified the same way by :func:`demoforge.engine.feasibility.check_feasible`.
"""

from __future__ import annotations

import numpy as np

from .limits import RobotLimits
from .path import PathSpline
from .retime import TimeLaw, sample_trajectory

__all__ = [
    "enforce_feasibility",
    "enforce_feasibility_discrete",
    "bound_jerk",
    "min_jerk_trajectory",
]


def _finite_diff_derivs(t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, ...]:
    """vel/accel/jerk by repeated np.gradient on a (near-)uniform time grid."""
    qd = np.gradient(q, t, axis=0, edge_order=2)
    qdd = np.gradient(qd, t, axis=0, edge_order=2)
    qddd = np.gradient(qdd, t, axis=0, edge_order=2)
    return qd, qdd, qddd


def _need_factor(
    t: np.ndarray, q: np.ndarray, limits: RobotLimits, margin: float, *, jerk_only: bool
) -> np.ndarray:
    """Per-sample time-dilation factor (>= 1) that would bring the sample within limits."""
    qd, qdd, qddd = _finite_diff_derivs(t, q)
    jr = np.max(np.abs(qddd) / limits.jerk[None, :], axis=1)
    need = np.maximum(np.ones_like(jr), np.cbrt(jr))
    if not jerk_only:
        vr = np.max(np.abs(qd) / limits.vel[None, :], axis=1)
        ar = np.max(np.abs(qdd) / limits.acc[None, :], axis=1)
        need = np.maximum.reduce([need, vr, np.sqrt(ar)])
    over = need > 1.0
    need[over] *= margin
    return need


def _apply_local(law: TimeLaw, s_samples: np.ndarray, need: np.ndarray) -> TimeLaw:
    """Stretch time where ``need > 1``, mapped from sample arc-lengths onto law knots."""
    order = np.argsort(s_samples)
    factor_knot = np.interp(law.s_knots, s_samples[order], need[order], left=1.0, right=1.0)
    factor_knot = np.maximum(factor_knot, 1.0)
    dt = np.diff(law.t_knots)
    seg_factor = np.maximum(factor_knot[:-1], factor_knot[1:])
    t_new = np.concatenate([[0.0], np.cumsum(dt * seg_factor)])
    return law.with_t_knots(t_new)


def _enforce(
    path: PathSpline,
    time_law: TimeLaw,
    limits: RobotLimits,
    fps: float,
    *,
    jerk_only: bool,
    max_iters: int,
    margin: float,
) -> TimeLaw:
    law = time_law
    local_budget = max(1, max_iters // 2)
    for it in range(max_iters):
        t, q = sample_trajectory(path, law, fps)
        if len(t) < 5:
            break
        need = _need_factor(t, q, limits, margin, jerk_only=jerk_only)
        worst = float(np.max(need))
        if worst <= 1.0 + 1e-9:
            break
        if it < local_budget:
            law = _apply_local(law, law.s_at(t), need)
        else:
            # global guarantee: one uniform stretch by the worst factor fixes every sample
            law = law.scaled(worst)
    return law


def enforce_feasibility(
    path: PathSpline,
    time_law: TimeLaw,
    limits: RobotLimits,
    fps: float,
    *,
    max_iters: int = 12,
    margin: float = 1.05,
) -> TimeLaw:
    """Return a time-law whose emitted-at-``fps`` velocity, acceleration AND jerk are feasible.

    Velocity/acceleration feasibility from the parameterization plus jerk bounding are unified
    here and *guaranteed* on the emitted discrete trajectory (global-dilation backstop).
    """
    return _enforce(
        path, time_law, limits, fps, jerk_only=False, max_iters=max_iters, margin=margin
    )


def bound_jerk(
    path: PathSpline,
    time_law: TimeLaw,
    limits: RobotLimits,
    fps: float,
    *,
    max_iters: int = 8,
    margin: float = 1.05,
) -> TimeLaw:
    """Jerk-only specialisation of :func:`enforce_feasibility` (kept as an engine primitive)."""
    return _enforce(path, time_law, limits, fps, jerk_only=True, max_iters=max_iters, margin=margin)


def enforce_feasibility_discrete(
    t: np.ndarray,
    q: np.ndarray,
    limits: RobotLimits,
    *,
    max_iters: int = 40,
    margin: float = 1.05,
) -> np.ndarray:
    """Re-timestamp a *fixed* sequence of positions ``q`` so its finite-difference velocity,
    acceleration and jerk are feasible. Positions are never changed — only the inter-frame
    times are dilated (so the recorded path, video and observations are preserved exactly).

    Returns new monotone timestamps starting at 0. Local dilation for most iterations, then a
    global uniform stretch guarantees feasibility (dilation only ever slows down).
    """
    t = np.asarray(t, dtype=float).copy()
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if len(t) < 5:
        return t - t[0] if len(t) else t
    t = t - t[0]
    local_budget = max(1, (max_iters * 3) // 4)
    for it in range(max_iters):
        need = _need_factor(t, q, limits, margin, jerk_only=False)
        worst = float(np.max(need))
        if worst <= 1.0 + 1e-9:
            break
        if it < local_budget:
            dt = np.diff(t)
            seg_factor = np.maximum(need[:-1], need[1:])
            t = np.concatenate([[0.0], np.cumsum(dt * seg_factor)])
        else:
            t = t * worst
    return t


def min_jerk_trajectory(
    q0: np.ndarray, q1: np.ndarray, duration: float, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-jerk point-to-point trajectory (Flash & Hogan, J. Neurosci. 1985).

    ``q(tau) = q0 + (q1 - q0) * (10 tau^3 - 15 tau^4 + 6 tau^5)``, ``tau = t / duration``.
    Zero velocity/acceleration at both ends. Exported as an engine primitive (used for
    blends and as a reference in tests). Returns ``(t, q)`` sampled at ``fps``.
    """
    q0 = np.atleast_1d(np.asarray(q0, dtype=float))
    q1 = np.atleast_1d(np.asarray(q1, dtype=float))
    if q0.shape != q1.shape:
        raise ValueError("q0 and q1 must have the same shape")
    if duration <= 0 or fps <= 0:
        raise ValueError("duration and fps must be > 0")
    n = max(2, int(round(duration * fps)) + 1)
    t = np.linspace(0.0, duration, n)
    tau = t / duration
    profile = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    q = q0[None, :] + (q1 - q0)[None, :] * profile[:, None]
    return t, q
