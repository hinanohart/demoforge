"""Time-law re-derivation (LeRobot-free).

Given a fixed geometric :class:`~demoforge.engine.path.PathSpline`, compute a new monotone
arc-length-vs-time law ``s(t)`` that respects velocity and acceleration limits. Two backends:

* ``"topp"`` — TOPP-RA (Pham & Pham, IEEE T-RO 2018) via the ``toppra`` library (MIT).
  Time-optimal under velocity + acceleration constraints. **Default.**
* ``"numpy"`` — dependency-light fallback: constant path-speed bounded by velocity and
  centripetal-acceleration limits, with minimum-jerk ramps at the ends. Feasible but not
  time-optimal.

TOPP-RA has **no native jerk constraint**; jerk is bounded by a separate post-hoc pass
(:mod:`demoforge.engine.jerk`) and verified on the emitted discrete trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .limits import RobotLimits
from .path import PathSpline

__all__ = ["TimeLaw", "parameterize", "sample_trajectory", "preserve_contact"]


@dataclass(frozen=True)
class TimeLaw:
    """A monotone arc-length-vs-time law as breakpoints ``(t_knots, s_knots)``.

    ``backend`` records which solver produced it. Warps (contact preservation, jerk
    bounding) return new ``TimeLaw`` objects with the same ``s_knots`` but stretched
    ``t_knots`` (dilation only ever slows down, so feasibility is preserved).
    """

    t_knots: np.ndarray
    s_knots: np.ndarray
    backend: str

    @property
    def duration(self) -> float:
        return float(self.t_knots[-1])

    @property
    def length(self) -> float:
        return float(self.s_knots[-1])

    def scaled(self, factor: float) -> TimeLaw:
        """Uniformly stretch time by ``factor`` (>= 1 slows down, preserving feasibility)."""
        if factor <= 0:
            raise ValueError("scale factor must be > 0")
        return TimeLaw(self.t_knots * factor, self.s_knots.copy(), self.backend)

    def with_t_knots(self, t_knots: np.ndarray) -> TimeLaw:
        return TimeLaw(np.asarray(t_knots, dtype=float), self.s_knots.copy(), self.backend)

    def s_at(self, t: np.ndarray) -> np.ndarray:
        """Arc-length at the given (clamped) times via monotone C1 (PCHIP) interpolation.

        PCHIP keeps ``s(t)`` monotone with no overshoot while being C1, so the reconstructed
        ``q(t) = path.q(s(t))`` has no piecewise-constant-velocity steps (which linear
        interpolation of the time-law would inject as spurious acceleration/jerk spikes).
        """
        from scipy.interpolate import PchipInterpolator

        t = np.clip(np.asarray(t, dtype=float), 0.0, self.duration)
        if len(self.t_knots) < 2:
            return np.full_like(t, self.s_knots[-1] if len(self.s_knots) else 0.0)
        interp = PchipInterpolator(self.t_knots, self.s_knots, extrapolate=True)
        return np.clip(interp(t), 0.0, self.length)


class _PathAdapter:
    """Duck-typed toppra geometric path wrapping a demoforge PathSpline (identical geometry)."""

    def __init__(self, path: PathSpline) -> None:
        self._path = path
        self.dof = path.n_joints

    @property
    def path_interval(self) -> np.ndarray:
        return np.array([0.0, self._path.length])

    def __call__(self, s: np.ndarray, order: int = 0) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if order == 0:
            out = self._path.q(s)
        elif order == 1:
            out = self._path.dqds(s)
        elif order == 2:
            out = self._path.d2qds2(s)
        else:  # pragma: no cover - toppra only requests 0/1/2
            raise ValueError(f"unsupported derivative order {order}")
        return out[0] if np.isscalar(s) or np.ndim(s) == 0 else out


def parameterize(
    path: PathSpline,
    limits: RobotLimits,
    *,
    backend: str = "auto",
    n_grid: int = 200,
) -> TimeLaw:
    """Compute a feasible time-optimal-ish ``s(t)`` for ``path`` under ``limits``.

    ``backend``:
      * ``"auto"`` (default) — TOPP-RA if ``toppra`` imports, else the pure-numpy backend.
        Degrades gracefully where toppra has no working wheel for the interpreter.
      * ``"topp"`` — require TOPP-RA (raises ImportError if unavailable).
      * ``"numpy"`` — pure-numpy, dependency-light (feasible, not time-optimal).
    """
    if limits.n != path.n_joints:
        raise ValueError(f"limits has {limits.n} joints but path has {path.n_joints}")
    if backend == "auto":
        try:
            return _parameterize_topp(path, limits, n_grid)
        except ImportError:
            return _parameterize_numpy(path, limits, n_grid)
    if backend == "topp":
        return _parameterize_topp(path, limits, n_grid)
    if backend == "numpy":
        return _parameterize_numpy(path, limits, n_grid)
    raise ValueError(f"unknown backend {backend!r}; use 'auto', 'topp' or 'numpy'")


def _parameterize_topp(path: PathSpline, limits: RobotLimits, n_grid: int) -> TimeLaw:
    try:
        import toppra.algorithm as algo
        import toppra.constraint as constraint
    except ImportError as exc:  # pragma: no cover - toppra is a core dep
        raise ImportError(
            "the 'topp' backend needs toppra (a core dependency); "
            "use backend='numpy' if it is unavailable"
        ) from exc

    adapter = _PathAdapter(path)
    s_grid = np.linspace(0.0, path.length, n_grid)
    vlim = np.column_stack([-limits.vel, limits.vel])
    alim = np.column_stack([-limits.acc, limits.acc])
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim)
    inst = algo.TOPPRA([pc_vel, pc_acc], adapter, gridpoints=s_grid, solver_wrapper="seidel")
    _, sd, _ = inst.compute_parameterization(0.0, 0.0)
    if sd is None or not np.all(np.isfinite(sd)):
        raise RuntimeError("TOPP-RA failed to find a feasible parameterization")
    sd = np.asarray(sd, dtype=float)
    sd = np.maximum(sd, 0.0)
    # Integrate ds/dt -> t with the trapezoidal rule. Guard the at-rest endpoints.
    ds = np.diff(s_grid)
    sd_avg = 0.5 * (sd[:-1] + sd[1:])
    floor = 1e-9 * (sd.max() if sd.max() > 0 else 1.0)
    sd_avg = np.maximum(sd_avg, floor)
    dt = ds / sd_avg
    t_knots = np.concatenate([[0.0], np.cumsum(dt)])
    return TimeLaw(t_knots, s_grid, backend="topp")


def _parameterize_numpy(path: PathSpline, limits: RobotLimits, n_grid: int) -> TimeLaw:
    """Constant path-speed bounded by vel + centripetal accel, with min-jerk end ramps."""
    s_grid = np.linspace(0.0, path.length, n_grid)
    dq = path.dqds(s_grid)  # (N, n) tangent
    d2q = path.d2qds2(s_grid)  # (N, n) curvature
    # velocity bound:   |dq_j| * sd <= vel_j           -> sd <= vel_j / |dq_j|
    # accel  bound:   |d2q_j| * sd^2 <= acc_j          -> sd <= sqrt(acc_j / |d2q_j|)
    eps = 1e-12
    sd_vel = np.min(limits.vel[None, :] / (np.abs(dq) + eps), axis=1)
    sd_acc = np.sqrt(np.min(limits.acc[None, :] / (np.abs(d2q) + eps), axis=1))
    sd_cap = np.minimum(sd_vel, sd_acc)
    sd_const = float(np.min(sd_cap))
    if not np.isfinite(sd_const) or sd_const <= 0:
        raise RuntimeError("numpy backend: degenerate path (zero feasible speed)")
    # cruise time at constant speed
    t_cruise = path.length / sd_const
    # add symmetric ramp time so we start/stop at rest within accel limits
    a_min = float(np.min(limits.acc))
    t_ramp = sd_const / a_min  # time to reach cruise speed
    # build a smooth (min-jerk) speed profile sd(s): ramp up, cruise, ramp down
    n = n_grid
    s_norm = s_grid / path.length
    ramp_frac = min(0.25, (t_ramp * sd_const) / (2.0 * path.length)) if path.length > 0 else 0.1
    ramp_frac = max(ramp_frac, 1e-3)
    prof = np.ones(n)
    up = s_norm < ramp_frac
    down = s_norm > (1.0 - ramp_frac)
    prof[up] = _minjerk_scalar(s_norm[up] / ramp_frac)
    prof[down] = _minjerk_scalar((1.0 - s_norm[down]) / ramp_frac)
    sd_prof = np.maximum(sd_const * prof, 1e-6 * sd_const)
    ds = np.diff(s_grid)
    sd_avg = 0.5 * (sd_prof[:-1] + sd_prof[1:])
    dt = ds / sd_avg
    t_knots = np.concatenate([[0.0], np.cumsum(dt)])
    _ = t_cruise  # informative only
    return TimeLaw(t_knots, s_grid, backend="numpy")


def _minjerk_scalar(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk position profile 10u^3 - 15u^4 + 6u^5 on u in [0, 1]."""
    u = np.clip(u, 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def preserve_contact(
    time_law: TimeLaw,
    segments: list[tuple[float, float, float]],
) -> TimeLaw:
    """Stretch time within contact arc-length ranges so they are not sped up below pace.

    ``segments`` is a list of ``(s_start, s_end, min_duration)``. For each segment whose
    re-timed duration fell below ``min_duration`` (e.g. the original recording pace over
    that range), the segment's time is dilated to exactly ``min_duration``. Dilation only
    slows down, so velocity/acceleration feasibility is preserved. Segments are assumed
    disjoint (the detector merges overlaps).
    """
    if not segments:
        return time_law
    s_knots = time_law.s_knots
    dt = np.diff(time_law.t_knots).astype(float)
    interval_factor = np.ones(len(dt))
    s_mid = 0.5 * (s_knots[:-1] + s_knots[1:])
    for s0, s1, min_dur in segments:
        if min_dur <= 0 or s1 <= s0:
            continue
        in_seg = (s_mid >= s0) & (s_mid <= s1)
        cur_dur = float(dt[in_seg].sum())
        if cur_dur <= 0:
            continue
        if cur_dur < min_dur:
            interval_factor[in_seg] = min_dur / cur_dur
    t_new = np.concatenate([[0.0], np.cumsum(dt * interval_factor)])
    return time_law.with_t_knots(t_new)


def sample_trajectory(
    path: PathSpline, time_law: TimeLaw, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``q(t)`` at a uniform ``fps`` over the whole time-law.

    Returns ``(t, q)`` with ``t`` uniform (last frame lands exactly on the duration).
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    dur = time_law.duration
    n = max(2, int(round(dur * fps)) + 1)
    t = np.linspace(0.0, dur, n)
    s = time_law.s_at(t)
    q = path.q(s)
    return t, q
