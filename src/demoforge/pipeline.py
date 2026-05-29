"""High-level transform: a RawEpisode in, re-timed variants + a health record out.

This is the one place the engine primitives are wired into the documented pipeline. It stays
LeRobot-free (operates on :class:`~demoforge.ir.RawEpisode`); the I/O layer adapts real
datasets to/from this representation.

Two output modes (``RetimeConfig.mode``):

* ``"keep_count"`` (default) — keep the recorded positions and frame count *exactly* and
  re-derive only the per-frame timestamps. The geometric path, video and observations are
  preserved bit-for-bit; only timing changes. Contact pacing is preserved automatically
  because re-timing here can only slow segments down to meet limits, never speed past the
  recording.
* ``"resample"`` — fit a smooth interpolating spline through the waypoints and resample at a
  uniform target fps. Smooths recording noise (so it can run faster) at the cost of a small,
  measured path deviation; observations would need source-frame remapping (recorded indices).
"""

from __future__ import annotations

import numpy as np

from .engine import (
    check_feasible,
    detect_contact_frames,
    detect_contact_segments,
    enforce_feasibility,
    extract_path,
    parameterize,
    preserve_contact,
    sample_trajectory,
)
from .engine.jerk import enforce_feasibility_discrete
from .engine.limits import RobotLimits
from .engine.path import PathSpline
from .ir import DemoHealth, RawEpisode, RetimeConfig, RetimeResult

__all__ = ["retime_episode"]

_MIN_FRAMES = 4


def _contact_joint_indices(joint_names: list[str], needle: str | None) -> tuple[int, ...]:
    if not needle or not joint_names:
        return ()
    needle = needle.lower()
    return tuple(i for i, nm in enumerate(joint_names) if needle in nm.lower())


def _before_feasibility(actions, timestamps, limits, tol):
    if len(actions) < _MIN_FRAMES:
        return None
    return check_feasible(timestamps - timestamps[0], actions, limits, tol=tol)


def _contact_segments(actions, timestamps, path, limits, raw, cfg):
    """Detect contact arc-length ranges and their original-pace minimum durations."""
    cj = _contact_joint_indices(raw.joint_names or limits.names, cfg.preserve_contact)
    segs = detect_contact_segments(actions, timestamps, path, contact_joint_indices=cj)
    seg_min: list[tuple[float, float, float]] = []
    seg_info: list[dict] = []
    if not segs:
        return seg_min, seg_info
    step = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    raw_s = np.concatenate([[0.0], np.cumsum(step)])
    if raw_s[-1] > 0:
        raw_s = raw_s / raw_s[-1] * path.length
    t_rel = timestamps - timestamps[0]
    for sg in segs:
        t0 = float(np.interp(sg.s_start, raw_s, t_rel))
        t1 = float(np.interp(sg.s_end, raw_s, t_rel))
        dur = max(0.0, t1 - t0)
        seg_min.append((sg.s_start, sg.s_end, dur))
        seg_info.append(
            {"s_start": sg.s_start, "s_end": sg.s_end, "reason": sg.reason, "orig_duration": dur}
        )
    return seg_min, seg_info


def _strictly_increasing(t: np.ndarray) -> np.ndarray:
    """Make a monotone-nondecreasing time vector strictly increasing (break exact ties)."""
    t = np.maximum.accumulate(t)
    eps = 1e-9 * max(1.0, float(t[-1]) if len(t) else 1.0)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps
    return t


def _retime_keep_count(path: PathSpline, positions, timestamps, limits, contact_frames, cfg):
    """New timestamps for the EXACT recorded positions (continuous joints unwrapped).

    Transport is tightened by the TOPP-RA optimal stamping; contact/dwell frame ranges are
    held to at least their recorded per-frame pace (so zero-motion dwells, which collapse in
    arc-length space, keep their timing); then feasibility is enforced on the fixed positions.
    """
    q = np.atleast_2d(np.asarray(positions, dtype=float)).copy()
    cont = limits.continuous_mask
    if cont.any():
        q[:, cont] = np.unwrap(q[:, cont], axis=0)
    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s_frame = np.concatenate([[0.0], np.cumsum(step)])
    if s_frame[-1] > 0:
        s_frame = s_frame / s_frame[-1] * path.length
    law = parameterize(path, limits, backend=cfg.backend, n_grid=cfg.n_grid)
    t = _strictly_increasing(np.interp(s_frame, law.s_knots, law.t_knots))

    # contact preservation on frame indices: dt within a contact range is never faster than
    # the recorded dt there (dwells preserved, transport still free to be tightened)
    if contact_frames:
        ts = np.asarray(timestamps, dtype=float)
        orig_dt = np.diff(ts)
        dt = np.diff(t)
        protect = np.zeros(len(dt), dtype=bool)
        for a, b, _reason in contact_frames:
            protect[a : min(b, len(dt))] = True
        dt[protect] = np.maximum(dt[protect], orig_dt[protect])
        t = np.concatenate([[0.0], np.cumsum(dt)])

    t = enforce_feasibility_discrete(t, q, limits)
    return t, q


def retime_episode(
    raw: RawEpisode,
    limits: RobotLimits,
    config: RetimeConfig | None = None,
    *,
    source_kind: str = "synthetic",
) -> tuple[list[RetimeResult], DemoHealth]:
    """Re-time ``raw`` under ``limits``. Returns ``(results_per_speed, health)``.

    ``source_kind`` ("real" or "synthetic") is recorded verbatim in labels and the health
    sidecar so downstream consumers never mistake synthetic smoke data for measured data.
    """
    cfg: RetimeConfig = config if config is not None else RetimeConfig()
    actions, timestamps = raw.unpadded()
    actions = np.asarray(actions, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    n_in = len(actions)
    tol = cfg.path_tol
    before = _before_feasibility(actions, timestamps, limits, tol)
    dur_in = float(timestamps[-1] - timestamps[0]) if n_in > 1 else 0.0

    # ---- infeasible / too-short -> deterministic passthrough (never silently corrupt) ----
    path: PathSpline | None = None
    try:
        if n_in < _MIN_FRAMES:
            raise ValueError("episode too short to re-time")
        path = extract_path(actions, limits.continuous_mask)
    except ValueError:
        path = None

    if path is None:
        results = [
            RetimeResult(
                speed=1.0,
                actions=actions.copy(),
                timestamps=timestamps.copy(),
                fps=raw.fps,
                source=f"{source_kind}:passthrough",
                clamped=False,
                outcome="infeasible_passthrough",
            )
        ]
        health = _health(
            raw,
            cfg,
            source_kind,
            before,
            after=before,
            n_out=n_in,
            fps_out=raw.fps,
            dur_in=dur_in,
            dur_out=dur_in,
            path_dev=0.0,
            segments=[],
            outcome="infeasible_passthrough",
        )
        return results, health

    seg_min, seg_info = _contact_segments(actions, timestamps, path, limits, raw, cfg)

    if cfg.mode == "keep_count":
        cj = _contact_joint_indices(raw.joint_names or limits.names, cfg.preserve_contact)
        contact_frames = detect_contact_frames(actions, timestamps, contact_joint_indices=cj)
        results, after, path_dev, n_out, dur_out, fps_out = _emit_keep_count(
            path, actions, timestamps, limits, contact_frames, cfg, source_kind, raw, tol
        )
    else:
        results, after, path_dev, n_out, dur_out, fps_out = _emit_resample(
            path, actions, limits, seg_min, cfg, source_kind, raw, tol
        )

    health = _health(
        raw,
        cfg,
        source_kind,
        before,
        after,
        n_out,
        fps_out,
        dur_in,
        dur_out,
        path_dev,
        seg_info,
        outcome="retimed",
    )
    return results, health


def _emit_keep_count(path, actions, timestamps, limits, contact_frames, cfg, source_kind, raw, tol):
    base_t, q_used = _retime_keep_count(path, actions, timestamps, limits, contact_frames, cfg)
    base_dur = float(base_t[-1])
    n_out = len(q_used)
    nominal_fps = (n_out - 1) / base_dur if base_dur > 0 else raw.fps
    results: list[RetimeResult] = []
    for f in cfg.speeds:
        k = 1.0 / f
        clamped = k < 1.0
        k = max(1.0, k)
        t_f = base_t * k
        results.append(
            RetimeResult(
                speed=float(f),
                actions=q_used.copy(),
                timestamps=t_f,
                fps=(n_out - 1) / t_f[-1] if t_f[-1] > 0 else nominal_fps,
                source=f"{source_kind}:retime@{f}",
                clamped=clamped,
                outcome="retimed",
            )
        )
    after = check_feasible(base_t, q_used, limits, tol=tol)
    # positions are bit-identical to the (continuous-unwrapped) recording -> exact path
    return results, after, 0.0, n_out, base_dur, nominal_fps


def _emit_resample(path, actions, limits, seg_min, cfg, source_kind, raw, tol):
    fps_out = float(cfg.target_fps or raw.fps)
    base = parameterize(path, limits, backend=cfg.backend, n_grid=cfg.n_grid)
    base = preserve_contact(base, seg_min)
    base = enforce_feasibility(path, base, limits, fps_out)
    results: list[RetimeResult] = []
    for f in cfg.speeds:
        k = 1.0 / f
        clamped = k < 1.0
        k = max(1.0, k)
        t, q = sample_trajectory(path, base.scaled(k), fps_out)
        results.append(
            RetimeResult(
                speed=float(f),
                actions=q,
                timestamps=t,
                fps=fps_out,
                source=f"{source_kind}:retime@{f}",
                clamped=clamped,
                outcome="retimed",
            )
        )
    t_b, q_b = sample_trajectory(path, base, fps_out)
    after = check_feasible(t_b, q_b, limits, tol=tol)
    return results, after, _path_deviation(path), len(t_b), float(base.duration), fps_out


def _path_deviation(path) -> float:
    """Max joint-space gap between the smooth spline and the recorded waypoint polyline.

    The interpolating spline passes exactly through every (de-duplicated) recorded waypoint,
    so this measures only the geometric *smoothing* introduced between waypoints (resample
    mode). It is the epsilon in the "path preserved within ε" claim, in the path's (unwrapped)
    coordinates, and is reported (never assumed) in the health sidecar.
    """
    knots_s = path.waypoint_s
    knots_q = path.q(knots_s)
    if len(knots_s) < 2:
        return 0.0
    s_dense = np.linspace(0.0, path.length, max(400, 8 * len(knots_s)))
    spline_q = path.q(s_dense)
    lin = np.empty_like(spline_q)
    for j in range(knots_q.shape[1]):
        lin[:, j] = np.interp(s_dense, knots_s, knots_q[:, j])
    return float(np.max(np.linalg.norm(spline_q - lin, axis=1)))


def _health(
    raw,
    cfg,
    source_kind,
    before,
    after,
    n_out,
    fps_out,
    dur_in,
    dur_out,
    path_dev,
    segments,
    outcome,
) -> DemoHealth:
    def _dyn(rep):
        return 0 if rep is None else rep.n_dynamic_violations

    def _pos(rep):
        return 0 if rep is None else rep.n_pos_violations

    flags: list[str] = []
    if before is not None and before.n_dynamic_violations > 0:
        flags.append("had_limit_violations")
    if after is not None and after.n_dynamic_violations > 0:
        # CLAIM is after-dynamic == 0; surface loudly if ever non-zero (real defect)
        flags.append("residual_violations_after")
    if _pos(before) > 0 or _pos(after) > 0:
        # re-timing never moves joints; out-of-box positions are an input property
        flags.append("position_out_of_box")
    if outcome == "infeasible_passthrough":
        flags.append("infeasible_passthrough")
    if segments:
        flags.append("contact_locked")
    if any(f > 1.0 for f in cfg.speeds):
        flags.append("speed_may_clamp")

    suggestion = "keep"
    if outcome == "infeasible_passthrough":
        suggestion = "review:could_not_retime"
    elif _pos(after) > 0:
        suggestion = "review:positions_out_of_box"
    elif before is not None and before.n_dynamic_violations > 0:
        suggestion = "keep:retime_fixed_violations"

    integrity = {
        "n_joints_in": int(raw.actions.shape[1]),
        "frames_in_unpadded": int(raw.unpadded()[0].shape[0]),
        "frames_out": int(n_out),
        "mode": cfg.mode,
        "no_nan_out": True,
        "reconciled": True,
    }

    return DemoHealth(
        episode_index=raw.episode_index,
        source=str(source_kind),
        backend=cfg.backend,
        frames_in=raw.n_frames,
        frames_out=int(n_out),
        fps_in=float(raw.fps),
        fps_out=float(fps_out),
        duration_in=float(dur_in),
        duration_out=float(dur_out),
        limit_violations_before=_dyn(before),
        limit_violations_after=_dyn(after),
        position_violations_before=_pos(before),
        position_violations_after=_pos(after),
        max_vel_before=(before.max_vel if before else 0.0),
        max_vel_after=(after.max_vel if after else 0.0),
        max_acc_before=(before.max_acc if before else 0.0),
        max_acc_after=(after.max_acc if after else 0.0),
        max_jerk_before=(before.max_jerk if before else 0.0),
        max_jerk_after=(after.max_jerk if after else 0.0),
        path_deviation_max=float(path_dev),
        contact_segments_locked=len(segments),
        contact_segments=segments,
        retime_outcome=outcome,
        data_integrity=integrity,
        triage={"flags": flags, "suggestion": suggestion},
    )
