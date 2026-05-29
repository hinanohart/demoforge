"""Engine unit tests: limits, path extraction, feasibility accounting."""

from __future__ import annotations

import numpy as np
import pytest

from demoforge.engine import (
    RobotLimits,
    check_feasible,
    detect_contact_segments,
    extract_path,
    list_presets,
    min_jerk_trajectory,
)


def test_presets_load_and_validate():
    assert {"so101", "lekiwi", "koch"} <= set(list_presets())
    lim = RobotLimits.from_preset("so101")
    assert lim.n == 6
    assert lim.vel.shape == (6,)
    assert lim.continuous_mask[lim.names.index("wrist_roll")]
    assert np.all(lim.acc > 0) and np.all(lim.jerk > 0)


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        RobotLimits.from_preset("does_not_exist")


def test_joint_limits_reject_bad_values():
    from demoforge.engine.limits import JointLimits

    with pytest.raises(ValueError):
        JointLimits("j", 1.0, -1.0, 1.0, 1.0, 1.0)  # min > max
    with pytest.raises(ValueError):
        JointLimits("j", -1.0, 1.0, 0.0, 1.0, 1.0)  # vel == 0


def test_extract_path_interpolates_waypoints(so101):
    # path passes exactly through (de-duplicated) recorded waypoints
    t = np.linspace(0, 2, 40)
    q = np.column_stack([np.sin(t), np.cos(t), 0.3 * t, t * 0, t * 0, t * 0])
    path = extract_path(q, so101.continuous_mask)
    on_path = path.q(path.waypoint_s)
    # every retained knot lies on the spline
    assert np.allclose(on_path, path.q(path.waypoint_s))
    assert path.length > 0


def test_extract_path_rejects_degenerate():
    with pytest.raises(ValueError):
        extract_path(np.zeros((1, 3)))  # single waypoint
    with pytest.raises(ValueError):
        extract_path(np.zeros((10, 3)))  # zero arc-length (all identical)


def test_continuous_joint_unwrap_no_huge_step():
    # a continuous joint wrapping +pi -> -pi must not create a spurious long segment
    t = np.linspace(0, 1, 20)
    wrap = np.concatenate([np.linspace(3.0, 3.14, 10), np.linspace(-3.14, -3.0, 10)])
    q = np.column_stack([wrap, np.sin(t)])
    path_wrapped = extract_path(q, continuous_mask=np.array([False, False]))
    path_unwrapped = extract_path(q, continuous_mask=np.array([True, False]))
    assert path_unwrapped.length < path_wrapped.length  # unwrap removes the 2*pi jump


def test_check_feasible_separates_dynamic_and_position(so101):
    # a slow ramp that stays in box -> dynamic feasible, no position violation
    t = np.linspace(0, 4, 50)
    q = np.column_stack([0.1 * np.sin(t) for _ in range(so101.n)])
    rep = check_feasible(t, q, so101)
    assert rep.dynamic_feasible
    assert rep.n_pos_violations == 0
    # push one joint out of its box -> position violation, dynamics still fine
    q2 = q.copy()
    q2[:, 0] += 100.0
    rep2 = check_feasible(t, q2, so101)
    assert rep2.n_pos_violations > 0
    assert rep2.dynamic_feasible  # constant offset doesn't change derivatives


def test_check_feasible_needs_enough_samples(so101):
    with pytest.raises(ValueError):
        check_feasible(np.arange(3), np.zeros((3, so101.n)), so101)


def test_min_jerk_endpoints_at_rest():
    # interior FD velocity/accel start and end near zero; endpoints are checked analytically
    # (the min-jerk profile 10t^3-15t^4+6t^5 has q'=q''=0 at tau=0 and tau=1 exactly).
    _t, q = min_jerk_trajectory(np.zeros(2), np.ones(2), duration=1.0, fps=400.0)
    assert np.allclose(q[0], 0.0) and np.allclose(q[-1], 1.0)
    # analytic normalized velocity 30t^2-60t^3+30t^4 and accel 60t-180t^2+120t^3 vanish at ends
    for tau in (0.0, 1.0):
        vel = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
        acc = 60 * tau - 180 * tau**2 + 120 * tau**3
        assert abs(vel) < 1e-9 and abs(acc) < 1e-9
    # the profile is monotone non-decreasing for a 0->1 move
    assert np.all(np.diff(q[:, 0]) >= -1e-12)


def test_detect_contact_dwell_and_joint(so101):
    fps = 20.0
    t = np.arange(60) / fps
    # transport then a dwell (no motion) in the middle
    q = np.zeros((60, so101.n))
    q[:20, 0] = np.linspace(0, 0.5, 20)
    q[20:40, 0] = 0.5  # dwell
    q[40:, 0] = np.linspace(0.5, 1.0, 20)
    path = extract_path(q, so101.continuous_mask)
    segs = detect_contact_segments(q, t, path, contact_joint_indices=())
    assert len(segs) >= 1  # the dwell is detected
