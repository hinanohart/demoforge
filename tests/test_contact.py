"""Contact preservation: a delicate dwell/contact segment must keep its pacing while a
normal transport segment is re-timed (sped up). keep_count never speeds a segment below its
recorded pace, so contact micro-motion timing is preserved by construction."""

from __future__ import annotations

import numpy as np

from demoforge import RawEpisode, RetimeConfig, retime_episode
from demoforge.engine import RobotLimits

LIM = RobotLimits.from_preset("so101")


def _episode_with_dwell(fps=20.0):
    """Slow transport in, a contact phase (gripper closes, arm holds), then transport out.

    The gripper ramps closed during frames 30:60 and *stays* closed (no discontinuous snap),
    so the contact-joint signal is the smooth ramp itself.
    """
    n = 90
    t = np.arange(n) / fps
    q = np.zeros((n, LIM.n))
    center = 0.5 * (LIM.pos_bounds[:, 0] + LIM.pos_bounds[:, 1])
    for j in range(LIM.n):
        q[:, j] = center[j]
    c0 = center[0]
    q[:30, 0] = np.linspace(c0, c0 + 0.4, 30)
    q[30:, 0] = c0 + 0.4  # arm holds at the contact pose from frame 30 on
    q[60:, 0] = np.linspace(c0 + 0.4, c0 + 0.8, 30)
    # gripper closes smoothly during the contact phase and stays closed (no snap-back)
    gi = LIM.names.index("gripper")
    q[30:60, gi] = center[gi] + np.linspace(0.0, 0.3, 30)
    q[60:, gi] = center[gi] + 0.3
    return RawEpisode(actions=q, timestamps=t, fps=fps, joint_names=LIM.names)


def test_contact_segment_detected_and_locked():
    raw = _episode_with_dwell()
    _, health = retime_episode(
        raw, LIM, RetimeConfig(mode="keep_count", preserve_contact="gripper")
    )
    assert health.contact_segments_locked > 0
    assert "contact_locked" in health.triage["flags"]


def test_keep_count_does_not_speed_contact_below_recorded_pace():
    raw = _episode_with_dwell()
    results, health = retime_episode(
        raw, LIM, RetimeConfig(mode="keep_count", speeds=(1.0,), preserve_contact="gripper")
    )
    out = results[0]
    # locate the dwell frames (joint 0 roughly constant) and check their emitted duration
    q0 = raw.actions[:, 0]
    moving = np.abs(np.diff(q0)) > 1e-6
    dwell_idx = np.where(~moving)[0]
    # the dwell's emitted timespan should be at least its recorded timespan (never sped up)
    rec_span = raw.timestamps[dwell_idx[-1]] - raw.timestamps[dwell_idx[0]]
    emit_span = out.timestamps[dwell_idx[-1]] - out.timestamps[dwell_idx[0]]
    assert emit_span >= rec_span - 1e-6


def test_disabling_preserve_contact_finds_no_locked_segments_via_joint():
    raw = _episode_with_dwell()
    # with contact disabled, the gripper-actuation segments are not locked as contact_joint
    _, health = retime_episode(raw, LIM, RetimeConfig(mode="keep_count", preserve_contact=None))
    reasons = {s["reason"] for s in health.contact_segments}
    assert all("contact_joint" not in r for r in reasons)
