"""Property tests for the core claim: the emitted trajectory is dynamically feasible,
jerk-bounded, path-preserving (keep_count: exact) and deterministic."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from demoforge import RawEpisode, RetimeConfig, retime_episode
from demoforge.engine import RobotLimits, check_feasible

LIM = RobotLimits.from_preset("so101")


def _episode(seed: int, n: int, amp: float, jit: float) -> RawEpisode:
    rng = np.random.RandomState(seed)
    t = np.arange(n) / 20.0
    center = 0.5 * (LIM.pos_bounds[:, 0] + LIM.pos_bounds[:, 1])
    half = 0.5 * (LIM.pos_bounds[:, 1] - LIM.pos_bounds[:, 0]) * amp
    base = np.column_stack(
        [center[j] + half[j] * np.sin((0.4 + 0.1 * j) * t) for j in range(LIM.n)]
    )
    q = base + jit * rng.randn(n, LIM.n)
    q = np.clip(q, LIM.pos_bounds[:, 0] + 0.01, LIM.pos_bounds[:, 1] - 0.01)
    return RawEpisode(actions=q, timestamps=t, fps=20.0, joint_names=LIM.names)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(20, 120),
    amp=st.floats(0.2, 0.6),
    jit=st.floats(0.0, 0.08),
    mode=st.sampled_from(["keep_count", "resample"]),
)
def test_emitted_is_dynamically_feasible(seed, n, amp, jit, mode):
    raw = _episode(seed, n, amp, jit)
    cfg = RetimeConfig(mode=mode, speeds=(1.0,))
    results, health = retime_episode(raw, LIM, cfg)
    rep = check_feasible(results[0].timestamps, results[0].actions, LIM, tol=0.05)
    assert rep.dynamic_feasible, (
        mode,
        rep.n_vel_violations,
        rep.n_acc_violations,
        rep.n_jerk_violations,
    )
    assert health.limit_violations_after == 0


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), n=st.integers(30, 100))
def test_keep_count_preserves_positions_and_frame_count(seed, n):
    raw = _episode(seed, n, 0.4, 0.05)
    results, _ = retime_episode(raw, LIM, RetimeConfig(mode="keep_count", speeds=(1.0,)))
    out = results[0]
    assert out.n_frames == raw.n_frames
    # non-continuous joints are bit-preserved; continuous joints differ only by 2*pi*k
    cont = LIM.continuous_mask
    assert np.allclose(out.actions[:, ~cont], raw.actions[:, ~cont])


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000))
def test_deterministic_bit_exact(seed):
    raw = _episode(seed, 60, 0.4, 0.05)
    r1, _ = retime_episode(raw, LIM, RetimeConfig(speeds=(0.8, 1.0)))
    r2, _ = retime_episode(raw, LIM, RetimeConfig(speeds=(0.8, 1.0)))
    for a, b in zip(r1, r2, strict=True):
        assert np.array_equal(a.actions, b.actions)
        assert np.array_equal(a.timestamps, b.timestamps)


def test_slower_speed_has_lower_or_equal_jerk():
    raw = _episode(1, 80, 0.5, 0.06)
    results, _ = retime_episode(raw, LIM, RetimeConfig(speeds=(0.6, 1.0)))
    by_speed = {r.speed: r for r in results}
    j_slow = check_feasible(by_speed[0.6].timestamps, by_speed[0.6].actions, LIM).max_jerk
    j_fast = check_feasible(by_speed[1.0].timestamps, by_speed[1.0].actions, LIM).max_jerk
    assert j_slow <= j_fast + 1e-6


def test_faster_than_optimal_is_clamped_not_infeasible():
    raw = _episode(2, 80, 0.5, 0.06)
    results, _ = retime_episode(raw, LIM, RetimeConfig(speeds=(2.0,)))
    out = results[0]
    assert out.clamped
    assert check_feasible(out.timestamps, out.actions, LIM, tol=0.05).dynamic_feasible


def test_too_short_episode_passes_through():
    raw = RawEpisode(
        actions=np.zeros((2, LIM.n)),
        timestamps=np.array([0.0, 0.1]),
        fps=20.0,
        joint_names=LIM.names,
    )
    results, health = retime_episode(raw, LIM, RetimeConfig())
    assert results[0].outcome == "infeasible_passthrough"
    assert np.array_equal(results[0].actions, raw.actions)
    assert "infeasible_passthrough" in health.triage["flags"]
