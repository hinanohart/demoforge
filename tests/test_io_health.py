"""I/O round-trip + health sidecar tests (torch-free, parquet level)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from demoforge.engine import RobotLimits, check_feasible
from demoforge.health import validate_record, write_sidecar
from demoforge.io import (
    list_episodes,
    make_mock_episode,
    read_episode,
    write_mock_dataset,
    write_retimed_dataset,
)
from demoforge.ir import RetimeConfig
from demoforge.pipeline import retime_episode

LIM = RobotLimits.from_preset("so101")


def test_mock_dataset_roundtrip(tmp_path):
    src = write_mock_dataset(
        tmp_path / "src",
        n_episodes=2,
        n_frames=60,
        n_joints=LIM.n,
        fps=20.0,
        joint_names=LIM.names,
        pos_bounds=LIM.pos_bounds,
    )
    assert list_episodes(src) == [0, 1]
    raw = read_episode(src, 0)
    assert raw.actions.shape == (60, LIM.n)
    assert raw.fps == 20.0
    assert raw.joint_names == LIM.names


def test_emit_roundtrip_preserves_count_and_is_feasible(tmp_path):
    src = write_mock_dataset(
        tmp_path / "src",
        n_episodes=1,
        n_frames=80,
        joint_names=LIM.names,
        pos_bounds=LIM.pos_bounds,
    )
    raw = read_episode(src, 0)
    results, _ = retime_episode(raw, LIM, RetimeConfig(mode="keep_count", speeds=(0.8, 1.0)))
    out = write_retimed_dataset(tmp_path / "out", results, fps=20.0, joint_names=LIM.names)
    assert list_episodes(out) == [0, 1]
    back = read_episode(out, 0)
    # keep_count: emitted frame count equals input frame count
    assert back.n_frames == raw.n_frames
    assert check_feasible(back.timestamps, back.actions, LIM, tol=0.05).dynamic_feasible


def test_reconciliation_in_equals_out_keep_count(tmp_path):
    raw = make_mock_episode(n_joints=LIM.n, joint_names=LIM.names, pos_bounds=LIM.pos_bounds)
    results, health = retime_episode(raw, LIM, RetimeConfig(mode="keep_count", speeds=(1.0,)))
    assert health.data_integrity["frames_in_unpadded"] == health.frames_out
    assert health.frames_out == results[0].n_frames


def test_integrity_flags_are_measured_not_hardcoded():
    raw = make_mock_episode(n_joints=LIM.n, joint_names=LIM.names, pos_bounds=LIM.pos_bounds)

    # keep_count keeps the input frame count exactly -> frame_count_preserved is True
    _, h_keep = retime_episode(raw, LIM, RetimeConfig(mode="keep_count", speeds=(1.0,)))
    assert h_keep.data_integrity["frame_count_preserved"] is True
    assert h_keep.data_integrity["no_nan_out"] is True
    assert "nan_in_input" not in h_keep.triage["flags"]

    # resample at a different fps changes the frame count by design -> frame_count_preserved
    # is honestly False, and the flag tracks the actual emitted vs input counts (not hardcoded)
    _, h_res = retime_episode(
        raw, LIM, RetimeConfig(mode="resample", speeds=(1.0,), target_fps=raw.fps * 2.0)
    )
    assert h_res.frames_out != h_res.data_integrity["frames_in_unpadded"]
    assert h_res.data_integrity["frame_count_preserved"] is False
    assert h_res.data_integrity["no_nan_out"] is True


def test_nan_input_is_flagged_and_not_falsely_clean():
    raw = make_mock_episode(n_joints=LIM.n, joint_names=LIM.names, pos_bounds=LIM.pos_bounds)
    raw.actions[raw.n_frames // 2, 0] = np.nan  # corrupt one recorded sample
    _, health = retime_episode(raw, LIM, RetimeConfig(mode="keep_count", speeds=(1.0,)))
    # the corrupt input is surfaced, and a NaN that survives to the output is NOT reported
    # clean (the previous hardcoded no_nan_out=True hid exactly this case)
    assert "nan_in_input" in health.triage["flags"]
    assert health.triage["suggestion"] == "review:nan_in_input"
    assert health.data_integrity["no_nan_out"] is False


def test_pad_mask_removed():
    raw = make_mock_episode(n_joints=LIM.n, joint_names=LIM.names, pos_bounds=LIM.pos_bounds)
    raw.action_is_pad = np.zeros(raw.n_frames, dtype=bool)
    raw.action_is_pad[-5:] = True  # last 5 frames are padding
    unp, _ = raw.unpadded()
    assert len(unp) == raw.n_frames - 5


def test_sidecar_validates_and_has_no_quality_score(tmp_path):
    raw = make_mock_episode(n_joints=LIM.n, joint_names=LIM.names, pos_bounds=LIM.pos_bounds)
    _, health = retime_episode(raw, LIM, RetimeConfig())
    path = write_sidecar(tmp_path / "h.jsonl", [health])
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(recs) == 1
    validate_record(recs[0])
    assert "quality_score" not in recs[0]


def test_sidecar_schema_rejects_quality_score():
    # negative fixture: an aggregate score must fail validation (eval-tool firewall)
    bad = {
        "episode_index": 0,
        "source": "x",
        "backend": "topp",
        "frames_in": 1,
        "frames_out": 1,
        "limit_violations_before": 0,
        "limit_violations_after": 0,
        "max_jerk_before": 0.0,
        "max_jerk_after": 0.0,
        "path_deviation_max": 0.0,
        "retime_outcome": "retimed",
        "triage": {"flags": [], "suggestion": "keep"},
        "quality_score": 0.9,
    }
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        validate_record(bad)
