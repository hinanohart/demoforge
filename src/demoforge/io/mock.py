"""Deterministic synthetic LeRobotDataset-v3-shaped data for tests and offline smoke runs.

No torch, no network, no real robot. Episodes are reproducible from a seed so they can be used
in CI without downloading anything. Data produced here is always labelled ``synthetic`` by the
pipeline, never ``real``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..ir import RawEpisode

__all__ = ["make_mock_episode", "write_mock_dataset"]


def make_mock_episode(
    *,
    n_frames: int = 80,
    n_joints: int = 6,
    fps: float = 20.0,
    seed: int = 0,
    jerk_amp: float = 0.05,
    joint_names: list[str] | None = None,
    pos_bounds: np.ndarray | None = None,
) -> RawEpisode:
    """A smooth multi-sine path plus reproducible high-frequency jitter (sloppy teleop).

    If ``pos_bounds`` (shape ``(n_joints, 2)``) is given, each joint's motion is centred in
    its box and kept inside it, so the synthetic demo has no position-box violations (only the
    dynamic vel/acc/jerk overshoot the re-timer is meant to fix). Otherwise a conservative
    ``+/-0.15`` rad swing around 0 is used.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_frames, dtype=float) / fps
    freqs = 0.4 + 0.15 * np.arange(n_joints)
    phases = 0.3 * np.arange(n_joints)
    if pos_bounds is not None:
        pos_bounds = np.asarray(pos_bounds, dtype=float)
        center = 0.5 * (pos_bounds[:, 0] + pos_bounds[:, 1])
        half = 0.6 * 0.5 * (pos_bounds[:, 1] - pos_bounds[:, 0])
    else:
        center = np.zeros(n_joints)
        half = np.full(n_joints, 0.15)
    base = np.column_stack(
        [center[j] + half[j] * np.sin(freqs[j] * t + phases[j]) for j in range(n_joints)]
    )
    jitter = jerk_amp * rng.randn(n_frames, n_joints)
    actions = base + jitter
    if pos_bounds is not None:
        margin = 0.02 * (pos_bounds[:, 1] - pos_bounds[:, 0])
        actions = np.clip(actions, pos_bounds[:, 0] + margin, pos_bounds[:, 1] - margin)
    return RawEpisode(
        actions=actions,
        timestamps=t,
        fps=fps,
        episode_index=0,
        joint_names=joint_names or [f"joint_{j}" for j in range(n_joints)],
    )


def write_mock_dataset(
    root: str | Path,
    *,
    n_episodes: int = 2,
    n_frames: int = 80,
    n_joints: int = 6,
    fps: float = 20.0,
    joint_names: list[str] | None = None,
    base_seed: int = 0,
    pos_bounds: np.ndarray | None = None,
) -> Path:
    """Write a minimal v3-compatible parquet dataset (+ ``meta/info.json``) readable by
    :func:`demoforge.io.read_episode`. Includes a stand-in ``observation.image_index`` column
    so frame/observation reconciliation can be exercised without a real video file.
    """
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    names = joint_names or [f"joint_{j}" for j in range(n_joints)]
    global_index = 0
    for ep in range(n_episodes):
        epi = make_mock_episode(
            n_frames=n_frames,
            n_joints=n_joints,
            fps=fps,
            seed=base_seed + ep,
            joint_names=names,
            pos_bounds=pos_bounds,
        )
        n = epi.n_frames
        action_list = [row.astype(np.float32).tolist() for row in epi.actions]
        table = pa.table(
            {
                "timestamp": pa.array(epi.timestamps.astype(np.float64)),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, ep, dtype=np.int64)),
                "index": pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
                "task_index": pa.array(np.zeros(n, dtype=np.int64)),
                "action": pa.array(action_list, type=pa.list_(pa.float32())),
                "observation.image_index": pa.array(np.arange(n, dtype=np.int64)),
            }
        )
        pq.write_table(table, root / "data" / f"episode_{ep:06d}.parquet")
        global_index += n

    info = {
        "codebase_version": "v3.0",
        "fps": float(fps),
        "total_episodes": n_episodes,
        "features": {
            "action": {"dtype": "float32", "shape": [n_joints], "names": names},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return root
