"""Torch-free read/write of LeRobotDataset v3 frame data via pyarrow.

A v3 dataset stores frames as parquet (columns ``action``, ``observation.*``, ``timestamp``,
``frame_index``, ``episode_index``, ``index``, ``task_index`` ...) plus ``meta/info.json``.
demoforge reads the action stream and timestamps from the parquet (no torch needed). The writer
emits the re-timed action/timestamp frame layer (``timestamp``, ``frame_index``,
``episode_index``, ``index``, ``task_index``, ``action``) plus ``meta/info.json``; it does not
copy source videos/observations or call LeRobot's ``finalize()`` — a canonical, hub-ready emit
is planned for a later release.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..ir import RawEpisode, RetimeResult

__all__ = ["read_info", "list_episodes", "read_episode", "write_retimed_dataset"]

_INFO_REL = "meta/info.json"


def _data_files(root: Path) -> list[Path]:
    """All frame parquet files under a dataset root (v3 ``data/`` tree or flat)."""
    data_dir = root / "data"
    search = data_dir if data_dir.is_dir() else root
    return sorted(p for p in search.rglob("*.parquet"))


def read_info(root: str | Path) -> dict[str, Any]:
    """Load ``meta/info.json`` if present, else ``{}``."""
    info_path = Path(root) / _INFO_REL
    if info_path.is_file():
        return json.loads(info_path.read_text(encoding="utf-8"))
    return {}


def _read_table(root: Path) -> pa.Table:
    files = _data_files(root)
    if not files:
        raise FileNotFoundError(f"no parquet frame files found under {root}")
    return pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")


def list_episodes(root: str | Path) -> list[int]:
    """Sorted unique episode indices present in the dataset."""
    table = _read_table(Path(root))
    if "episode_index" not in table.column_names:
        return [0]
    return sorted({int(x) for x in table.column("episode_index").to_pylist()})


def _list_col_to_2d(col: pa.ChunkedArray) -> np.ndarray:
    rows = col.to_pylist()
    return np.array(rows, dtype=float)


def read_episode(
    root: str | Path,
    episode_index: int | None = None,
    *,
    action_key: str = "action",
    fps: float | None = None,
) -> RawEpisode:
    """Read one episode's action stream + timestamps into a :class:`RawEpisode` (torch-free)."""
    root = Path(root)
    table = _read_table(root)
    cols = table.column_names
    if action_key not in cols:
        raise KeyError(f"action column {action_key!r} not found; columns: {cols}")

    if episode_index is not None and "episode_index" in cols:
        # pyarrow.compute is populated dynamically: older builds ship no stub for `equal`
        # (needs attr-defined), newer builds do (then the ignore is unused). Tolerate both.
        mask = pc.equal(table.column("episode_index"), episode_index)  # type: ignore[attr-defined, unused-ignore]
        table = table.filter(mask)
        if table.num_rows == 0:
            raise ValueError(f"episode {episode_index} not present")

    # sort by frame_index for a well-ordered trajectory
    if "frame_index" in cols:
        order = np.argsort(np.array(table.column("frame_index").to_pylist(), dtype=int))
    else:
        order = np.arange(table.num_rows)

    actions = _list_col_to_2d(table.column(action_key))[order]

    info = read_info(root)
    fps_val = float(fps or info.get("fps") or 0.0)
    if "timestamp" in cols:
        ts = np.array(table.column("timestamp").to_pylist(), dtype=float)[order]
    elif fps_val > 0:
        ts = np.arange(len(actions), dtype=float) / fps_val
    else:
        raise ValueError("dataset has no 'timestamp' column and no fps to synthesize one")
    if fps_val <= 0:
        diffs = np.diff(ts)
        fps_val = float(1.0 / np.median(diffs)) if len(diffs) and np.median(diffs) > 0 else 30.0

    pad = None
    if "action_is_pad" in cols:
        pad = np.array(table.column("action_is_pad").to_pylist(), dtype=bool)[order]

    names = []
    feat = info.get("features", {}).get(action_key, {})
    if isinstance(feat, dict):
        names = list(feat.get("names") or [])

    return RawEpisode(
        actions=actions,
        timestamps=ts,
        fps=fps_val,
        episode_index=int(episode_index or 0),
        joint_names=names,
        action_is_pad=pad,
    )


def write_retimed_dataset(
    out_root: str | Path,
    results: list[RetimeResult],
    *,
    fps: float,
    action_key: str = "action",
    joint_names: list[str] | None = None,
    source_info: dict[str, Any] | None = None,
) -> Path:
    """Write re-timed variants as v3-compatible frame parquet + ``meta/info.json`` (torch-free).

    Each :class:`RetimeResult` becomes one episode; its ``speed``/source is recorded in a
    ``demoforge_source`` column. This emits the action/timestamp frame layer only — source
    videos/observations are not copied and LeRobot's ``finalize()`` is not called (a canonical,
    hub-ready emit is planned for a later release).
    """
    out_root = Path(out_root)
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)

    n_joints = int(results[0].actions.shape[1]) if results else 0
    global_index = 0
    for ep_i, res in enumerate(results):
        n = res.n_frames
        action_list = [row.astype(np.float32).tolist() for row in res.actions]
        table = pa.table(
            {
                "timestamp": pa.array(res.timestamps.astype(np.float64)),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, ep_i, dtype=np.int64)),
                "index": pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
                "task_index": pa.array(np.zeros(n, dtype=np.int64)),
                action_key: pa.array(action_list, type=pa.list_(pa.float32())),
                "demoforge_source": pa.array([res.source] * n),
            }
        )
        pq.write_table(table, out_root / "data" / f"episode_{ep_i:06d}.parquet")
        global_index += n

    info = dict(source_info or {})
    info.update(
        {
            "codebase_version": info.get("codebase_version", "v3.0"),
            "fps": float(fps),
            "total_episodes": len(results),
            "demoforge": {"retimed": True, "tool": "demoforge"},
        }
    )
    feats = info.get("features", {})
    feats[action_key] = {
        "dtype": "float32",
        "shape": [n_joints],
        "names": list(joint_names or []),
    }
    info["features"] = feats
    (out_root / _INFO_REL).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out_root
