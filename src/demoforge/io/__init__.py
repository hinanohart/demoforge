"""I/O adapters between LeRobot datasets and demoforge's :class:`~demoforge.ir.RawEpisode`.

The default read/write path is **torch-free** (pyarrow over the v3 frame parquet), so the whole
re-timing pipeline runs on a laptop without installing LeRobot/torch. v0.1.0a1 emits the
re-timed action/timestamp frame layer; a canonical, hub-ready emit via the real
``LeRobotDataset`` (media copy + ``finalize()``) is planned for a later release.
"""

from __future__ import annotations

from .lerobot_v3 import (
    list_episodes,
    read_episode,
    read_info,
    write_retimed_dataset,
)
from .mock import make_mock_episode, write_mock_dataset

__all__ = [
    "read_episode",
    "list_episodes",
    "read_info",
    "write_retimed_dataset",
    "make_mock_episode",
    "write_mock_dataset",
]
