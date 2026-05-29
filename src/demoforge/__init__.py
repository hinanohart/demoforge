"""demoforge — deterministic, CPU-only re-timing of LeRobot teleop demonstrations.

Public surface:

* :mod:`demoforge.engine` — the LeRobot-free trajectory core (importable on its own).
* :class:`demoforge.RawEpisode`, :class:`demoforge.RetimeResult`, :class:`demoforge.DemoHealth`
  — the intermediate representation shared by the I/O and health layers.
* :func:`demoforge.retime_episode` — the high-level transform (path -> re-time -> emit arrays).

demoforge claims geometric/kinematic feasibility of the emitted trajectory only. It makes no
claim about downstream training outcomes; see the README.
"""

from __future__ import annotations

from .ir import DemoHealth, RawEpisode, RetimeConfig, RetimeResult
from .pipeline import retime_episode

__version__ = "0.1.0a1"

__all__ = [
    "__version__",
    "RawEpisode",
    "RetimeConfig",
    "RetimeResult",
    "DemoHealth",
    "retime_episode",
]
