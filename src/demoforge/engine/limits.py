"""Joint kinematic limits — the contract demoforge re-times against.

This module is part of ``demoforge.engine`` and has **no LeRobot / torch import**.
A :class:`RobotLimits` is the machine-readable target every emitted trajectory is
checked against: position, velocity, acceleration and jerk bounds, per joint.
"""

from __future__ import annotations

# demoforge requires Python >=3.10; importlib.resources.files() is available since 3.9.
import importlib.resources as resources  # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

__all__ = ["JointLimits", "RobotLimits", "list_presets"]


@dataclass(frozen=True)
class JointLimits:
    """Limits for a single joint. ``pos`` is (min, max); the rest are symmetric maxima."""

    name: str
    pos_min: float
    pos_max: float
    vel: float
    acc: float
    jerk: float
    continuous: bool = False

    def __post_init__(self) -> None:
        if self.pos_min > self.pos_max:
            raise ValueError(
                f"joint {self.name!r}: pos_min {self.pos_min} > pos_max {self.pos_max}"
            )
        for fld in ("vel", "acc", "jerk"):
            v = getattr(self, fld)
            if not (v > 0):
                raise ValueError(f"joint {self.name!r}: {fld} must be > 0, got {v}")


@dataclass(frozen=True)
class RobotLimits:
    """Ordered collection of :class:`JointLimits` with vectorised accessors."""

    robot: str
    joints: tuple[JointLimits, ...]
    description: str = ""
    units: dict[str, str] = field(default_factory=dict)

    # ---- construction -------------------------------------------------------
    @staticmethod
    def from_preset(name: str) -> RobotLimits:
        """Load a bundled preset (``so101``, ``lekiwi``, ``koch`` ...)."""
        try:
            files = resources.files("demoforge.data.limits")
            text = (files / f"{name}.yaml").read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            avail = ", ".join(list_presets())
            raise ValueError(f"unknown robot preset {name!r}; available: {avail}") from exc
        return RobotLimits._from_mapping(yaml.safe_load(text))

    @staticmethod
    def from_yaml(path: str | Path) -> RobotLimits:
        """Load limits from a user-supplied YAML file (same schema as presets)."""
        text = Path(path).read_text(encoding="utf-8")
        return RobotLimits._from_mapping(yaml.safe_load(text))

    @staticmethod
    def _from_mapping(data: dict) -> RobotLimits:
        joints = tuple(
            JointLimits(
                name=str(j["name"]),
                pos_min=float(j["pos"][0]),
                pos_max=float(j["pos"][1]),
                vel=float(j["vel"]),
                acc=float(j["acc"]),
                jerk=float(j["jerk"]),
                continuous=bool(j.get("continuous", False)),
            )
            for j in data["joints"]
        )
        if not joints:
            raise ValueError("limits file declares no joints")
        return RobotLimits(
            robot=str(data.get("robot", "unknown")),
            joints=joints,
            description=str(data.get("description", "")),
            units=dict(data.get("units", {})),
        )

    def with_urdf_positions(self, urdf_path: str | Path) -> RobotLimits:
        """Override position (and velocity, if present) bounds from a URDF.

        Acceleration and jerk are kept from this preset, since URDF carries neither.
        Requires the optional ``[urdf]`` extra (``yourdfpy``). Joints are matched by name.
        """
        try:
            import yourdfpy
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "URDF parsing needs the optional extra: pip install 'demoforge[urdf]'"
            ) from exc
        model = yourdfpy.URDF.load(str(urdf_path))
        by_name = {j.name: j for j in model.robot.joints}
        new = []
        for jl in self.joints:
            uj = by_name.get(jl.name)
            if uj is None or uj.limit is None:
                new.append(jl)
                continue
            pos_min = float(uj.limit.lower) if uj.limit.lower is not None else jl.pos_min
            pos_max = float(uj.limit.upper) if uj.limit.upper is not None else jl.pos_max
            vel = float(uj.limit.velocity) if uj.limit.velocity else jl.vel
            new.append(JointLimits(jl.name, pos_min, pos_max, vel, jl.acc, jl.jerk, jl.continuous))
        return RobotLimits(self.robot, tuple(new), self.description, self.units)

    # ---- vectorised views ---------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.joints)

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def vel(self) -> np.ndarray:
        return np.array([j.vel for j in self.joints], dtype=float)

    @property
    def acc(self) -> np.ndarray:
        return np.array([j.acc for j in self.joints], dtype=float)

    @property
    def jerk(self) -> np.ndarray:
        return np.array([j.jerk for j in self.joints], dtype=float)

    @property
    def pos_bounds(self) -> np.ndarray:
        """Shape (n, 2): [min, max] per joint."""
        return np.array([[j.pos_min, j.pos_max] for j in self.joints], dtype=float)

    @property
    def continuous_mask(self) -> np.ndarray:
        return np.array([j.continuous for j in self.joints], dtype=bool)


def list_presets() -> list[str]:
    """Names of bundled robot presets."""
    try:
        files = resources.files("demoforge.data.limits")
        return sorted(p.name[:-5] for p in files.iterdir() if p.name.endswith(".yaml"))
    except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover
        return []
