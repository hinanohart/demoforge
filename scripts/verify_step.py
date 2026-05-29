"""Lightweight self-check used for /compact-resume sanity and as a CI smoke.

Exit 0 means: the package imports, the engine produces a dynamically-feasible re-timing on a
synthetic episode, the result is deterministic, and the health sidecar carries no aggregate
score. This is intentionally fast and dependency-light (no real dataset, no network).
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    from demoforge import RetimeConfig, retime_episode
    from demoforge.engine import RobotLimits, check_feasible
    from demoforge.io import make_mock_episode

    limits = RobotLimits.from_preset("so101")
    raw = make_mock_episode(
        n_joints=limits.n, joint_names=limits.names, pos_bounds=limits.pos_bounds, seed=0
    )
    for mode in ("keep_count", "resample"):
        results, health = retime_episode(raw, limits, RetimeConfig(mode=mode, speeds=(1.0,)))
        rep = check_feasible(results[0].timestamps, results[0].actions, limits, tol=0.05)
        if not rep.dynamic_feasible:
            print(f"FAIL: {mode} emitted a dynamically-infeasible trajectory", file=sys.stderr)
            return 1
        r2, _ = retime_episode(raw, limits, RetimeConfig(mode=mode, speeds=(1.0,)))
        if not np.array_equal(results[0].actions, r2[0].actions):
            print(f"FAIL: {mode} is not deterministic", file=sys.stderr)
            return 1
        if "quality_score" in health.to_dict():
            print("FAIL: health record carries an aggregate quality_score", file=sys.stderr)
            return 1
    print("verify_step OK: feasible + deterministic + no aggregate score (keep_count, resample)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
