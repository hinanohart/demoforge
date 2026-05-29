"""S6 real-measurement: run the re-timer on data and record measured metrics to results/.

By default this runs on **synthetic** mock episodes (no real LeRobot dataset is bundled), so
the metrics demonstrate the geometric/kinematic claim only and are labelled ``mode=synthetic``.
Point ``--dataset`` at a real LeRobotDataset v3 root to measure on real data (labelled
``mode=real``). No number is ever hand-written into the README; gen_readme_numbers.py reads
this file.

Usage: python scripts/measure.py --date 2026-05-29 [--dataset PATH] [--robot so101]
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from demoforge import RetimeConfig, retime_episode
from demoforge.engine import RobotLimits, check_feasible
from demoforge.io import make_mock_episode, read_episode


def _episodes(args, limits):
    if args.dataset:
        from demoforge.io import list_episodes

        return [
            (read_episode(args.dataset, e), "real") for e in list_episodes(args.dataset)
        ], "real"
    eps = [
        make_mock_episode(
            n_frames=90,
            n_joints=limits.n,
            fps=20.0,
            seed=s,
            joint_names=limits.names,
            pos_bounds=limits.pos_bounds,
        )
        for s in range(args.n)
    ]
    return [(e, "synthetic") for e in eps], "synthetic"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="ISO date stamp (injected; clock not read)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--robot", default="so101")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results/v0.1.0a1.json")
    args = ap.parse_args()

    limits = RobotLimits.from_preset(args.robot)
    episodes, mode = _episodes(args, limits)
    cfg = RetimeConfig(mode="keep_count", speeds=(0.8, 1.0, 1.2))

    viol_before = viol_after = pos_before = pos_after = 0
    jerk_before = jerk_after = 0.0
    path_dev = 0.0
    contact_flagged = 0
    determinism_ok = True

    for raw, _ in episodes:
        results, health = retime_episode(raw, limits, cfg, source_kind=mode)
        base = next(r for r in results if r.speed == 1.0)
        rep_after = check_feasible(base.timestamps, base.actions, limits)
        viol_before += health.limit_violations_before
        viol_after += rep_after.n_dynamic_violations
        pos_before += health.position_violations_before
        pos_after += rep_after.n_pos_violations
        jerk_before = max(jerk_before, health.max_jerk_before)
        jerk_after = max(jerk_after, rep_after.max_jerk)
        path_dev = max(path_dev, health.path_deviation_max)
        contact_flagged += health.contact_segments_locked
        # determinism: re-run and compare bit-exact
        r2, _ = retime_episode(raw, limits, cfg, source_kind=mode)
        b2 = next(r for r in r2 if r.speed == 1.0)
        if not (
            np.array_equal(base.actions, b2.actions)
            and np.array_equal(base.timestamps, b2.timestamps)
        ):
            determinism_ok = False

    out = {
        "version": "0.1.0a1",
        "mode": mode,
        "date": args.date,
        "robot": args.robot,
        "dataset_n": len(episodes),
        "backend": cfg.backend,
        "hw": platform.machine(),
        "os": platform.system(),
        "python": platform.python_version(),
        "seed": 0,
        "disclaimer": (
            "Metrics demonstrate geometric/kinematic feasibility of the emitted trajectory only. "
            "Mode=synthetic means data is reproducible mock teleop, not a real robot dataset; "
            "no claim is made about downstream training outcomes."
            if mode == "synthetic"
            else "Measured on a real LeRobotDataset v3; geometric/kinematic feasibility only."
        ),
        "metrics": {
            "limit_violations_before": int(viol_before),
            "limit_violations_after": int(viol_after),
            "position_violations_before": int(pos_before),
            "position_violations_after": int(pos_after),
            "max_jerk_before": round(float(jerk_before), 3),
            "max_jerk_after": round(float(jerk_after), 3),
            "path_deviation_max": round(float(path_dev), 6),
            "determinism_bit_exact": bool(determinism_ok),
            "contact_segments_flagged": int(contact_flagged),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
