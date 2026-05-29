"""demoforge command-line interface (Typer).

Commands:
  demoforge limits   — show a robot's bundled joint limits
  demoforge inspect  — report demo health for a dataset (no write)
  demoforge process  — re-time a dataset and emit a new one + a health sidecar
  demoforge doctor   — report the runtime environment (backends / optional extras)
"""

from __future__ import annotations

import typer

from .engine.limits import RobotLimits, list_presets
from .health import write_sidecar
from .io import list_episodes, read_episode, read_info, write_retimed_dataset
from .ir import RetimeConfig
from .pipeline import retime_episode

app = typer.Typer(add_completion=False, help="black . for robot demonstrations.")


def _load_limits(robot: str | None, limits_file: str | None) -> RobotLimits:
    if limits_file:
        return RobotLimits.from_yaml(limits_file)
    if robot:
        return RobotLimits.from_preset(robot)
    raise typer.BadParameter("provide --robot <preset> or --limits <file.yaml>")


def _parse_speeds(speeds: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(x) for x in speeds.split(",") if x.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"could not parse --speeds {speeds!r}") from exc
    if not vals:
        raise typer.BadParameter("--speeds must list at least one positive number")
    return vals


@app.command()
def limits(
    robot: str = typer.Option(..., "--robot", "-r", help="preset name, e.g. so101"),
) -> None:
    """Print the bundled joint limits for a robot preset."""
    lim = RobotLimits.from_preset(robot)
    typer.echo(f"# {lim.robot}: {lim.description}")
    typer.echo(f"# units: {lim.units}")
    typer.echo(f"{'joint':<20}{'pos_min':>9}{'pos_max':>9}{'vel':>8}{'acc':>8}{'jerk':>9}  cont")
    for j in lim.joints:
        typer.echo(
            f"{j.name:<20}{j.pos_min:>9.3f}{j.pos_max:>9.3f}{j.vel:>8.2f}"
            f"{j.acc:>8.2f}{j.jerk:>9.2f}  {j.continuous}"
        )


@app.command()
def inspect(
    dataset: str = typer.Argument(..., help="path to a LeRobotDataset v3 root"),
    robot: str = typer.Option(None, "--robot", "-r"),
    limits_file: str = typer.Option(None, "--limits", help="custom limits YAML"),
    episode: int = typer.Option(None, "--episode", "-e", help="single episode index"),
    preserve_contact: str = typer.Option("gripper", "--preserve-contact"),
) -> None:
    """Report demo health for a dataset without writing anything."""
    lim = _load_limits(robot, limits_file)
    eps = [episode] if episode is not None else list_episodes(dataset)
    cfg = RetimeConfig(preserve_contact=preserve_contact or None, speeds=(1.0,))
    for ep in eps:
        raw = read_episode(dataset, ep)
        _, health = retime_episode(raw, lim, cfg, source_kind="real")
        h = health
        typer.echo(
            f"ep {h.episode_index:>4}: frames {h.frames_in} viol {h.limit_violations_before}"
            f"->{h.limit_violations_after}  jerk {h.max_jerk_before:.0f}->{h.max_jerk_after:.0f}"
            f"  path_dev {h.path_deviation_max:.4f}  contacts {h.contact_segments_locked}"
            f"  [{h.retime_outcome}] {h.triage['suggestion']} {h.triage['flags']}"
        )


@app.command()
def process(
    dataset: str = typer.Argument(..., help="path to a LeRobotDataset v3 root"),
    out: str = typer.Option(..., "--out", "-o", help="output dataset root"),
    robot: str = typer.Option(None, "--robot", "-r"),
    limits_file: str = typer.Option(None, "--limits", help="custom limits YAML"),
    mode: str = typer.Option("keep_count", "--mode", help="keep_count | resample"),
    backend: str = typer.Option("topp", "--retime", help="topp | numpy"),
    speeds: str = typer.Option("1.0", "--speeds", help="comma list, e.g. 0.8,1.0,1.2"),
    preserve_contact: str = typer.Option("gripper", "--preserve-contact"),
    target_fps: float = typer.Option(None, "--fps", help="resample mode target fps"),
    health_out: str = typer.Option(None, "--health-out", help="sidecar JSONL path"),
) -> None:
    """Re-time every episode of a dataset and emit a new dataset + a health sidecar."""
    lim = _load_limits(robot, limits_file)
    cfg = RetimeConfig(
        mode=mode,
        backend=backend,
        speeds=_parse_speeds(speeds),
        preserve_contact=preserve_contact or None,
        target_fps=target_fps,
    )
    info = read_info(dataset)
    eps = list_episodes(dataset)
    all_results = []
    healths = []
    out_fps = float(target_fps or info.get("fps") or 0.0)
    for ep in eps:
        raw = read_episode(dataset, ep)
        results, health = retime_episode(raw, lim, cfg, source_kind="real")
        all_results.extend(results)
        healths.append(health)
        if out_fps <= 0:
            out_fps = results[0].fps
    write_retimed_dataset(out, all_results, fps=out_fps, joint_names=lim.names, source_info=info)
    if health_out:
        write_sidecar(health_out, healths)
    typer.echo(
        f"forged {len(all_results)} episode(s) from {len(eps)} source episode(s) -> {out}"
        + (f"  (health: {health_out})" if health_out else "")
    )


@app.command()
def doctor() -> None:
    """Report the runtime environment and available backends/extras."""
    typer.echo(f"demoforge presets: {', '.join(list_presets())}")

    def _probe(mod: str) -> str:
        try:
            __import__(mod)
            return "available"
        except ImportError:
            return "MISSING"

    typer.echo(f"  toppra  (topp backend) : {_probe('toppra')}")
    typer.echo(f"  pyarrow (dataset I/O)  : {_probe('pyarrow')}")
    typer.echo(f"  lerobot (canonical emit): {_probe('lerobot')}")
    typer.echo(f"  yourdfpy (URDF limits) : {_probe('yourdfpy')}")
    if _probe("toppra") == "MISSING":
        typer.echo("  -> install toppra, or use --retime numpy")
    if _probe("lerobot") == "MISSING":
        typer.echo("  -> torch-free parquet I/O is used; install demoforge[lerobot] for finalize()")


def main() -> None:  # pragma: no cover - console entrypoint shim
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
